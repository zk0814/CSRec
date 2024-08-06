import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.typing import OptTensor, Adj

"""
Our model
"""

class GAT_HO(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=3, negative_slope=0.2, dropout=0.3):
        super(GAT_HO, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=negative_slope, dropout=dropout))
        self.edge_embedding = nn.Embedding(22754, 1)
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None, ddi_weight=None):
        x = self.lin(x)
        edge_weight = self.edge_embedding(edge_weight).squeeze(1)
        ddi_weight = ddi_weight.unsqueeze(1)
        edge_weight = edge_weight - ddi_weight
        head_outputs = [head(x, edge_index, edge_weight) for head in self.heads]
        out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        result = edge_weight * x_j
        return result

class GAT_HE(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=3, negative_slope=0.2, dropout=0.3):
        super(GAT_HE, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=negative_slope, dropout=dropout))
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_embedding = nn.Embedding(22754, in_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        edge_weight = self.edge_embedding(edge_weight).squeeze(1)
        head_outputs = [head(x, edge_index, edge_weight) for head in self.heads]
        out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        return out

    def message(self, x_j, edge_weight):
        result = edge_weight * x_j
        return result


class GraphNet(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device):
        super(GraphNet, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.ehr_adj = ehr_adj
        self.ddi_adj = ddi_adj

        # 嵌入数量+1表示一个padding位置
        self.node_embedding = nn.Embedding(voc_size[0] + voc_size[1] + voc_size[2], emb_dim)

        # 同构图训练
        self.med_med = GAT_HO(emb_dim, emb_dim)

        # 异构图训练
        self.diag_med = GAT_HE(emb_dim, emb_dim)
        self.pro_med = GAT_HE(emb_dim, emb_dim)
        self.diag_pro = GAT_HE(emb_dim, emb_dim)

        # 用来对抗过拟合
        self.dropout = nn.Dropout(p=0.2)  # 0.5

        # 用来链接残差
        self.residual_weight_diag = nn.Parameter(torch.tensor(1.0))
        self.residual_weight_proc = nn.Parameter(torch.tensor(1.0))
        self.residual_weight_med = nn.Parameter(torch.tensor(1.0))

        # 对各种Embedding的初始化
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, adm):
        """将同次就诊，不同医疗实体类型编号合并"""
        adm, new_adm, nodes = self.get_node(adm, self.voc_size)

        """同类型节点交互"""
        # 构建药物-药物的同构图
        x = torch.LongTensor(sorted(nodes)).to(self.device)
        x = self.node_embedding(x)
        if len(new_adm[2]) == 0:  # 当前就诊，不考虑药物
            x1 = self.linear(x)  # 实体embedding
        else:
            # 生成边
            edges = self.build_isomorphic_edges(new_adm[2])
            # 转换边的格式
            edge_index = self.node2complete_graph2coo(nodes, edges)
            # 生成边的权重
            edge_weight = self.edges2weight(edges)
            ddi_weight = self.ddi2weight(edges)
            # 开始训练
            x1 = self.med_med(x, edge_index, edge_weight, ddi_weight)
            x1 /= len(adm[2])

        x1 = F.relu(x1)  # 里面是按照诊断，程序，药物，顺序排列的节点的结果(n*64)

        """异构节点交互，不应该包含同构节点的交互"""
        # 构建诊断-药物的异构图
        edges = self.build_heterogeneous_edges(new_adm, 2, 0)
        if len(edges) != 0:
            edge_index = self.node2complete_graph2coo(nodes, edges)  # 利用图节点和边变成pyg需要的coo模式
            edge_weight = self.edges2weight(edges)  # 从EHR adj中得到相应的边权重
            x2 = self.diag_med(x1, edge_index, edge_weight)  # n*64
            x2 /= len(adm[2])
            x2 = F.relu(x2)
        else:
            x2 = torch.zeros([len(nodes), self.emb_dim], dtype=torch.float).to(self.device)

        # 构建程序-药物的异构图
        edges = self.build_heterogeneous_edges(new_adm, 2, 1)
        if len(edges) != 0:
            edge_index = self.node2complete_graph2coo(nodes, edges)  # 利用图节点和边变成pyg需要的coo模式
            edge_weight = self.edges2weight(edges)
            x3 = self.pro_med(x1, edge_index, edge_weight)  # n*64
            x3 /= len(adm[2])
            x3 = F.relu(x3)
        else:
            x3 = torch.zeros([len(nodes), self.emb_dim], dtype=torch.float).to(self.device)

        # 构建诊断-程序的异构图
        edges = self.build_heterogeneous_edges(new_adm, 0, 1)
        if len(edges) != 0:
            edge_index = self.node2complete_graph2coo(nodes, edges)  # 利用图节点和边变成pyg需要的coo模式
            edge_weight = self.edges2weight(edges)
            x4 = self.diag_pro(x1, edge_index, edge_weight)  # n*64
            x4 /= len(adm[0])
            x4 = F.relu(x4)
        else:
            x4 = torch.zeros([len(nodes), self.emb_dim], dtype=torch.float).to(self.device)

        x_out = x1 + x2 + x3 + x4
        x_out = self.dropout(x_out)

        embedding_trained = [torch.zeros([1, self.emb_dim], dtype=torch.float).to(self.device) for _ in range(3)]

        i = 0
        for diag in adm[0]:
            embedding_trained[0] += x_out[i]
            i += 1
        for pro in adm[1]:
            embedding_trained[1] += x_out[i]
            i += 1

        if len(adm[2]) == 0:
            embedding_trained[2] = torch.zeros((1, self.emb_dim)).to(self.device)
        for med in adm[2]:
            embedding_trained[2] += x_out[i]
            i += 1

        i1 = embedding_trained[0].unsqueeze(0)  # 维度数值相加
        i2 = embedding_trained[1].unsqueeze(0)
        i3 = embedding_trained[2].unsqueeze(0)

        return i1, i2, i3

    def ddi2weight(self, edges):
        ddi_weight = []
        for edge in edges:
            # 映射边的节点编号到ddi_adj中对应位置，获取DDI权重
            ddi_weight.append(self.ddi_adj[edge[0] - self.voc_size[0] - self.voc_size[1]][edge[1] - self.voc_size[0] - self.voc_size[1]])
        ddi_weight = torch.LongTensor(ddi_weight).to(self.device)
        return ddi_weight

    def get_node(self, adm, voc_size):
        """将一次就诊不同类型的节点合并到一个列表"""
        new_adm = [[] for _ in range(3)]
        for diag in adm[0]:  # diag
            new_adm[0].append(diag)
        for pro in adm[1]:  # pro
            new_adm[1].append(pro + voc_size[0])
        for med in adm[2]:  # med
            new_adm[2].append(med + voc_size[0] + voc_size[1])

        adm = [sorted(adm[0]), sorted(adm[1]), sorted(adm[2])]
        new_adm = [sorted(new_adm[0]), sorted(new_adm[1]), sorted(new_adm[2])]
        nodes = sorted(new_adm[0] + new_adm[1] + new_adm[2])

        return adm, new_adm, nodes

    def node2complete_graph2coo(self, nodes, edges):
        """将节点的原始编号转换为PyG库中coo格式的边索引"""
        if len(edges) == 0:  # 边列表为空，即图中没有边
            edge_index = torch.combinations(torch.LongTensor([0])).t().contiguous()
        else:  # 边列表不为空
            voc = {}
            for i, node in enumerate(nodes):
                voc[node] = i  # 原始节点编号为key，PyG格式编号为值，储存为字典
            edge_index = []
            for edge in edges:
                edge_index.append((voc[edge[0]], voc[edge[1]]))
            edge_index = torch.LongTensor(edge_index).t().contiguous()  # 边列表变成coo形式

        edge_index = edge_index.to(self.device)

        return edge_index

    def build_isomorphic_edges(self, nodes):
        """通过一串节点构建一个同构图"""
        edges = []
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2:  # 避免形成自环的边
                    edges.append([node1, node2])  # 创建两个不同节点之间的边

        return edges

    def build_heterogeneous_edges(self, new_adm, type_a, type_b):
        """在两种节点之间建立异构的单向图（同种节点无连接）"""
        edges = []
        for node1 in new_adm[type_a]:
            for node2 in new_adm[type_b]:
                edges.append([node1, node2])

        return edges

    def edges2weight(self, edges):
        edge_weight = []
        for j in range(len(edges)):  # 遍历图中的每条边
            weight = self.ehr_adj[edges[j][0]][edges[j][1]]  # 从ehr_adj中获取边的权重
            edge_weight.append(weight)
        edge_weight = torch.LongTensor(edge_weight).to(self.device)
        return edge_weight  # 返回包含边权重的张量


class CSRec(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, ehr_adj, emb_dim=256, device=torch.device("cpu:0")):
        super(CSRec, self).__init__()
        self.device = device
        self.graph_embeddings = GraphNet(vocab_size, emb_dim, ehr_adj, ddi_adj, device)
        self.encoders_f = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True) for _ in range(3)])  # 双向GRU
        self.encoders_o = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True) for _ in range(3)])  # 双向GRU
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.fo = nn.Linear(emb_dim * 2, 1)  # 双向GRU
        self.op = nn.Linear(emb_dim * 2, emb_dim)  # 双向GRU

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

    def forward(self, patient):
        '''input: [[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [0]],
                   [[1, 3, 9, 10, 6, 5], [0], [0]]]'''
        diag_seq = []
        proc_seq = []
        med_seq = []

        for adm_num, adm in enumerate(patient):
            if adm_num == 0:  # 第一次就诊
                adm_new = adm[:]
                adm_new[2] = []
                diag_embedding, proc_embedding, med_embedding = self.graph_embeddings(adm_new)
            else:  # 多次就诊
                adm_new = adm[:]
                adm_new[2] = patient[adm_num - 1][2][:]
                diag_embedding, proc_embedding, med_embedding = self.graph_embeddings(adm_new)

            diag_seq.append(diag_embedding)
            proc_seq.append(proc_embedding)
            med_seq.append(med_embedding)

        diag_seq = torch.cat(diag_seq, dim=1)  # (1,visit,dim)，一个患者的多次visit
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,visit,dim)，一个患者的多次visit
        med_seq = torch.cat(med_seq, dim=1)  # (1,visit-1,dim)，一个患者的多次visit

        diag_f, _ = self.encoders_f[0](diag_seq)  # diag_f:(1, visit, dim)
        proc_f, _ = self.encoders_f[1](proc_seq)
        med_f, _ = self.encoders_f[2](med_seq)
        diag_o, _ = self.encoders_o[0](diag_seq)  # diag_o:(1, visit, dim)
        proc_o, _ = self.encoders_o[1](proc_seq)
        med_o, _ = self.encoders_o[2](med_seq)

        weight_diag_f = F.tanhshrink(self.fo(diag_f.squeeze(dim=0)))  # (visit, 1)
        weight_proc_f = F.tanhshrink(self.fo(proc_f.squeeze(dim=0)))  # (visit, 1)
        weight_med_f = F.tanhshrink(self.fo(med_f.squeeze(dim=0)))  # (visit-1, 1)
        weight_diag_o = F.tanh(self.op(diag_o.squeeze(dim=0)))  # (visit, dim)
        weight_proc_o = F.tanh(self.op(proc_o.squeeze(dim=0)))  # (visit, dim)
        weight_med_o = F.tanh(self.op(med_o.squeeze(dim=0)))  # (visit-1, dim)

        diag = torch.sum(weight_diag_f * weight_diag_o * diag_seq, dim=1)  # (1, dim)
        proc = torch.sum(weight_proc_f * weight_proc_o * proc_seq, dim=1)  # (1, dim)
        med = torch.sum(weight_med_f * weight_med_o * med_seq, dim=1)  # (1, dim)


        output = self.output(torch.cat([diag, proc, med], dim=-1))

        neg_pred_prob = F.sigmoid(output)  # (1, 14)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return output, batch_neg