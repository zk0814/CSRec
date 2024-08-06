import argparse
import math
import time
from collections import defaultdict

import dill
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import os
from models import CSRec
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

torch.manual_seed(1203)

# Training settings
parser = argparse.ArgumentParser()
# 模式参数
# parser.add_argument("--Test", action="store_true", default=False, help="test mode")
parser.add_argument("--Test", action="store_true", default=True, help="test mode")

# 设备参数
parser.add_argument("--cuda", type=int, default=1, help="which cuda")
parser.add_argument("--resume_path_trained", default="saved/trained_model.pt", help="trained model")

# 训练参数
parser.add_argument("--ddi", action="store_true", default=True, help="using ddi")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--regular", type=float, default=0.008, help="regularization parameter")  # 0.005
parser.add_argument("--target_ddi", type=float, default=0.06, help="target ddi")
parser.add_argument("--T", type=float, default=2.0, help="T")
parser.add_argument("--decay_weight", type=float, default=0.85, help="decay weight")
parser.add_argument("--dim", type=int, default=64, help="dimension")

args = parser.parse_args()


# evaluate
def eval(model, data_eval, voc_size, device):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]

    # """自己加入一些loss来看"""
    loss_bce, loss_multi, loss = [[] for _ in range(3)]

    med_cnt, visit_cnt = 0, 0

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        if len(input) < 2:
            continue
        for adm_idx, adm in enumerate(input):
            target_output, _ = model(input[: adm_idx + 1])

            # """自己加的loss，在输出时候用来看loss的改变，不训练"""
            loss_bce_target = np.zeros((1, voc_size[2]))
            loss_bce_target[:, adm[2]] = 1

            loss_multi_target = np.full((1, voc_size[2]), -1)
            for idx, item in enumerate(adm[2]):
                loss_multi_target[0][idx] = item

            with torch.no_grad():
                loss_bce1 = F.binary_cross_entropy_with_logits(
                    target_output, torch.FloatTensor(loss_bce_target).to(device)
                ).cpu()
                loss_multi1 = F.multilabel_margin_loss(
                    F.sigmoid(target_output), torch.LongTensor(loss_multi_target).to(device)
                ).cpu()
                loss1 = 0.95 * loss_bce1.item() + 0.05 * loss_multi1.item()

            loss_bce.append(loss_bce1)
            loss_multi.append(loss_multi1)
            loss.append(loss1)
            """"""

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint("\rtest step: {} / {}".format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path="../data/output/ddi_A_final.pkl")

    llprint(
        "\nDDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f},"
        "AVG_Loss: {:.4f}, AVG_MED: {:.4f}\n".format(
            ddi_rate,
            np.mean(ja),
            np.mean(prauc),
            np.mean(avg_p),
            np.mean(avg_r),
            np.mean(avg_f1),
            np.mean(loss),
            med_cnt / visit_cnt,
        )
    )

    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        np.mean(loss),
        med_cnt / visit_cnt,
    )


def main():
    # load data
    data_path = "../data/output/records_final.pkl"  # 0,1,2,3...编码后
    voc_path = "../data/output/voc_final.pkl"  # 原编码
    ddi_adj_path = "../data/output/ddi_A_final.pkl"
    ddi_mask_path = "../data/output/ddi_mask_H.pkl"
    ehr_adj_path = "../data/output/ehr_adj_final.pkl"

    device = torch.device('cpu')
    data = dill.load(open(data_path, "rb"))
    voc = dill.load(open(voc_path, "rb"))  # diag_voc, med_voc, pro_voc
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, "rb"))
    ddi_mask_H = dill.load(open(ddi_mask_path, "rb"))

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    print(diag_voc.idx2word)
    print(pro_voc.idx2word)
    print(med_voc.idx2word)
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")


    # 数据集划分
    # data = data[:500]  # 少量数据，测试实验是否跑通
    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))  # 词表大小(2000, 5477, 14)

    # 药物推荐模型
    model = CSRec(voc_size, ddi_adj, ddi_mask_H, ehr_adj, emb_dim=args.dim, device=device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path_trained, 'rb')))
        model.to(device=device)
        tic = time.time()
        result = []
        for _ in range(10):
            test_sample = np.random.choice(data_test, size=round(len(data_test) * 0.8), replace=True)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, _, avg_med = eval(model, test_sample, voc_size, device)
            result.append([ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med])

        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print(outstring)
        print('test time: {}'.format(time.time() - tic))
        return

    print("***********training*********")

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=args.lr)
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    # 训练模式
    EPOCH = 15
    for epoch in range(EPOCH):
        tic = time.time()
        print("\nepoch {} --------------------------".format(epoch))
        model.train()

        for step, patient in enumerate(data_train):
            loss = 0
            if len(patient) < 2:
                continue
            for idx, adm in enumerate(patient):
                seq_input = patient[: idx + 1]
                loss_bce_target = np.zeros((1, voc_size[2]))
                loss_bce_target[:, adm[2]] = 1

                loss_multi_target = np.full((1, voc_size[2]), -1)
                for idx, item in enumerate(adm[2]):
                    loss_multi_target[0][idx] = item

                target_output1, loss_ddi = model(seq_input)
                # 计算损失
                loss_bce = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_bce_target).to(device))
                loss_multi = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss_multi_target).to(device))
                if args.ddi:
                    target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                    target_output1[target_output1 >= 0.5] = 1
                    target_output1[target_output1 < 0.5] = 0
                    y_label = np.where(target_output1 == 1)[0]
                    current_ddi_rate = ddi_rate_score([[y_label]], path="../data/output/ddi_A_final.pkl")
                    if current_ddi_rate <= args.target_ddi:
                        loss = 0.9 * loss_bce + 0.1 * loss_multi
                    else:
                        rnd = np.exp((args.target_ddi - current_ddi_rate) / args.T)
                        if np.random.rand(1) < rnd:
                            loss = loss_ddi
                        else:
                            loss = 0.9 * loss_bce + 0.1 * loss_multi
                else:
                    loss = 0.9 * loss_bce + 0.1 * loss_multi

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint("\rtraining step: {} / {}".format(step, len(data_train)))
        args.T *= args.decay_weight

        print()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_loss, avg_med = eval(model, data_eval, voc_size, device)
        print("training time: {}, test time: {}".format(time.time() - tic, time.time() - tic2))

        history["ja"].append(ja)
        history["ddi_rate"].append(ddi_rate)
        history["avg_p"].append(avg_p)
        history["avg_r"].append(avg_r)
        history["avg_f1"].append(avg_f1)
        history["prauc"].append(prauc)
        history["med"].append(avg_med)

        if epoch >= 5:
            print(
                "ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}".format(
                    np.mean(history["ddi_rate"][-5:]),
                    np.mean(history["med"][-5:]),
                    np.mean(history["ja"][-5:]),
                    np.mean(history["avg_f1"][-5:]),
                    np.mean(history["prauc"][-5:]),
                )
            )


        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            best_model = model

        print("best_epoch: {}".format(best_epoch))

    torch.save(best_model.state_dict(), args.resume_path_trained)


if __name__ == "__main__":
    main()
