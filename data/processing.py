import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS
from numpy.linalg import norm

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={"ndc": "category"})

    med_pd.drop(columns=['stoptime', 'drug_type', 'formulary_drug_cd', 'gsn', 'poe_id', 'poe_seq',
                         'prod_strength', 'form_rx', 'doses_per_24_hrs', 'dose_val_rx', 'order_provider_id',
                         'dose_unit_rx', 'form_val_disp', 'form_unit_disp', 'route', 'pharmacy_id'], axis=1,
                inplace=True)
    # subject_id, hadm_id, starttime, drug, ndc
    med_pd.drop(index=med_pd[med_pd["ndc"] == "0"].index, axis=0, inplace=True)
    med_pd.fillna(method="pad", inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["starttime"] = pd.to_datetime(med_pd["starttime"], format="%Y-%m-%d %H:%M:%S")
    med_pd.sort_values(by=["subject_id", "hadm_id", "starttime"], inplace=True)
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[["ndc", "drug"]].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

    return atc3toDrugDict


def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[["name", "moldb_smiles"]].values:
        if type(smiles) == type("a"):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]

    return atc3tosmiles

# medication mapping
def codeMapping2atc4(med_pd):
    with open(ndc2RXCUI_file, "r") as f:
        ndc2rxnorm = eval(f.read())
    med_pd["rxcui"] = med_pd["ndc"].map(ndc2rxnorm)
    med_pd = med_pd.dropna(subset=['rxcui'], how='any')  # 删除空行

    rxnorm2atc = pd.read_csv(RXCUI2atc4_file)
    rxnorm2atc = rxnorm2atc.drop(columns=["YEAR", "MONTH", "NDC"])
    rxnorm2atc.drop_duplicates(subset=["RXCUI"], inplace=True)
    rxnorm2atc = rxnorm2atc.rename(columns={'RXCUI': 'rxcui', 'ATC4': 'atc'})

    rxnorm2atc['rxcui'] = rxnorm2atc['rxcui'].astype('str')  # 改变数据类型为string
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=["rxcui"])
    med_pd.drop(columns=["ndc", "rxcui"], inplace=True)
    med_pd = med_pd.rename(columns={"atc": "ndc"})
    med_pd["ndc"] = med_pd["ndc"].map(lambda x: x[:4])  # 使用药物前4位ATC编码
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


# visit >= 2
def process_visit_lg2(med_pd):
    """筛除admission次数小于两次的患者数据"""
    a = med_pd[['subject_id', 'hadm_id']].groupby(by='subject_id')['hadm_id'].unique().reset_index()
    a['hadm_id_len'] = a['hadm_id'].map(lambda x: len(x))
    a = a[a['hadm_id_len'] > 1]
    return a


# most common medications
def filter_300_most_med(med_pd):
    # 按照NDC出现的次数降序排列，取前300
    med_count = med_pd.groupby(by=['ndc']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['ndc'].isin(med_count.loc[:299, 'ndc'])]
    return med_pd.reset_index(drop=True)


##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['seq_num'], inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['subject_id', 'hadm_id'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['icd_code']).size().reset_index().rename(columns={0: 'count'}).sort_values(
            by=['count'], ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['icd_code'].isin(diag_count.loc[:1999, 'icd_code'])]

        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd


##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'icd_code':'category'})
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['subject_id', 'hadm_id', 'seq_num'], inplace=True)
    pro_pd.drop(columns=['seq_num'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['icd_code']).size().reset_index().rename(columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['icd_code'].isin(pro_count.loc[:1000, 'icd_code'])]

    return pro_pd.reset_index(drop=True)


###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):
    """药物、症状、proc的数据结合"""

    med_pd_key = med_pd[['subject_id', 'hadm_id']].drop_duplicates()
    diag_pd_key = diag_pd[['subject_id', 'hadm_id']].drop_duplicates()
    pro_pd_key = pro_pd[['subject_id', 'hadm_id']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['subject_id', 'hadm_id'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['subject_id', 'hadm_id'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['subject_id', 'hadm_id'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['subject_id','hadm_id'])['icd_code'].unique().reset_index()
    med_pd = med_pd.groupby(by=['subject_id', 'hadm_id'])['ndc'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['subject_id','hadm_id'])['icd_code'].unique().reset_index().rename(columns={'icd_code':'pro_code'})

    med_pd['ndc'] = med_pd['ndc'].map(lambda x: list(x))
    pro_pd['pro_code'] = pro_pd['pro_code'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['subject_id', 'hadm_id'], how='inner')
    data = data.merge(pro_pd, on=['subject_id', 'hadm_id'], how='inner')
    data['ndc_Len'] = data['ndc'].map(lambda x: len(x))
    return data


def statistics(data):
    print('#patients ', data['subject_id'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['icd_code'].values
    med = data['ndc'].values
    pro = data['pro_code'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['subject_id'].unique():
        item_data = data[data['subject_id'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['icd_code']))
            y.extend(list(row['ndc']))
            z.extend(list(row['pro_code']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['subject_id'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)


##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['icd_code'])
        med_voc.add_sentence(row['ndc'])
        pro_voc.add_sentence(row['pro_code'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc}, file=open(vocabulary_file, 'wb'))
    return diag_voc, med_voc, pro_voc


# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    """
    保存list类型的记录
    每一项代表一个患者，患者中有多个visit，每个visit包含三者数组，按顺序分别表示诊断、proc与药物
    存储的均为编号，可以通过voc_final.pkl来查看对应的具体word
    """
    records = []  # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['subject_id'].unique():
        item_df = df[df['subject_id'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['icd_code']])  # 20300, 4589, 42731, 5853, 2851..
            admission.append([pro_voc.word2idx[i] for i in row['pro_code']])  # 4923, 4523
            admission.append([med_voc.word2idx[i] for i in row['ndc']])  # C07A, C03C
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open(ehr_sequence_file, 'wb'))
    return records


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):
    TOPK = 40  # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    pro_voc_size = len(pro_voc.idx2word)
    diag_voc_size = len(diag_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]  # 所有的药物的ATC4
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)  #

    with open(cid2atc6_file, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])

    # 加载DDI数据
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect，也是采取topK的形式
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(
        columns={0: 'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)

    # weighted ehr adj
    # 初始化相关性矩阵和各个部分的子矩阵
    ehr_adj = np.zeros((diag_voc_size + pro_voc_size + med_voc_size, diag_voc_size + pro_voc_size + med_voc_size))
    ehr_adj_med_diag = np.zeros((med_voc_size, diag_voc_size))
    ehr_adj_med_proc = np.zeros((med_voc_size, pro_voc_size))
    ehr_adj_med_med = np.zeros((med_voc_size, med_voc_size))
    ehr_adj_diag_proc = np.zeros((diag_voc_size, pro_voc_size))

    for patient in records:  # 对每个患者
        for adm in patient:  # 对每次就诊
            for i in range(len(adm[2])):  # 对药物列表
                for j in range(len(adm[0])):  # 对诊断列表
                    node1 = adm[2][i]
                    node2 = adm[0][j]
                    ehr_adj_med_diag[node1, node2] += 1  # 药物、诊断相关性矩阵 14,2000

                for k in range(len(adm[1])):  # 对程序列表
                    node1 = adm[2][i]
                    node2 = adm[1][k]
                    ehr_adj_med_proc[node1, node2] += 1  # 药物、程序相关性矩阵 14,5488

                for l in range(i + 1, len(adm[2])):  # 对药物列表
                    node1 = adm[2][i]
                    node2 = adm[2][l]
                    ehr_adj_med_med[node1, node2] += 1
                    ehr_adj_med_med[node2, node1] += 1  # 药物、药物相关性矩阵 14,14

            for m in range(len(adm[0])):  # 对诊断列表
                for n in range(len(adm[1])):  # 对程序列表
                    node1 = adm[0][m]
                    node2 = adm[1][n]
                    ehr_adj_diag_proc[node1, node2] += 1  # 诊断、程序相关性矩阵 2000,5488

    for med, line in enumerate(ehr_adj_med_diag):
        for diag, weight in enumerate(line):
            ehr_adj[diag][med + diag_voc_size + pro_voc_size] = weight
            ehr_adj[med + diag_voc_size + pro_voc_size][diag] = weight

    for med, line in enumerate(ehr_adj_med_proc):
        for proc, weight in enumerate(line):
            ehr_adj[proc + diag_voc_size][med + diag_voc_size + pro_voc_size] = weight
            ehr_adj[med + diag_voc_size + pro_voc_size][proc + diag_voc_size] = weight

    for med, line in enumerate(ehr_adj_med_med):
        for med2, weight in enumerate(line):
            ehr_adj[med + diag_voc_size + pro_voc_size][med2 + diag_voc_size + pro_voc_size] = weight
            ehr_adj[med2 + diag_voc_size + pro_voc_size][med + diag_voc_size + pro_voc_size] = weight

    for diag, line in enumerate(ehr_adj_diag_proc):
        for proc, weight in enumerate(line):
            ehr_adj[diag][proc + diag_voc_size] = weight
            ehr_adj[proc + diag_voc_size][diag] = weight
    # mean_value = np.mean(ehr_adj)
    # std_value = np.std(ehr_adj)
    # normalized_ehr_adj = (ehr_adj - mean_value) / std_value
    # ehr_adj = np.array(ehr_adj)  # 将输入转换为NumPy数组
    # l2_norm = norm(ehr_adj, ord='fro')  # 计算矩阵的L2范数
    # normalized_ehr_adj = ehr_adj / l2_norm  # 将矩阵的每个元素除以L2范数
    # dill.dump(normalized_ehr_adj, open(ehr_adjacency_file, 'wb'))
    dill.dump(ehr_adj, open(ehr_adjacency_file, 'wb'))

    # ddi adj，DDI表是CID编码的，因此需要将CID映射到ACT编码，才能记录数据集中药物之间的冲突信息
    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']

        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:

                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, 'wb'))

    return ddi_adj

def get_ddi_mask(atc42SMLES, med_voc):

    # ATC3_List[22] = {0}
    # ATC3_List[25] = {0}
    # ATC3_List[27] = {0}
    fraction = []
    for k, v in med_voc.idx2word.items():
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet))  # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix, fracSet

if __name__ == '__main__':
    # MIMIC数据文件，分别包括药物、诊断和proc
    med_file = './input/PRESCRIPTIONS.csv'
    diag_file = './input/DIAGNOSES_ICD.csv'
    procedure_file = './input/PROCEDURES_ICD.csv'

    # input auxiliary files
    med_structure_file = "./output/atc32SMILES.pkl"
    RXCUI2atc4_file = "./input/RXCUI2atc4.csv"
    cid2atc6_file = "./input/drug-atc.csv"
    ndc2RXCUI_file = "./input/ndc2RXCUI.txt"
    ddi_file = "./input/drug-DDI.csv"
    drugbankinfo = "./input/drugbank_drugs_info.csv"

    # output files
    ddi_adjacency_file = "./output/ddi_A_final.pkl"
    ehr_adjacency_file = "./output/ehr_adj_final.pkl"
    ehr_sequence_file = "./output/records_final.pkl"
    vocabulary_file = "./output/voc_final.pkl"
    ddi_mask_H_file = "./output/ddi_mask_H.pkl"
    atc3toSMILES_file = "./output/atc3toSMILES.pkl"
    substructure_smiles_file = "./output/substructure_smiles.pkl"

    # for prescription
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)  # 注意这里仅仅是针对med表中出现了两次以上admission的patient
    med_pd = med_pd.merge(med_pd_lg2[['subject_id']], on='subject_id', how='inner').reset_index(drop=True)
    med_pd = codeMapping2atc4(med_pd)
    # med_pd = filter_300_most_med(med_pd)  # ATC04编码的药物总共就14种，不需要再排除了

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3toSMILES = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3toSMILES, open(atc3toSMILES_file, "wb"))
    med_pd = med_pd[med_pd.ndc.isin(atc3toSMILES.keys())]
    print("complete medication processing")

    # for diagnosis
    diag_pd = diag_process(diag_file)
    print('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    print('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    print('complete combining')

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print("obtain ehr sequence data")

    # create ddi adj matrix
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print("obtain ddi adj matrix")

    # get ddi_mask_H
    ddi_mask_H, fracSet = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, "wb"))
    dill.dump(fracSet, open(substructure_smiles_file, "wb"))
    print("obtain ddi mask H")
