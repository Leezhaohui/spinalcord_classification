import pandas as pd
import os
import pickle

"""
astrocytoma. -------------
60
ependymoma. -------------
23
MS. -------------
30
NMO. -------------
85
"""

def convert_subname(name):
    if "sub" not in name:
        return name
    ex = "sub"
    lat = name[3:]
    lat = str(int(lat))
    new_name = ex + lat
    return new_name

zhongzhang_dic = {}
excel_root = r"D:\work\9.15_backup\jisui\excel_data"
for file in os.listdir(excel_root):
    lesion = file[:-4]
    zhongzhang_dic[lesion] = []
    df = pd.read_excel(os.path.join(excel_root, file))
    values = df.values
    for i in range(values.shape[0]):
        filedir = str(values[i, 0]).strip()
        filedir = convert_subname(filedir)
        zhongzhang = str(values[i, -1]).strip()
        if zhongzhang == "是":
            zhongzhang_dic[lesion].append(filedir)

zhong = []
for key in zhongzhang_dic:
    zhong.extend(zhongzhang_dic[key])

path = r"D:\work\spinal_classification\data\multi_xulie\nii_split\NMO"
for set in os.listdir(path):
    num = 0
    set_path = os.path.join(path, set)
    for filedir in os.listdir(set_path):
        if filedir in zhong:
            num += 1
    print(set, num)

# with open("../zhongzhang_dic.pkl", "wb") as f:
#     pickle.dump(zhongzhang_dic, f)
#
# for lesion in zhongzhang_dic:
#     print(lesion, "-------------")
#     print(len(zhongzhang_dic[lesion]))