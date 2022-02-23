import numpy as np
import random
import os
import sys
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
sys.path.append(os.path.split(os.path.split(os.getcwd())[0])[0])
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_rel
import copy

def calc_acc_sen_spe_ppv_npv_auc(labels, pres, cut_off=0.5, pian=False):
    labels, pres = copy.deepcopy(labels), copy.deepcopy(pres)
    if not pian:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
        roc_auc = auc(false_positive_rate, true_positive_rate)
    else:
        roc_auc = 12345
    pres1 = []
    for i in pres:
        if i >= cut_off:
            pres1.append(1)
        else:
            pres1.append(0)
    pres = pres1
    labels = [int(i) for i in labels]
    num = len(labels)
    num0 = labels.count(0)
    num1 = labels.count(1)
    pre_num0 = pres.count(0)
    pre_num1 = pres.count(1)
    right = 0
    right0 = 0
    right1 = 0
    for i in range(len(labels)):
        if labels[i] == pres[i]:
            right += 1
            if labels[i] == 1:
                right1 += 1
            elif labels[i] == 0:
                right0 += 1


    acc = right / num
    if num1 != 0:
        sen = right1 / num1
    else:
        sen = 12345
    if num0 != 0:
        spe = right0 / num0
    else:
        spe = 12345
    if pre_num1 != 0:
        ppv = right1 / pre_num1
    else:
        ppv = 12345
    if pre_num0 != 0:
        npv = right0 / pre_num0
    else:
        npv = 12345
    return acc, sen, spe, ppv, npv, roc_auc

def calc_pvalue(lis1, lis2):
    """
    配对样本T检验，计算两组数据p值
    """
    ttest, pvalue = ttest_rel(lis1, lis2)
    avg1, avg2 = sum(lis1) / len(lis1), sum(lis2) / len(lis2)
    # print("组1均值：", avg1)
    # print("组2均值：", avg2)
    # print("p值：", pvalue)
    return pvalue

def plot_roc(labels, pros):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pros)
    for i in range(len(thresholds)):
        print(1-false_positive_rate[i], true_positive_rate[i], thresholds[i])

def get_calc_acc_sen_spe_ppv_npv_auc_bootstrap_lis(labels_, pres_, cutoff=0.5, pian=False):
    n = len(pres_)

    indexs0 = [i for i in range(n) if labels_[i] == 0]
    indexs1 = [i for i in range(n) if labels_[i] == 1]

    labels_, pres_ = np.array(labels_).astype(np.int), np.array(pres_).astype(np.float)
    accs, sens, spes, ppvs, npvs, aucs = [], [], [], [], [], []
    for _ in range(1000):
        temp_labels, temp_pres = [], []
        for j in range(len(indexs0)):
            index = random.choice(indexs0)
            temp_labels.append(labels_[index])
            temp_pres.append(pres_[index])
        for j in range(len(indexs1)):
            index = random.choice(indexs1)
            temp_labels.append(labels_[index])
            temp_pres.append(pres_[index])

        acc, sen, spe, ppv, npv, roc_auc = calc_acc_sen_spe_ppv_npv_auc(list(temp_labels), list(temp_pres), cutoff, pian)
        accs.append(acc)
        sens.append(sen)
        spes.append(spe)
        ppvs.append(ppv)
        npvs.append(npv)
        aucs.append(roc_auc)
    accs, sens, spes, ppvs, npvs, aucs = sorted(accs), sorted(sens), sorted(spes), sorted(ppvs), sorted(npvs), sorted(
        aucs)
    return accs, sens, spes, ppvs, npvs, aucs

def get_CI_result_from_lis(accs, sens, spes, ppvs, npvs, aucs):
    column = ["acc 95CI", "sen 95CI", "spe 95CI", "ppv 95CI", "npv 95CI", "auc 95CI"]
    accCI = "%.3f [%.3f, %.3f]" % (accs[500], accs[25], accs[975])
    senCI = "%.3f [%.3f, %.3f]" % (sens[500], sens[25], sens[975])
    speCI = "%.3f [%.3f, %.3f]" % (spes[500], spes[25], spes[975])
    ppvCI = "%.3f [%.3f, %.3f]" % (ppvs[500], ppvs[25], ppvs[975])
    npvCI = "%.3f [%.3f, %.3f]" % (npvs[500], npvs[25], npvs[975])
    aucCI = "%.3f [%.3f, %.3f]" % (aucs[500], aucs[25], aucs[975])
    res = [accCI, senCI, speCI, ppvCI, npvCI, aucCI]
    return res

def get_mean_std(lis):
    return np.mean(lis), np.std(lis)

def get_meanvalue_result_from_lis(accs, sens, spes, ppvs, npvs, aucs):
    column = ["acc_mean_std", "sen_mean_std", "spe_mean_std", "ppv_mean_std", "npv_mean_std", "auc_mean_std"]
    acc_mean, acc_std = get_mean_std(accs)
    sen_mean, sen_std = get_mean_std(sens)
    spe_mean, spe_std = get_mean_std(spes)
    ppv_mean, ppv_std = get_mean_std(ppvs)
    npv_mean, npv_std = get_mean_std(npvs)
    auc_mean, auc_std = get_mean_std(aucs)
    acc = "%.3f ± %.3f" % (acc_mean, acc_std)
    sen = "%.3f ± %.3f" % (sen_mean, sen_std)
    spe = "%.3f ± %.3f" % (spe_mean, spe_std)
    ppv = "%.3f ± %.3f" % (ppv_mean, ppv_std)
    npv = "%.3f ± %.3f" % (npv_mean, npv_std)
    auc = "%.3f ± %.3f" % (auc_mean, auc_std)
    res = [acc, sen, spe, ppv, npv, auc]
    res = pd.DataFrame([res], columns=column)
    res.to_excel(excel_path, index=False)

def get_pvalue_from_lis(accs1, sens1, spes1, ppvs1, npvs1, aucs1, accs2, sens2, spes2, ppvs2, npvs2, aucs2):
    column = ["acc_p", "sen_p", "spe_p", "ppv_p", "npv_p", "auc_p"]
    acc_p = "%.3f" % calc_pvalue(accs1, accs2)
    sen_p = "%.3f" % calc_pvalue(sens1, sens2)
    spe_p = "%.3f" % calc_pvalue(spes1, spes2)
    ppv_p = "%.3f" % calc_pvalue(ppvs1, ppvs2)
    npv_p = "%.3f" % calc_pvalue(npvs1, npvs2)
    auc_p = "%.3f" % calc_pvalue(aucs1, aucs2)
    res = [acc_p, sen_p, spe_p, ppv_p, npv_p, auc_p]
    return res

if __name__ == "__main__":

    task = "astr_epen"


    pkl_path = "../results/pkl_zhongzhang"
    excel_dir = "../results/excel_zhongzhang"
    excel_name = task + ".xlsx"
    excel_path = os.path.join(excel_dir, excel_name)

    files = []
    for file in os.listdir(pkl_path):
        if task in file:
            files.append(file)

    data = []
    column = ["group", "accuracy", "sensitivity", "specificity", "ppv", "npv", "auc"]
    #cutoffs = [0.4680, 0.6426, 0.5, 0.5, 0.4958, 0.4595]
    #cutoffs = [0.7164, 0.6073, 0.4309, 0.5707, 0.4532, 0.6011]
    #cutoffs = [0.5264, 0.5968, 0.5254, 0.5249, 0.3841, 0.6001]

    # lises = []

    for i in range(len(files)):
        file = files[i]
        if task in file:
            print(file, "------------------------")
            file_path = os.path.join(pkl_path, file)
            with open(file_path, "rb") as f:
                res = pickle.load(f)
            labels, pres = res[0], res[1]

            if sum(labels) == 0 or sum(labels) == len(labels):
                cutoff = 0.5
                accs, sens, spes, ppvs, npvs, aucs = get_calc_acc_sen_spe_ppv_npv_auc_bootstrap_lis(labels, pres, cutoff=cutoff, pian=True)
            else:
                yue = 0
                false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
                for i in range(len(thresholds)):
                    tmpyue = 1 - false_positive_rate[i] + true_positive_rate[i]
                    if tmpyue > yue:
                        yue = tmpyue
                        cutoff = thresholds[i]
                accs, sens, spes, ppvs, npvs, aucs = get_calc_acc_sen_spe_ppv_npv_auc_bootstrap_lis(labels, pres,
                                                                                                    cutoff=cutoff)

            # lises.append([accs, sens, spes, ppvs, npvs, aucs])

            res = get_CI_result_from_lis(accs, sens, spes, ppvs, npvs, aucs)

            name = file.split("_")[2] + "_" + file.split("_")[3]
            one_data = []
            one_data.append(name)
            one_data.extend(res)
            data.append(one_data)

    # for i in files:
    #     print(i)
    #
    # for i in range(3):
    #     for j in range(3):
    #         lis1 = lises[i]
    #         lis2 = lises[j]
    #         lis = lis1 + lis2
    #         print(get_pvalue_from_lis(*lis), end=" ")
    #     print()

    data = pd.DataFrame(data, columns=column)
    data.to_excel(excel_path, index=False)



