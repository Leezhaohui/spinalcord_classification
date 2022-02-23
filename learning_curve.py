from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import os
import copy

def calc_acc_sen_spe_ppv_npv_auc(labels, pres, cut_off=0.5):
    labels, pres = copy.deepcopy(labels), copy.deepcopy(pres)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
    roc_auc = auc(false_positive_rate, true_positive_rate)
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
    sen = right1 / num1
    spe = right0 / num0
    if pre_num1 != 0:
        ppv = right1 / pre_num1
    else:
        ppv = 0
    if pre_num0 != 0:
        npv = right0 / pre_num0
    else:
        npv = 0
    return acc, sen, spe, ppv, npv, roc_auc

if __name__ == "__main__":
    results_dir = "../results/pkl_temp"
    trainx = []
    trainy = []
    valx = []
    valy = []
    task = "astr_epen"
    for file in os.listdir(results_dir):
        if task in file and "train" in file:
            x = file.split("_")[3]
            trainx.append(x)
            with open(os.path.join(results_dir, file), "rb") as f:
                labels, pres = pickle.load(f)
            acc, sen, spe, ppv, npv, roc_auc = calc_acc_sen_spe_ppv_npv_auc(labels, pres)
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
            # for j in range(len(thresholds)):
            #     print(1-false_positive_rate[j], true_positive_rate[j], thresholds[j])
            # roc_auc = auc(false_positive_rate, true_positive_rate)
            trainy.append(roc_auc)
        elif task in file and "val" in file:
            x = file.split("_")[3]
            valx.append(x)
            with open(os.path.join(results_dir, file), "rb") as f:
                labels, pres = pickle.load(f)
            acc, sen, spe, ppv, npv, roc_auc = calc_acc_sen_spe_ppv_npv_auc(labels, pres)
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
            # for j in range(len(thresholds)):
            #     print(1-false_positive_rate[j], true_positive_rate[j], thresholds[j])
            # roc_auc = auc(false_positive_rate, true_positive_rate)
            valy.append(roc_auc)

    plt.plot(trainx, trainy, label="training set auc")
    plt.plot(valx, valy, label="validation set auc")
    plt.grid()

    plt.title("learning curve")
    plt.legend(loc="best")

    plt.ylabel("auc")
    plt.xlabel("training ratio")
    plt.show()