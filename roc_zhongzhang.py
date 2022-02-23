from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import os

if __name__ == "__main__":
    results_dir = "../results/pkl_zhongzhang"
    for file in os.listdir(results_dir):
        if "test" in file and "astr_epen" in file:
            label = file[:-4]
            with open(os.path.join(results_dir, file), "rb") as f:
                labels, pres = pickle.load(f)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, pres)
            for j in range(len(thresholds)):
                print(1-false_positive_rate[j], true_positive_rate[j], thresholds[j])
            roc_auc = auc(false_positive_rate, true_positive_rate)
            plt.plot(false_positive_rate, true_positive_rate, label="%s AUC=%0.3f" % (label, roc_auc))
    plt.plot([0, 1], [0, 1], "r--")

    plt.title("roc")
    plt.legend(loc="lower right")

    plt.ylabel("sensitivity")
    plt.xlabel("specificity")
    plt.show()