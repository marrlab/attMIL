import os
import pickle
import numpy as np
import sklearn.metrics as metrics

what = "MILSILatt"
type = "smp" #type = "smp"



classes = []
with open("data-features/classes.txt") as f:
    clsdata = f.readlines()
    for cls in clsdata:
        classes.append(cls.strip("\n"))

data = []
for fold in range(3):
    with open("evals/" + what + "-" + type + "-bag-test-" + str(fold) + ".pkl","rb") as f:
        data.append(pickle.load(f))


accuracy = np.zeros((5,3))
precision = np.zeros((5,3))
recall = np.zeros((5,3))
f1 = np.zeros((5,3))
auc_roc = np.zeros((5,3))

auc_pr = np.zeros((5,3,len(classes)))

for run in range(5):
    for fold in range(3):
        [res, gt] = data[fold][run]
        res = np.array(res).squeeze()
        gt = np.array(gt).squeeze()

        gt0 = [np.argmax(g) for g in gt]
        res0 = [np.argmax(r) for r in res]

        accuracy[run][fold] = metrics.accuracy_score(gt0, res0)
        t = dict()
        p = dict()
        r = dict()
        for i  in range (len(classes)):
            r = res[:,i]
            g = gt[:, i]
            t[i] = metrics.average_precision_score(gt[:, i], res[:, i])

        for i in range(len(classes)):
            p, r, _ = metrics.precision_recall_curve(gt[:, i],res[:, i])
            auc_pr[run][fold][i] = metrics.auc(r,p)

        precision[run][fold] = np.mean([metrics.average_precision_score(gt[:, i], res[:, i]) for i in range (len(classes))])
        recall[run][fold] = metrics.recall_score(gt0,res0, average="macro")
        f1[run][fold] = metrics.f1_score(gt0,res0, average="weighted")

        auc_roc[run][fold] = metrics.roc_auc_score(gt,res)


pra_med = np.median(accuracy)
print("median: " + str(pra_med))
per_run_accuracy = np.mean(accuracy,axis=1)
acc_mean = np.mean(per_run_accuracy)
acc_std = np.std(per_run_accuracy)
print("ACCURACY: \t" + str('%.4f'% acc_mean) + "+-" + str('%.4f'% acc_std))

per_run_map = np.mean(precision,axis=1)
map_mean = np.mean(per_run_map)
map_std = np.std(per_run_map)
print("mAP: \t" + str('%.4f'% map_mean) + "+-" + str('%.4f'% map_std))


per_run_recall = np.mean(recall,axis=1)
recall_mean = np.mean(per_run_recall)
recall_std = np.std(per_run_recall)
print("RECALL: \t" + str('%.4f'% recall_mean) + "+-" + str('%.4f'% recall_std))

prf_med = np.median(f1)
print(prf_med)
per_run_f1 = np.mean(f1,axis=1)
f1_mean = np.mean(per_run_f1)
f1_std = np.std(per_run_f1)
print("F1: \t" + str('%.4f'% f1_mean) + "+-" + str('%.4f'% f1_std))


per_run_auc_roc = np.mean(auc_roc,axis=1)
auc_roc_mean = np.mean(per_run_auc_roc)
auc_roc_std = np.std(per_run_auc_roc)
print("AUC ROC: \t" + str('%.4f'% auc_roc_mean) + "+-" + str('%.4f'% auc_roc_std))


#
# with open("auprc"+ what + ".pkl", "wb") as f:
#     pickle.dump(auc_pr,f)
#
# exit()

print ("---------------")
for i in range(len(classes)):
    a = auc_pr[:][:][i]
    per_run_auprc = np.mean(auc_pr[:][:][i], axis=1)
    auprc_mean = np.mean(per_run_auprc)
    auprc_std = np.std(per_run_auprc)
    print("AUPRC[" + classes[i] + "]:\t " + str('%.4f'% auprc_mean) + "±" + str('%.4f' % auprc_std))