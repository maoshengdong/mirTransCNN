import numpy as np
import torch

from Evaluation import testModel
from precessor import fold5CV
from train import Train

MAX_LEN = 400
DIM_ENC = 16


def cv():
  sensitivity_scores = []
  specificity_scores = []
  f1_scores = []
  mcc_scores = []
  accuracy_scores = []
  roc_auc_scores = []
  confusion_matrices = []
  gmeans = []
  ppv_scores = []
  pr_auc_scores = []


  for i in range(5):
    # i = 4
    print("fold = ", i + 1)
    train_file = f"data/human/cv/train/train_fold_{i + 1}.csv"
    val_file = f"data/human/cv/validation/validation_fold_{i + 1}.csv"
    signal = "cv_human_" + str(i)

    x_train_list, y_train_list, x_validation_list, y_validation_list, \
      feature_train_list, feature_validation_list, bn_size = fold5CV(train_file, val_file, MAX_LEN, DIM_ENC,signal)

    model = Train(x_train_list, y_train_list, feature_train_list,i,signal, bn_size)
    model_path = "model/human/model_" + signal + ".pkl"
    torch.save(model.state_dict(), model_path)
    print(model_path, " is stored in the current directory.")

    sensitivity, specifity, f1_score, mcc, accuracy, conf_matrix, roc_auc,gmean,ppv,pr_auc = \
      testModel(model_path, x_validation_list, y_validation_list, feature_validation_list, signal, bn_size)
    write_to_file(model_path, sensitivity, specifity, accuracy, f1_score, mcc, roc_auc)
    print("val_Accuracy:", accuracy)
    print("val_F1 Score:", f1_score)
    print("test_auROC:", roc_auc)
    print("val_MCC:", mcc)
    print("val_Sensitivity (Recall):", sensitivity)
    print("val_Specificity:", specifity)
    print("val_Gmean:",gmean)
    print("val_PPV:", ppv)
    print("val_pr_ruc:", pr_auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    confusion_matrices.append(conf_matrix)
    sensitivity_scores.append(sensitivity)
    specificity_scores.append(specifity)
    f1_scores.append(f1_score)
    mcc_scores.append(mcc)
    accuracy_scores.append(accuracy)
    roc_auc_scores.append(roc_auc)
    gmeans.append(gmean)
    pr_auc_scores.append(pr_auc)
    ppv_scores.append(ppv)


  avg_sensitivity = np.mean(sensitivity_scores)
  avg_specificity = np.mean(specificity_scores)
  avg_f1 = np.mean(f1_scores)
  avg_mcc = np.mean(mcc_scores)
  avg_accuracy = np.mean(accuracy_scores)
  avg_roc_auc = np.mean(roc_auc_scores)
  avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
  avg_gmean = np.mean(gmeans)
  avg_ppv = np.mean(ppv_scores)
  avg_pr_ruc = np.mean(pr_auc_scores)


  print("平均灵敏度（Sensitivity）：", avg_sensitivity)
  print("平均特异度（Specificity）：", avg_specificity)
  print("平均gmean:", avg_gmean)
  print("平均F1分数（F1 Score）：", avg_f1)
  print("平均Matthews相关系数（MCC）：", avg_mcc)
  print("平均准确率（Accuracy）：", avg_accuracy)
  print("平均ROC AUC：", avg_roc_auc)
  print("平均ruc：", avg_pr_ruc)
  print("平均ppv：", avg_ppv)
  print("平均混淆矩阵（Confusion Matrix）：")
  print(avg_confusion_matrix)

def write_to_file(model_path, sensitivity, specifity, accuracy, f1_score, mcc, roc_auc):
  """ write the performace parameters to file
  """
  fd = open("model_performance", "a+")
  import datetime
  current_time = datetime.datetime.now()

  formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
  fd.write(f"{formatted_time} {model_path} _performance:")

  fd.write("\n")
  fd.write("accuracy:{}\n".format(accuracy))
  fd.write("f1_score:{}\n".format(f1_score))
  fd.write("auROC:{}\n".format(roc_auc))
  fd.write("sensitivity:{}\n".format(sensitivity))
  fd.write("specifity:{}\n".format(specifity))
  fd.write("mcc:{}\n".format(mcc))
  fd.write("\n\n")
  fd.close()


if __name__ == "__main__":
    cv()
    print("finished!")