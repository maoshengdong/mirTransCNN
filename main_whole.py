import numpy as np
import pandas as pd
import torch

from Evaluation import testModel
from precessor import import_data, import_test_data, fold5CV
from train import Train

MAX_LEN = 400
DIM_ENC = 16


def Main():
  # train
  pos_file = "data/whole/train/pos/whole_train_pos.csv"
  neg_file = "data/whole/train/neg/whole_train_neg.csv"
  signal = 'train'
  x_train_dataset, y_train_dataset, feature_train_dataset, bn_size = \
    import_data(pos_file, neg_file, MAX_LEN, DIM_ENC,signal)
  model = Train(x_train_dataset, y_train_dataset, feature_train_dataset, 0, signal, bn_size)
  model_path = "model/whole/model_" + signal + ".pkl"
  torch.save(model.state_dict(), model_path)
  print("The model is saved as", model_path, "in the current directory.")

  # validation
  pos_file = "data/whole/test/pos/whole_test_pos.csv"
  neg_file = "data/whole/test/neg/whole_test_neg.csv"
  x_test_dataset, y_test_dataset, feature_test_dataset = \
    import_test_data(pos_file, neg_file,MAX_LEN, DIM_ENC)
  flag = "test"
  sensitivity, specifity, f1_score, mcc, accuracy, conf_matrix, roc_auc,gmean,ppv,pr_auc = \
    testModel(model_path, x_test_dataset, y_test_dataset, feature_test_dataset, flag, bn_size)
  write_to_file(model_path, sensitivity, specifity, accuracy, f1_score, mcc, roc_auc)
  print("test_Accuracy:", accuracy)
  print("test_F1 Score:", f1_score)
  print("test_auROC:", roc_auc)
  print("test_MCC:", mcc)
  print("test_Sensitivity (Recall):", sensitivity)
  print("test_Specificity:", specifity)
  print("test_Gmean:", gmean)
  print("test_PPV:", ppv)
  print("test_pr_auc:", pr_auc)
  print("Confusion Matrix:")
  print(conf_matrix)

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
    Main()
    print("finished!")