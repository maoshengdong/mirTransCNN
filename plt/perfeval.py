import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def perfeval(predictions, Y_test,total_train_loss, total_val_loss,epoch,num_epochs, verbose=0):
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))
    CM = metrics.confusion_matrix(R, class_label, labels=None)

    CM = np.double(CM)
    acc = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])
    se = (CM[0][0]) / (CM[0][0] + CM[0][1])
    sp = (CM[1][1]) / (CM[1][0] + CM[1][1])
    f1 = (2 * CM[0][0]) / (2 * CM[0][0] + CM[0][1] + CM[1][0])
    ppv = (CM[0][0]) / (CM[0][0] + CM[1][0])
    mcc = (CM[0][0] * CM[1][1] - CM[0][1] * CM[1][0]) / np.sqrt((CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]) * (CM[0][1] + CM[1][1]) * (CM[1][0] + CM[1][1]))
    gmean = np.sqrt(se * sp)
    auroc = metrics.roc_auc_score(Y_test[:, 0], predictions[:, 0])
    aupr = metrics.average_precision_score(Y_test[:, 0], predictions[:, 0], average="micro")

    if verbose == 1:
        print(f"Epoch: {epoch}/{num_epochs}","SE:", "{:.3f}".format(se), "SP:", "{:.3f}".format(sp), "F-Score:", "{:.3f}".format(f1), "PPV:",
              "{:.3f}".format(ppv), "gmean:", "{:.3f}".format(gmean), "AUROC:", "{:.3f}".format(auroc), "AUPR:",
              "{:.3f}".format(aupr))
    if verbose == 2:
        print(f"Epoch: {epoch}/{num_epochs}","SE:", "{:.3f}".format(se), "SP:", "{:.3f}".format(sp), "F-Score:", "{:.3f}".format(f1),
              "AUROC:", "{:.3f}".format(auroc), "AUPR:",
              "{:.3f}".format(aupr),"ACC:", "{:.3f}".format(acc),"train_loss:", "{:.3f}".format(total_train_loss),"val_loss:", "{:.3f}".format(total_val_loss))

    return [se, sp, f1, ppv, gmean, auroc, aupr, CM,mcc, acc]


def perfeval1(predictions, Y_test, verbose=0):
    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))
    CM = metrics.confusion_matrix(R, class_label, labels=None)

    CM = np.double(CM)
    acc = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])
    se = (CM[0][0]) / (CM[0][0] + CM[0][1])
    sp = (CM[1][1]) / (CM[1][0] + CM[1][1])
    f1 = (2 * CM[0][0]) / (2 * CM[0][0] + CM[0][1] + CM[1][0])
    ppv = (CM[0][0]) / (CM[0][0] + CM[1][0])
    mcc = (CM[0][0] * CM[1][1] - CM[0][1] * CM[1][0]) / np.sqrt(
        (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]) * (CM[0][1] + CM[1][1]) * (CM[1][0] + CM[1][1]))
    gmean = np.sqrt(se * sp)
    auroc = metrics.roc_auc_score(Y_test[:, 0], predictions[:, 0])
    aupr = metrics.average_precision_score(Y_test[:, 0], predictions[:, 0], average="micro")

    if verbose == 1:
        print("SE:", "{:.3f}".format(se), "SP:", "{:.3f}".format(sp), "F-Score:", "{:.3f}".format(f1),
              "PPV:", "{:.3f}".format(ppv), "gmean:", "{:.3f}".format(gmean), "AUROC:", "{:.3f}".format(auroc),
              "AUPR:", "{:.3f}".format(aupr), "ACC:", "{:.3f}".format(acc))
    if verbose == 2:
        print("SE:", "{:.3f}".format(se), "SP:", "{:.3f}".format(sp), "F-Score:", "{:.3f}".format(f1),
              "AUROC:", "{:.3f}".format(auroc), "AUPR:",
              "{:.3f}".format(aupr),"ACC:", "{:.3f}".format(acc))

    return [se, sp, f1, auroc, aupr, acc]


def wrtrst(filehandle, rst, nfold=0, nepoch=0):
 filehandle.write(str(nfold+1)+" "+str(nepoch+1)+" ")
 filehandle.write("SE: %s SP: %s F-score: %s PPV: %s g-mean: %s AUROC: %s AUPR: %s\n" %
 ("{:.3f}".format(rst[0]),
 "{:.3f}".format(rst[1]),
 "{:.3f}".format(rst[2]),
 "{:.3f}".format(rst[3]),
 "{:.3f}".format(rst[4]),
 "{:.3f}".format(rst[5]),
 "{:.3f}".format(rst[6])))
 filehandle.flush()
 return
def wrtrst(filehandle, rst, nepoch=0):
 # import datetime
 # current_time = datetime.datetime.now()
 #
 # # 格式化为字符串，例如：2023-11-13_14:30:00
 # formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
 # filehandle.write(f"{formatted_time}_miRe2e1_performance:")
 # # fd.write(str(time.time())+model_path + "_performance:")
 # filehandle.write("\n")
 filehandle.write(" "+str(nepoch+1)+" ")
 filehandle.write("SE: %s SP: %s F-score: %s AUROC: %s AUPR: %s ACC: %s\n" %
 ("{:.3f}".format(rst[0]),
 "{:.3f}".format(rst[1]),
 "{:.3f}".format(rst[2]),
 "{:.3f}".format(rst[3]),
 "{:.3f}".format(rst[4]),
 "{:.3f}".format(rst[5])))
 # filehandle.write("\n\n")
 filehandle.flush()
 return

def wrtrst1(filehandle, rst,total_train_loss, total_val_loss, nepoch=0):
 # [se0, sp1, f12, ppv3, gmean4, auroc5, aupr6, CM7,mcc8,acc9]
 filehandle.write(" "+str(nepoch+1)+" ")
 filehandle.write("SE: %s SP: %s F-score: %s AUROC: %s AUPR: %s ACC: %s train_loss: %s val_loss: %s\n" %
 ("{:.3f}".format(rst[0]),
 "{:.3f}".format(rst[1]),
 "{:.3f}".format(rst[2]),
 "{:.3f}".format(rst[5]),
 "{:.3f}".format(rst[6]),
 "{:.3f}".format(rst[9]),
 "{:.3f}".format(total_train_loss),
 "{:.3f}".format(total_val_loss)))
 # filehandle.write("\n\n")
 filehandle.flush()
 return


def mkdir():
     subdirectory = "plots"
     if not os.path.exists(subdirectory):
          os.makedirs(subdirectory)
     return subdirectory


def Plt_ROC(true_labels, predictions,flag):
     subdirectory = mkdir()
     fpr, tpr,threshold = roc_curve(true_labels[:, 0], predictions[:, 0])
     roc_auc = auc(fpr, tpr)
     plt.figure(figsize=(10, 10))
     plt.plot(fpr, tpr, '-', \
              linewidth=2, label='test-AUC:%0.4f)' % roc_auc)
     plt.xlim([0.0, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     # plt.title('Receiver operating characteristic')
     plt.legend(loc="center right")
     plt.savefig(os.path.join(subdirectory,"ROC_curve_" + flag + ".png"))


     print("ROC曲线已保存")


def Plt_CM(predictions, Y_test, flag):
     subdirectory = mkdir()
     class_label = np.uint8(np.argmax(predictions, axis=1))
     R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))
     CM = metrics.confusion_matrix(R, class_label, labels=None)
     CM = np.double(CM)
     # 绘制混淆矩阵
     plt.figure()
     plt.imshow(CM, interpolation='nearest', cmap=plt.cm.Blues)
     plt.title('Confusion Matrix')
     plt.colorbar()
     plt.xticks([0, 1], ['Negative', 'Positive'])
     plt.yticks([0, 1], ['Negative', 'Positive'])
     plt.xlabel('Predicted Label')
     plt.ylabel('True Label')
     plt.savefig(os.path.join(subdirectory,"CM_" + flag + ".png"))



def Plt_PR(predictions,Y_test,flag):
     subdirectory = mkdir()
     # 绘制 PR 曲线
     precision, recall, _ = precision_recall_curve(Y_test[:, 0], predictions[:, 0])
     plt.figure()
     plt.plot(recall, precision, label='PR Curve')
     plt.xlabel('Recall')
     plt.ylabel('Precision')
     plt.title('Precision-Recall (PR) Curve')
     plt.legend()
     plt.savefig(os.path.join(subdirectory,"PR_" + flag + ".png"))



def Plt_F1(predictions, Y_test, flag):
     subdirectory = mkdir()
     # 绘制 F1 曲线
     thresholds = np.linspace(0, 1, 100)
     f1_scores = []
     Epochs = range(1, len(predictions) + 1)
     for threshold in Epochs:
          class_label = np.uint8(predictions[:, 0] >= threshold)
          CM = metrics.confusion_matrix(Y_test[:, 0], class_label)
          f1 = (2 * CM[0][0]) / (2 * CM[0][0] + CM[0][1] + CM[1][0])
          f1_scores.append(f1)
     plt.figure()
     plt.plot(Epochs, f1_scores, label='F1 Score')
     plt.xlabel('Epochs')
     plt.ylabel('F1 Score')
     plt.title('F1 Score vs. Epochs')
     plt.legend()
     plt.savefig(os.path.join(subdirectory,"F1-Score_" + flag + ".png"))



def Plt_ACC(predictions, Y_test, flag):
     subdirectory = mkdir()
     # 绘制 ACC 曲线
     thresholds = np.linspace(0, 1, 100)
     acc_scores = []
     Epochs = range(1, len(predictions) + 1)
     for threshold in Epochs:
          class_label = np.uint8(predictions[:, 0] >= threshold)
          CM = metrics.confusion_matrix(Y_test[:, 0], class_label)
          acc = (CM[0][0] + CM[1][1]) / (CM[0][0] + CM[0][1] + CM[1][0] + CM[1][1])
          acc_scores.append(acc)
     plt.figure()
     plt.plot(Epochs, acc_scores, label='Accuracy')
     plt.xlabel('Epochs')
     plt.ylabel('Accuracy')
     plt.title('Accuracy vs. Epochs')
     plt.legend()
     plt.savefig(os.path.join(subdirectory,"ACC_" + flag + ".png"))



def Plt_SE_SP(predictions, Y_test, flag):
     subdirectory = mkdir()
     # 绘制 sensitivity 和 specificity 曲线
     thresholds = np.linspace(0, 1, 100)
     sensitivity_scores = []
     specificity_scores = []
     Epochs = range(1, len(predictions) + 1)
     for threshold in Epochs:
          class_label = np.uint8(predictions[:, 0] >= threshold)
          CM = metrics.confusion_matrix(Y_test[:, 0], class_label)
          sensitivity = (CM[0][0]) / (CM[0][0] + CM[0][1])
          specificity = (CM[1][1]) / (CM[1][0] + CM[1][1])
          sensitivity_scores.append(sensitivity)
          specificity_scores.append(specificity)
     plt.figure()
     plt.plot(Epochs, sensitivity_scores, label='Sensitivity')
     plt.plot(Epochs, specificity_scores, label='Specificity')
     plt.xlabel('Epochs')
     plt.ylabel('Score')
     plt.title('Sensitivity and Specificity vs. Epochs')
     plt.legend()
     plt.savefig(os.path.join(subdirectory,"SE_SP_" + flag + ".png"))




def Plt_Valid_Acc(val_acc_scores,flag,fold,signal):
    subdirectory = mkdir()
    epochs = range(1, len(val_acc_scores) + 1)
    plt.figure()
    plt.plot(epochs, val_acc_scores, label='Validation Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    # if signal == 'cv':
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + flag + fold +".png"))
    # else:
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + flag + ".png"))
    plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + signal +".png"))



def Plt_Train_Acc(train_acc_scores,flag,fold,signal):
    subdirectory = mkdir()
    epochs = range(1, len(train_acc_scores) + 1)
    plt.figure()
    plt.plot(epochs, train_acc_scores, label='Train Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    # if signal == 'cv':
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + flag +fold+ ".png"))
    # else:
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + flag + ".png"))
    plt.savefig(os.path.join(subdirectory, "ACC_epoch_" + flag+ ".png"))



def Plt_SE_SP(predictions, Y_test, flag):
     subdirectory = mkdir()
     # 绘制 sensitivity 和 specificity 曲线
     thresholds = np.linspace(0, 1, 100)
     sensitivity_scores = []
     specificity_scores = []
     Epochs = range(1, len(predictions) + 1)
     for threshold in Epochs:
          class_label = np.uint8(predictions[:, 0] >= threshold)
          CM = metrics.confusion_matrix(Y_test[:, 0], class_label)
          sensitivity = (CM[0][0]) / (CM[0][0] + CM[0][1])
          specificity = (CM[1][1]) / (CM[1][0] + CM[1][1])
          sensitivity_scores.append(sensitivity)
          specificity_scores.append(specificity)
     plt.figure()
     plt.plot(Epochs, sensitivity_scores, label='Sensitivity')
     plt.plot(Epochs, specificity_scores, label='Specificity')
     plt.xlabel('Epochs')
     plt.ylabel('Score')
     plt.title('Sensitivity and Specificity vs. Epochs')
     plt.legend()
     plt.savefig(os.path.join(subdirectory,"SE_SP_" + flag + ".png"))

def Plt_loss(train_losses, epochs,fold,signal):
    subdirectory = mkdir()
    min_loss_index = np.argmin(train_losses)
    min_loss_value = train_losses[min_loss_index]
    # 绘制损失图
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    # 在图中标记最低的损失值
    # 标记最低损失及其位置
    min_loss = min(train_losses)
    min_loss_epoch = epochs[train_losses.index(min_loss)]
    plt.annotate(f'Min Loss: {min_loss:.4f}', xy=(min_loss_epoch, min_loss),
                 xytext=(min_loss_epoch, min_loss + 0.1),
                 arrowprops=dict(facecolor='black', arrowstyle='-', linewidth=0.5), horizontalalignment='right')
    # 在最低损失值所在的位置画一条垂直于 x 轴的直线
    plt.axvline(x=min_loss_epoch, color='gray', linestyle='--')
    plt.savefig(os.path.join(subdirectory, 'training_loss_plot'+signal+'.png'))


def Plt_SE(sensitivity, flag,fold,signal):
    subdirectory = mkdir()
    epochs = range(1, len(sensitivity) + 1)
    plt.figure()
    plt.plot(epochs, sensitivity, label='Sensitivity')
    plt.xlabel('epochs')
    plt.ylabel('sensitivity')
    plt.title('Sensitivity vs. epochs')
    plt.legend()
    plt.savefig(os.path.join(subdirectory, "Sensitivity_" + signal+".png"))


def Plt_SP(specificity, flag,fold,signal):
    subdirectory = mkdir()
    epochs = range(1, len(specificity) + 1)
    plt.figure()
    plt.plot(epochs, specificity, label='Specificity')
    plt.xlabel('epochs')
    plt.ylabel('specificity')
    plt.title('Specificity vs. Epochs')
    plt.legend()
    plt.savefig(os.path.join(subdirectory, "Specificity_" + signal + ".png"))



def Plt_F1_Score(f1_scores, flag,fold,signal):
    subdirectory = mkdir()
    # 绘制 F1 曲线
    epochs = range(1, len(f1_scores) + 1)
    plt.figure()
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Epochs')
    plt.legend()
    plt.savefig(os.path.join(subdirectory, "F1-Score_" + signal+".png"))


def Plt_gmean(g_means, flag,fold,signal):
    subdirectory = mkdir()
    # 绘制 F1 曲线
    epochs = range(1, len(g_means) + 1)
    plt.figure()
    plt.plot(epochs, g_means, label='g-mean')
    plt.xlabel('epoch')
    plt.ylabel('g-mean')
    plt.title('G-mean vs. Epochs')
    plt.legend()
    plt.savefig(os.path.join(subdirectory, "G-mean_" + signal+".png"))



def Plt_MCC(mccs, flag,fold,signal):
    subdirectory = mkdir()
    # 绘制 F1 曲线
    epochs = range(1, len(mccs) + 1)
    plt.figure()
    plt.plot(epochs, mccs, label='mcc')
    plt.xlabel('epoch')
    plt.ylabel('mcc')
    plt.title('MCC vs. Epochs')
    plt.legend()
    plt.savefig(os.path.join(subdirectory, "MCC_" + signal+".png"))



def Plt_PPV(ppv, flag,fold,signal):
    subdirectory = mkdir()
    # 绘制 F1 曲线
    epochs = range(1, len(ppv) + 1)
    plt.figure()
    plt.plot(epochs, ppv, label='ppv')
    plt.xlabel('epoch')
    plt.ylabel('ppv')
    plt.title('PPV vs. Epochs')
    plt.legend()
    # if signal =='cv':
    #     plt.savefig(os.path.join(subdirectory, "PPV_" + flag + fold+".png"))
    # else:
    #     plt.savefig(os.path.join(subdirectory, "PPV_" + flag + ".png"))
    plt.savefig(os.path.join(subdirectory, "PPV_" + signal+".png"))





def Plt_Accuracy_Curve(train_acc_scores, val_acc_scores,fold,signal):
    subdirectory = mkdir()
    epochs = range(1, len(train_acc_scores) + 1)
    plt.figure()
    plt.plot(epochs, train_acc_scores, label='Train Accuracy')
    plt.plot(epochs, val_acc_scores, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    # if signal == 'cv':
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch"+fold+".png"))
    # else:
    #     plt.savefig(os.path.join(subdirectory, "ACC_epoch.png"))
    plt.savefig(os.path.join(subdirectory, "ACC_epoch"+signal+".png"))

