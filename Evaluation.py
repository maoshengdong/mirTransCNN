"""Evaluate the performance of the trained model using the test dataset
"""
import torch
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, f1_score, \
    matthews_corrcoef, recall_score
from torch.utils.data import DataLoader, TensorDataset
from model import TransformerCNN
from plt.perfeval import perfeval1, wrtrst, Plt_ROC, Plt_CM, Plt_PR, Plt_F1, Plt_ACC, Plt_SE_SP

sys.path.append("../data")

def testModel(model_path,x_test_dataset, y_test_dataset, mfe_test_dataset, Kfold, bn_size):
    print("load the model")
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using device:", device)
        model = TransformerCNN(bn_size,device)
        model.to(device)
        # print(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception:
        print("The model file ", model_path, " doesn't exist!")
        exit(1)
    x_tensor = torch.Tensor(x_test_dataset).to(device=device)
    y_tensor = torch.Tensor(y_test_dataset).to(device=device)
    mfe_tensor = torch.Tensor(mfe_test_dataset).to(device=device)
    test_dataset = TensorDataset(x_tensor, y_tensor, mfe_tensor)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    predictions = []
    true_labels = []
    preall = np.empty((0, 2))
    laball = np.empty((0, 2))
    # all_roc_auc_scores = []
    WriteFile = open("./test_performance.rst", "a+")
    with torch.no_grad():
        for inputs, labels, mfes in test_loader:    #20231218_sam++

            outputs = model(inputs, mfes) #20231218_sam++
            _, predicted = torch.max(outputs, 1)

            predictions.extend(outputs.argmax(1).tolist())
            true_labels.extend(labels.argmax(1).tolist())
            z_np = outputs.detach().cpu().numpy()
            y_np = labels.detach().cpu().numpy()
            preall = np.append(preall, z_np, axis=0)
            laball = np.append(laball, y_np, axis=0)

        rst = perfeval1(preall, laball, verbose=2)
        wrtrst(WriteFile, rst)
        # np.save('predict_result_'+Kfold+'.npy', preall)
        # np.save('true_labels_'+Kfold+'.npy', laball)



    accuracy = accuracy_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    sensitivity = recall_score(true_labels, predictions)

    conf_matrix = confusion_matrix(true_labels, predictions)

    tn, fp, fn, tp = conf_matrix.ravel()

    specificity = tn / (tn + fp)
    gmean = np.sqrt(sensitivity*specificity)
    ppv = tp / (tp + fp)

    Plt_ROC(laball, preall, Kfold)
    Plt_CM(preall, laball, Kfold)
    Plt_PR(preall, laball, Kfold)

    return sensitivity, specificity, f1, mcc, accuracy,conf_matrix,roc_auc,gmean,ppv,pr_auc

