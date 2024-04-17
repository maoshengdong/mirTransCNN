import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from model import TransformerCNN
from focal_loss import FocalLoss
from plt.perfeval import perfeval, wrtrst1, Plt_SE, Plt_SP, Plt_F1_Score, Plt_gmean, Plt_MCC, Plt_PPV, Plt_Train_Acc, \
    Plt_Valid_Acc, Plt_Accuracy_Curve, Plt_loss


def Train(x_dataset,y_dataset, f_dataset, i, signal, bn_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    model = TransformerCNN(bn_size,device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # optimizer = optim.RAdam(model.parameters(), lr=5e-3, weight_decay=1e-5)

    x_tensor = torch.Tensor(x_dataset).to(device=device)
    y_tensor = torch.Tensor(y_dataset).to(device=device)
    f_tensor = torch.Tensor(f_dataset).to(device=device)
    dataset = TensorDataset(x_tensor, f_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ce_weights = torch.Tensor([0.75, 0.25]).to(device=device)

    loss_function = FocalLoss(alpha=1, gamma=5, logits=True,
                              coef=ce_weights).to(device=device)

    num_epochs = 150
    WriteFile = open("./TransformerCNN.rst", "a+")
    train_acc_scores = []
    val_acc_scores = []
    threshold = 0.5
    f1_max = 0
    verbose = True
    epochs = []
    train_losses = []
    sentitivitys = []
    specificitys = []
    F1_scores = []
    MCCs = []
    g_means = []
    PPV = []
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_true_labels = []
        for x, f, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x, f)
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_train_loss += loss.item()
            train_predictions.extend(outputs.argmax(1).tolist())
            train_true_labels.extend(y.argmax(1).tolist())

        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        train_acc_scores.append(train_accuracy)

        # scheduler.step()    #sam++ 20231207
        predictions = []
        true_labels = []
        model.eval()
        preall = np.empty((0, 2))
        laball = np.empty((0, 2))
        total_val_loss = 0.0
        with torch.no_grad():
            for x, f, y in val_loader:
                outputs = model(x, f)
                val_loss = loss_function(outputs, y)
                total_val_loss += val_loss.item()
                _, predicted = torch.max(outputs, 1)
                binary_predictions = (outputs[:, 1] >= threshold).long()
                # predictions.extend(outputs.argmax(1).tolist())
                predictions.extend(binary_predictions.tolist())
                true_labels.extend(y.argmax(1).tolist())
                z_np = outputs.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                preall = np.append(preall, z_np, axis=0)
                laball = np.append(laball, y_np, axis=0)
        # true_labels = np.concatenate(true_labels, axis=0)
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_val_loss = total_val_loss / len(val_loader)
        rst = perfeval(preall, laball, total_train_loss, total_val_loss, epoch, num_epochs, verbose=2)
        wrtrst1(WriteFile, rst, total_train_loss, total_val_loss, epoch)
        accuracy = accuracy_score(true_labels, predictions)
        val_acc_scores.append(accuracy)
        sentitivitys.append(rst[0])
        specificitys.append(rst[1])
        F1_scores.append(rst[2])
        PPV.append(rst[3])
        g_means.append(rst[4])
        MCCs.append(rst[8])
        epochs.append(epoch)
        # avg_train_loss = total_train_loss / len(train_loader)
        # print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Val Accuracy: {accuracy}')
        # aucv, f1v, prev, recv = get_error(laball[:, 0], preall[:, 0])
        # early stop
        f1v = rst[2]
        # if f1v > f1_max:
        #     f1_max = f1v
        #     best_epoch = epoch
        #     epoch_max = 0
        #     model_path = "model_" + signal + ".pkl"
        #     torch.save(model.state_dict(), model_path)
        # else:
        #     epoch_max += 1
        # if epoch_max >= 30:
        #     model_path = "model_" + signal + ".pkl"
        #     model.load_state_dict(torch.load(model_path))
        #     if verbose:
        #         print(f"Best epoch {best_epoch}: F1 {f1_max}")
        #     break
    flag = 'valid'
    Plt_loss(train_losses, epochs, i, signal)
    Plt_SE(sentitivitys, flag, i, signal)
    Plt_SP(specificitys, flag, i, signal)
    Plt_F1_Score(F1_scores, flag, i, signal)
    Plt_gmean(g_means, flag, i, signal)
    Plt_MCC(MCCs, flag, i, signal)
    Plt_PPV(PPV, flag, i, signal)
    Plt_Train_Acc(train_acc_scores, 'train', i, signal)
    Plt_Valid_Acc(val_acc_scores, flag, i, signal)
    Plt_Accuracy_Curve(train_acc_scores, val_acc_scores, i, signal)
    return model