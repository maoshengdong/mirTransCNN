import torch.nn as nn
import torch

from positional_encoding import PositionalEncoder


class TransformerCNN(nn.Module):
    def __init__(self,bn_size, device="cpu"):
        super(TransformerCNN, self).__init__()

        self.bn_size = bn_size
        self.dense_size = ((400 + bn_size) // 8) * 128
        self.feature_dim = 64
        self.max_length = 400 + bn_size

        self.positional_encoding = PositionalEncoder(16, self.max_length, device=device)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=8,
            dim_feedforward=self.feature_dim // 2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=3)

        self.dropout = nn.Dropout(0.5)
        # First layer of convolution and max-pooling
        self.conv1 = nn.Conv1d(16, 32, kernel_size=4, stride=1, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second layer of convolution and max-pooling
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Third layer of convolution and max-pooling
        self.conv3 = nn.Conv1d(64, 128, kernel_size=6, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2)



        self.flatten = nn.Flatten()
        # 20231218_sam++ >>
        self.ivar_layers = nn.BatchNorm1d(1)
        self.bn = nn.BatchNorm1d(self.bn_size)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.dense_size, 400)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(400, 100)

        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x, f):
        self.num = f.size(1)
        x = x.permute(0, 2, 1)  # 将通道维度放在最后
        x = x.contiguous()
        f1 = self.bn(f)
        f2 = f1.unsqueeze(1)
        f3 = f2.repeat(1, 16, 1)
        x = torch.cat([x, f3], 2)
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)

        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout3(x)
        x = nn.functional.softmax(self.fc3(x), dim=1)

        return x


if __name__ == "__main__":
    model_with_lstm = CNNModelWithLSTM()
    print(model_with_lstm)
