from re import X
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
torch.set_default_dtype(torch.float64)
#Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.LeakyReLU(True)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        # x = self.tanh(x)
        return x.permute(0, 2, 1)  # Permute back


class Generator(nn.Module): #对每个时间点的真假作出判断
    def __init__(self, win_size, latent_dim, input_c, dropout=0.2):
        super(Generator, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = self.n_feats//2+1
        self.n = self.n_feats# * self.win_size
        self.norm1 = nn.LayerNorm(self.n)
        self.norm2 = nn.LayerNorm(self.n)
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvLayer(input_c, 3)
        )
        # self.conv1d = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Conv1d(self.win_size,self.n_hidden, 3, padding=1), nn.LeakyReLU(True),
        #     nn.Conv1d(self.n_hidden,self.n_hidden, 3, padding=1), nn.LeakyReLU(True),
        #     nn.Conv1d(self.n_hidden,self.win_size, 3, padding=1), nn.Tanh(),
        # ) 
        self.discriminator = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            # nn.Linear(self.n_hidden, self.win_size), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        ) 

    def forward(self, d): #(b,1,n)
            # print(d.shape)
            # validity = self.conv1d(d.view(d.shape[0],*(self.win_size, self.n_feats)))#(b,w,n)
            # d = self.norm1(d)
            # validity = self.conv(d)
            # print('a',validity.shape,validity.view(validity.shape[0],*(self.win_size, self.n_feats)).shape)
            validity = self.discriminator(d)#(b,1,n)#.contiguous().view(validity.shape[0],-1))#(b,w,n)
            # validity = self.norm2(validity)
            return validity#(b,1,n).view(validity.shape[0],*(self.win_size, self.n_feats)) #(b,w)

class Discriminator(nn.Module): #对每个时间点的真假作出判断
    def __init__(self, win_size, input_c, dropout=0.2):
        super(Discriminator, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = self.n_feats//2+1
        self.n = self.n_feats# * self.win_size
        self.norm2 = nn.LayerNorm(self.n)
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvLayer(input_c, 3)
        )
        # self.conv1d = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Conv1d(self.win_size,self.n_hidden, 3, padding=1), nn.LeakyReLU(True),
        #     nn.Conv1d(self.n_hidden,self.n_hidden, 3, padding=1), nn.LeakyReLU(True),
        #     nn.Conv1d(self.n_hidden,self.win_size, 3, padding=1), nn.Tanh(),
        # ) 
        self.discriminator = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            # nn.Linear(self.n_hidden, self.win_size), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        ) 

    def forward(self, d): #(b,1,n)
            # print(d.shape)
            # validity = self.conv1d(d.view(d.shape[0],*(self.win_size, self.n_feats)))#(b,w,n)
            validity = self.conv(d)
            # print('a',validity.shape,validity.view(validity.shape[0],*(self.win_size, self.n_feats)).shape)
            validity = self.discriminator(validity)#(b,1,n)#.contiguous().view(validity.shape[0],-1))#(b,w,n)
            # validity = self.norm2(validity)
            return validity#(b,1,n).view(validity.shape[0],*(self.win_size, self.n_feats)) #(b,w)




## LSTM_AD Model
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super(LSTM_AD, self).__init__()
        self.name = 'LSTM_AD'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(3*feats, self.n_hidden)
        self.lstm2 = nn.LSTM(3*feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(2 * self.n_feats, 2 * self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (torch.rand(1, 2, self.n_hidden, dtype=torch.float64).to(device), torch.randn(1, 2, self.n_hidden, dtype=torch.float64).to(device))
        hidden2 = (torch.rand(1, 2, self.n_feats, dtype=torch.float64).to(device), torch.randn(1, 2, self.n_feats, dtype=torch.float64).to(device))
        outputs = []
        for i, g in enumerate(x):
            # g = g.reshape(1, 2, -1)
            # print(g.shape)
            out, hidden = self.lstm(g.reshape(1, 2, -1), hidden)
            out, hidden2 = self.lstm2(g.reshape(1, 2, -1), hidden2)
            # print(out.shape)
            out = self.fcn(out.reshape(-1))
            # print(out.shape)
            # outputs.append(2 * out.view(-1))
            outputs.append(1 * out.reshape(-1))
        # print(outputs)
        v = torch.stack(outputs)
        # print(v.shape)
        v = v.view(v.shape[0],*(2, self.n_feats))#(0-1)
        # print(v.shape)
        return v#torch.stack(outputs)


class Predictor(nn.Module):

    def __init__(self, win_size, input_c, dropout=0.2):
        super(Predictor, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = self.n_feats//2+1
        self.n = self.n_feats# * self.win_size
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvLayer(input_c, 3)
        )
        self.gru = GRULayer(3 * self.n_feats, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, self.n_feats, forecast_n_layers, dropout)


    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        print('x',x.shape)
        x = self.conv(x)
        _, h_end = self.gru(x)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp
        print('h_end',h_end.shape)
        predictions = self.forecasting_model(h_end)
        print('predictions',predictions.shape)
        return predictions


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer
        return out, h

class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)