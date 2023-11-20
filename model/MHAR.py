import math
from math import sqrt
import torch
import torch.nn as nn
import numpy as np


torch.set_default_dtype(torch.float64)
class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


def TemporalcorrelationLayer(x):
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if(j <= i):
                    matrix[i][j] = np.correlate(data[i, :], data[j, :])
                # matrix[i][j] = cosine_similarit(np.array(data[i, :]), np.array(data[j, :]))
                # matrix[i][j] = pearsonr(data[:, i], data[:, j])[0]
                # if math.isnan(matrix[i][j]):
                #     matrix[i][j] = 0

        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float64)

    attention = attention.to(device)
    h = torch.sigmoid(torch.matmul(attention, x))  # (b, n, k)

    return h
def FeaturecorrelationLayer(x):
   # print(f'x={x.shape}')
    use_cuda = True  #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    matrix_all = []
    y = x.data.cpu().numpy()

    for k in range(y.shape[0]):
        data = y[k]
        matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i <= j):
                    matrix[i][j] = np.inner(data[:, i], data[:, j])
                    # matrix[i][j] = cosine_similarit(np.array(data[:, i]), np.array(data[:, j]))
                    #  matrix[i][j] = pearsonr(data[:, i], data[:, j])[0]
                    #  if math.isnan(matrix[i][j]):
                    #      matrix[i][j] = 0
                else:
                    break
        matrix = matrix / data.shape[0]
        matrix_all.append(matrix)
    attention = torch.from_numpy(np.array(matrix_all))
    attention = attention.to(dtype=torch.float64)
    attention=attention.to(device)
   # print(attention.shape)
    h = torch.sigmoid(torch.matmul(attention, x.permute(0, 2, 1)))
    #print(f'h={h.shape}')
    return h.permute(0, 2, 1)


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


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out


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

        layers.append(nn.Linear(hid_dim, out_dim * 2))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)


        return self.layers[-1](x)
def Denoising(train):
    use_cuda=True #
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    io_all = []
    for i in range(train.shape[0]):
        data = train[i]
        data = data.data.cpu().numpy()
        io_time = []
        for j in range(data.shape[1]):
            x = data[:, j]
            #x = x.data.cpu().numpy()
            f = np.fft.rfft(x)
            yf_abs = np.abs(f)
            indices = yf_abs > yf_abs.mean()  # filter out those value under 300
            yf_clean = indices * f
            new_f_clean = np.fft.irfft(yf_clean)
            io_time.append(new_f_clean)
        io_time = np.array(io_time)
        io_all.append(io_time)
    io_all = np.array(io_all)
    io_all = torch.from_numpy(np.array(io_all))
    io_all = io_all.to(dtype=torch.float64)
    io_all = io_all.permute(0, 2, 1)
    io_all = io_all.to(device)
    return io_all


class AR(nn.Module):

    def __init__(self, window):
        super(AR, self).__init__()
        self.linear = nn.Linear(window, 2)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)

        return x


class MHSA(nn.Module):
    def __init__(self, num_heads, dim):
        super().__init__()

        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.num_heads = num_heads

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        #print(v.shape)
        return v




class MHSA_AR(nn.Module):
    """ CGNN-MHSA-AR model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function
        d_k (int): d_model / n_head
        d_v (int): d_model / n_head
        d_model (int): outputs of dimension
        n_head (int): num of Multi-head
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        gru_n_layers=2,
        gru_hid_dim=150,
        forecast_n_layers=2,
        forecast_hid_dim=150,
        dropout=0.4,
        alpha=0.2,

    ):
        super(MHSA_AR, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        #
        # self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        # self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)

        self.multiheadattention=MHSA(n_features,3*n_features)
        self.gru = GRULayer(3* n_features, gru_hid_dim, gru_n_layers, dropout)

        self.ar=AR(window_size)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        #self.forecasting_model = Forecasting_Model(3*n_features*window_size, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        #self.linear=nn.Linear(n_features,out_dim)
    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        gamma=0.5
        x=Denoising(x)
        h_a = self.ar(x)
        h_a = h_a.reshape(x.shape[0], -1)


        x = self.conv(x)
        # h_feat = self.feature_gat(x)
        # h_temp = self.temporal_gat(x)
        h_feat=FeaturecorrelationLayer(x)
        h_temp=TemporalcorrelationLayer(x)
        #h_cat = torch.cat([x,  h_temp], dim=2)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)   (256,100,38*3)
        #print(h_cat.shape)
        h_in=self.multiheadattention(h_cat)
        #print(h_in.shape)
        #h_in=self.mhsa(h_cat)
        #print(h_in.shape)
        #print(h_in_r.shape)
        #h_in = h_in.view(x.shape[0], -1)   #
        #print(h_in.shape)
        _, h_end = self.gru(h_in)

        h_end = h_end.reshape(x.shape[0], -1)   # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end)
        # print(predictions.shape)
        # print(h_a.shape)
        predictions_a=gamma*predictions+(1-gamma)*h_a

        return predictions_a






