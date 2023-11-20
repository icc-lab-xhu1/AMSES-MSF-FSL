import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

#Use GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
class Generator(nn.Module):
    def __init__(self, win_size, latent_dim, output_c):
        super(Generator, self).__init__()
        self.win_size = win_size
        self.n_feats = output_c
        self.n_hidden = 16
        self.n = self.n_feats * win_size
        latent_dim = latent_dim
        self.generator = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(latent_dim, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
       

    def forward(self, z):
        # Generate
        d = self.generator(z)#.view(1,-1))
        d = d.view(d.shape[0],*(self.win_size, self.n_feats))

        return d

class Discriminator(nn.Module):
    def __init__(self, win_size, input_c, output_c):
        super(Discriminator, self).__init__()
        self.win_size = win_size
        self.n_feats = input_c
        self.n_hidden = 16
        self.n = self.n_feats * self.win_size
        self.discriminator = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        ) 

    def forward(self, d):
            validity = self.discriminator(d.view(d.shape[0],-1))#.view(1,-1))
            return validity
        
# class RNNDecoder(nn.Module):
#     """GRU-based Decoder network that converts latent vector into output
#     :param in_dim: number of input features
#     :param n_layers: number of layers in RNN
#     :param hid_dim: hidden size of the RNN
#     :param dropout: dropout rate
#     """

#     def __init__(self, in_dim, hid_dim, n_layers, dropout):
#         super(RNNDecoder, self).__init__()
#         self.in_dim = in_dim
#         self.dropout = 0.0 if n_layers == 1 else dropout
#         self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

#     def forward(self, x):
#         decoder_out, _ = self.rnn(x)
#         return decoder_out
    
# class ReconstructionModel(nn.Module):
#     """Reconstruction Model
#     :param window_size: length of the input sequence
#     :param in_dim: number of input features
#     :param n_layers: number of layers in RNN
#     :param hid_dim: hidden size of the RNN
#     :param in_dim: number of output features
#     :param dropout: dropout rate
#     """

#     def __init__(self, window_size, in_dim, out_dim):
#         hid_dim = 16
#         n_layers = 3
#         dropout = 0.1
#         super(ReconstructionModel, self).__init__()
#         self.window_size = window_size
#         self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
#         self.fc = nn.Linear(hid_dim, out_dim)

#     def forward(self, x):
#         # x will be last hidden state of the GRU layer
#         h_end = x
#         h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)

#         decoder_out = self.decoder(h_end_rep)
#         out = self.fc(decoder_out)
#         return out