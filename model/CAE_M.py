import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)


class CAE_M(nn.Module):
	def __init__(self, feats, window):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = 0.001
		self.n_feats = feats
		self.n_window = window
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		# print(f'gshape={g.shape}')
		z = g.reshape(1, 1, g.shape[0], self.n_feats * self.n_window)
		z = self.encoder(z)
		# print(f'zshape={z.shape}')
		## Decode
		x = self.decoder(z)
		x = x.view(g.shape)
		# print(x.shape)
		return x