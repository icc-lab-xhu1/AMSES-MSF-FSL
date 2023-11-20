import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)


class USAD(nn.Module):
	def __init__(self, feats, window, latent):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 150
		self.n_latent = latent
		self.n_window = window # USAD w_size = 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g)
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1, ae2, ae2ae1