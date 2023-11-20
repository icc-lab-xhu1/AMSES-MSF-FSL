import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

class MAD_GAN(nn.Module):
	def __init__(self, feats, window):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 150
		self.n_window = window # MAD_GAN w_size = 5
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, g):
		z = self.generator(g)
		z = z.view(g.shape[0], g.shape[1],-1)
		real_score = self.discriminator(g)
		fake_score = self.discriminator(z)
		return z, real_score.view(-1), fake_score.view(-1)
