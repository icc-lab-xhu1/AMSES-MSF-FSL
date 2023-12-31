import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


class DAGMM(nn.Module):
	def __init__(self, window, feats, latent):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 150
		self.n_latent = latent
		self.n_window = window # DAGMM w_size = 5
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		## Encode Decoder
		x = x.view(x.shape[0], -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		# print(x_hat.shape)
		# ## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		# ## Estimate
		gamma = self.estimate(z)
		# print(x_hat.shape)
		# print(gamma.shape)
		return z_c, x_hat, z, gamma
