import torch.nn as nn
import torch
torch.set_default_dtype(torch.float64)

class OmniAnomaly(nn.Module):
	def __init__(self, feats, latent):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 150
		self.n_latent = latent
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			# nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):

		hidden = torch.rand(2, self.n_feats, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x, hidden)
		# print('out', out.shape)

		x = self.encoder(out)
		# print('x', x.shape)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		# print(mu.shape)
		# print(logvar.shape)
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		x = self.decoder(x)
		# print('x', x.shape)
		# return x.view(-1), mu.view(-1), logvar.view(-1), hidden
		return x, mu.reshape(-1), logvar.reshape(-1), hidden