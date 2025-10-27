import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
  def __init__(self,latent_dim,image_size):
    super().__init__()
    self.image_size = image_size
    def block(in_dim, out_dim,normalize=True):
      layers = [(nn.Linear(in_dim,out_dim))]
      if normalize:
        layers.append(nn.BatchNorm1d(out_dim))
      layers.append(nn.ReLU(True))
      return layers
    self.model = nn.Sequential(
        nn.Linear(latent_dim,128),
        *block(128,256),
        *block(256,512),
        *block(512,1024),
        nn.Linear(1024,int(np.prod(image_size))),
        nn.Tanh(),
    )
  def forward(self,z):
    img = self.model(z)
    img = img.view(img.size(0),*self.image_size)
    return img


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024,512,256]):

        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
