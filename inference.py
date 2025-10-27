import torchvision
import torch
from model import Generator,Discriminator
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
latent_dim = 100  # same as used during training
image_size = (1, 28, 28)

G = Generator(latent_dim, image_size).to(device)
G.load_state_dict(torch.load("weights/generator.pth", map_location=device))
G.eval()
z = torch.randn(16,latent_dim).to(device)
with torch.no_grad():
  fake_images = G(z).cpu().view(-1,1,28,28)
grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0).squeeze())
plt.axis('off')
plt.show()