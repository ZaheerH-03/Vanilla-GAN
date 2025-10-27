import matplotlib.pyplot as plt
import torch
from model import Generator,Discriminator
import torch.optim as optim
import torch.nn  as nn
from data_utils import train_dataloader,test_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
G = Generator(100,(1,28,28)).to(device)
D = Discriminator(784).to(device)
G_opt = optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
D_opt = optim.Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))
criterion = nn.BCELoss()
epochs = 50
latent_dim = 100

for epoch in range(epochs):
  for i, (real_images,_) in enumerate(train_dataloader()):
    real_images = real_images.to(device)
    cur_batch_size = real_images.size(0)

    D_opt.zero_grad()
    real_labels = torch.ones(cur_batch_size,1).to(device)
    real_outputs = D(real_images)
    loss_real = criterion(real_outputs,real_labels)

    z = torch.randn(cur_batch_size,latent_dim).to(device)
    fake_images = G(z).detach()
    fake_labels = torch.zeros(cur_batch_size,1).to(device)
    d_fake = D(fake_images)
    loss_fake = criterion(d_fake,fake_labels)

    loss_d = loss_real + loss_fake
    loss_d.backward()
    D_opt.step()

    G_opt.zero_grad()
    z = torch.randn(cur_batch_size,latent_dim).to(device)
    G_images = G(z)
    G_labels = torch.ones(cur_batch_size,1).to(device)
    G_output = D(G_images)
    loss_g = criterion(G_output,G_labels)
    loss_g.backward()
    G_opt.step()
  print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")
  if (epoch+1) % 10 == 0:
    with torch.no_grad():
      z1 = torch.randn(16, latent_dim).to(device)
      fake_samples = G(z).view(-1,1,28,28)
      plt.imshow(fake_samples[0].cpu().squeeze(), cmap='gray')
      plt.show()

torch.save(G.state_dict(), "weights/generator.pth")
torch.save(D.state_dict(), "weights/discriminator.pth")
