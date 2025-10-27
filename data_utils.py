from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

def train_dataloader():
  dataloader = DataLoader(
    datasets.MNIST(root='.',train=True,download=True,transform=transforms),batch_size=batch_size,shuffle=True)
  return dataloader
def test_dataloader():
  dataloader = DataLoader(
      datasets.MNIST(root = '.',train=False,download=True,transform=transforms),batch_size=batch_size,shuffle=True
  )
  return dataloader
