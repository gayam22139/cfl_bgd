import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
  """
  CNN with two 5x5 convolution lauers(the first with 32 channels, second with 64,
  each followed with 2x2 max pooling), a fully connected layer with 512 uunits and 
  ReLu activation, and the final Softmax output layer

  Total Expected Params: 1,663,370
  """
  def __init__(self,**kwargs):
    super(MNIST_CNN, self).__init__()

    self.ds_idx = 0

    self.num_of_datasets = kwargs.get("num_of_datasets", 3)
    self.num_of_classes = kwargs.get("num_of_classes", 10)

    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    
    self.pool = nn.MaxPool2d(2,2)
    self.dropout = nn.Dropout(p=0.2)

    self.fc1 = nn.Linear(1024, 512)
    # self.out = nn.Linear(512, 10)
    self.last_layer = nn.ModuleList([nn.Linear(512,self.num_of_classes) for _ in range(self.num_of_datasets)])

  def set_dataset(self, ds_idx):
    self.ds_idx = ds_idx % self.num_of_datasets


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.dropout(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.last_layer[self.ds_idx](x)
    out = F.log_softmax(x, dim=1)

    return out
  

def custom_cnn(**kwargs):
  return MNIST_CNN(**kwargs)