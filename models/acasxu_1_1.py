import torch
from torch import nn

class Acasxu1_1(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc0 = nn.Linear(5, 50, bias=True)
    self.relu0 = nn.ReLU()
    self.fc1 = nn.Linear(50, 50, bias=True)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(50, 50, bias=True)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(50, 50, bias=True)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(50, 50, bias=True)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(50, 50, bias=True)
    self.relu5 = nn.ReLU()
    self.output_layer = nn.Linear(50, 5, bias=True)


  def forward(self, x):
    x0 = self.fc0(x)
    x0 = self.relu0(x0)

    x1 = self.fc1(x0)
    x1 = self.relu1(x1)

    x2 = self.fc2(x1)
    x2 = self.relu2(x2)

    x3 = self.fc3(x2)
    x3 = self.relu3(x3)

    x4 = self.fc4(x3)
    x4 = self.relu4(x4)

    x5 = self.fc5(x4)
    x5 = self.relu5(x5)

    logits = self.output_layer(x5)
    return logits
