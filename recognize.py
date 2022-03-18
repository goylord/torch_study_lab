import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import gzip

filepath = "./data/mnist.json.gz"
print("loading dataset...")
data = json.load(gzip.open(filepath))
print(len(data))

train_data, val_set, eval_data = data
result_data = np.array(train_data[1])
train_data = np.array(train_data[0])
print(len(train_data), result_data)
train_data = np.c_[train_data, result_data]
print('train_data length: ', len(train_data), len(train_data[0]))

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
net = Net()

optimizer = optim.Adam(net.parameters(), lr = 0.01)
loss_calc = nn.CrossEntropyLoss()
for one_data in DataLoader(train_data, batch_size=20, shuffle=True):
  optimizer.zero_grad()
  one_data = one_data.float()
  need_train = one_data[:, :-1]
  need_train = need_train.view(20, 1, 28, 28)
  result = one_data[:, -1]
  out = net(need_train)
  loss = loss_calc(out, result.long())
  pred = out.data.max(1, keepdim=True)[1]
  print("损失函数", loss)
  loss.backward()
  optimizer.step()
