import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import gzip


TRAIN_NUM = 10
BATCH_SIZE = 20

filepath = "./data/mnist.json.gz"
print("loading dataset...")
data = json.load(gzip.open(filepath))

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

for i in range(TRAIN_NUM):
  for one_data in DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True):
    optimizer.zero_grad()
    one_data = one_data.float()
    need_train = one_data[:, :-1]
    need_train = need_train.view(BATCH_SIZE, 1, 28, 28)
    result = one_data[:, -1]
    out = net(need_train)
    loss = loss_calc(out, result.long())
    print("损失函数", loss)
    loss.backward()
    optimizer.step()


test_data, test_result = val_set
test_data = np.array(test_data)
for i in range(len(test_data)):
  test_out = net(torch.FloatTensor(test_data[i]).view(1, 1, 28, 28))
  pred = test_out.data.max(1, keepdim=True)[1]
  print("预测值:{}, 真实值: {}\n".format(pred, test_result[i]))
