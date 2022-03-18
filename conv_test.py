import torch
import torch.nn.functional as F

# x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = torch.randn(6,1, 6)
print("卷积之前", y)
conv = torch.nn.Conv2d(1, 1, 3)
conv2 = torch.nn.Conv2d(2, 1, 3)
print("卷积之后", conv(y), )
print("查看relu函数效果", F.relu(conv(y)))
print("查看池化层函数效果", F.max_pool2d(conv(y), 2, 2))

print(torch.randn(2, 2, 3, 4))