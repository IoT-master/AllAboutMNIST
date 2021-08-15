import torch
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

BATCH_SIZE = 10

trainset = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=.001)

EPOCHS = 1

for epocs in range(EPOCHS):
    for X, y in trainset:
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        print(X.shape)
        print(y)
        print(y.shape)
        break
#         loss = F.nll_loss(output, y)
#         loss.backward()
#         optimizer.step()
#     print(loss)

# correct, total = 0, 0
# with torch.no_grad():
#     for X, y in trainset:
#         output = net(X.view(-1, 28*28))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total += 1
# print(f"Accuracy: {round(correct/total, 3)}")

# correct, total = 0, 0
# with torch.no_grad():
#     for X, y in testset:
#         output = net(X.view(-1, 28*28))
#         for idx, i in enumerate(output):
#             if torch.argmax(i) == y[idx]:
#                 correct += 1
#             total += 1
# print(f"Validation: {round(correct/total, 3)}")
