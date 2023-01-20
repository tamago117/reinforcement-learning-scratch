import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

lr = 0.02
iters = 10000
hidden_size = 25
x_size = 100

# Dataset
np.random.seed(0)
x = np.random.rand(x_size, 1)
y = np.sin(10 * np.pi * x) + np.random.rand(x_size, 1)
x_t = torch.FloatTensor(x)
y_t = torch.FloatTensor(y)
test = np.arange(0, 1, .01)[:, np.newaxis]
test = torch.FloatTensor(test)

class LinearNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.l1 = nn.Linear(1, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.l3 = nn.Linear(int(self.hidden_size/2), 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.sig(self.l1(x))
        y = self.sig(self.l2(y))
        y = self.l3(y)
        return y


model = LinearNet(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
losses = []

for i in range(iters):
    optimizer.zero_grad()
    y_pred = model(x_t)
    loss = criterion(y_t, y_pred)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
    if i % 50 == 0:
        print("iter: {}/{}, loss: {}".format(i, iters, loss.item()))

        plt.cla()
        plt.scatter(x, y, s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        y_pred = model(test)
        plt.plot(test, y_pred.detach(), color='r')
        plt.pause(0.0001)

# Plot
plt.cla()
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

y_pred = model(test)
plt.plot(test, y_pred.data, color='r')
plt.show()

plt.cla()
plt.plot(losses)
plt.show()