import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

# Dataset
np.random.seed(0)
x = np.random.rand(200, 1)
y = np.sin(6 * np.pi * x) + np.random.rand(200, 1)

lr = 0.5
iters = 50000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(int(hidden_size/2))
        self.l3 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(y))
        y = self.l3(y)
        return y

model = TwoLayerNet(30, 1)
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 50 == 0:
        print(loss.data)

        plt.cla()
        plt.scatter(x, y, s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        t = np.arange(0, 1, .01)[:, np.newaxis]
        y_pred = model(t)
        plt.plot(t, y_pred.data, color='r')
        plt.pause(0.0001)

# Plot
plt.cla()
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color='r')
plt.show()