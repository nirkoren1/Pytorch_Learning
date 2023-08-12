from dataset import preprocessing
from model import Model
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np

file_name = "./data/mnist_train_small.csv"
torch.manual_seed(1)
x, y = preprocessing(file_name)
batch_size = 32
model = Model()
epochs = 200
loss = nn.CrossEntropyLoss()
lossess = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(epochs):
    rand = torch.randint(0, x.shape[0], (batch_size,))
    batch_x, batch_y = x[rand], y[rand]
    y_hat = model.feed_forward(batch_x)
    current_loss = loss(y_hat, batch_y)
    lossess.append(current_loss.item())
    current_loss.backward()
    optimizer.step()
file_name_test = "./data/mnist_test.csv"
x_test, y_test = preprocessing(file_name_test)
s = 0
for i in range(1000):
    y_hat = model.feed_forward(x_test[i])
    print(y_hat, y_test[i])
    if torch.argmax(y_hat) == torch.argmax(y_test[i]):
        s += 1
print(f"precision loss {s/1000}")


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plt.plot(moving_average(lossess, 10))
plt.show()
