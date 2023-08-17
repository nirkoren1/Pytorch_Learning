from dataset import preprocessing
from model import Model
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np

def show_progress(epoch, step, total_steps, loss, accuracy, width=30, bar_char='█', empty_char='░'):
    print('\r', end='')
    progress = ""
    for i in range(width):
        progress += bar_char if i < int(step / total_steps * width) else empty_char
    print(f"epoch:{epoch + 1} [{progress}] {step}/{total_steps} loss: {loss:.4f} accuracy: {accuracy:.4f}", end='')
    if step >= total_steps - 1:
        print()

file_name = "./data/mnist_train_small.csv"
torch.manual_seed(1)
x, y = preprocessing(file_name)
batch_size = 64
model = Model()
epochs = 5
loss = nn.CrossEntropyLoss()
lossess = []
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.00)
model.train()
for epoch in range(epochs):
    acc = 0
    loss_sum = 0
    for i in range(x.shape[0] // batch_size):
        optimizer.zero_grad()
        rand = torch.randint(0, x.shape[0], (batch_size,))
        batch_x, batch_y = x[rand], y[rand]
        y_hat = model.feed_forward(batch_x)
        current_loss = loss(y_hat, batch_y)
        lossess.append(current_loss.item())
        loss_sum += current_loss.item()
        current_loss.backward()
        optimizer.step()
        for j in range(batch_size):
            if torch.argmax(y_hat[j]) == torch.argmax(batch_y[j]):
                acc += 1
        # print(loss_sum)
        show_progress(epoch, i, x.shape[0] // batch_size, current_loss.item(), acc / (batch_size * (i + 1)))


file_name_test = "./data/mnist_test.csv"
x_test, y_test = preprocessing(file_name_test)
s = 0
model.eval()
for i in range(1000):
    y_hat = model.feed_forward(x_test[i].reshape(1, 1, 28, 28))
    # print(y_hat, y_test[i])
    if torch.argmax(y_hat) == torch.argmax(y_test[i]):
        s += 1
print(f"precision loss {s/1000}")


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




plt.plot(moving_average(lossess, 10))
plt.show()
