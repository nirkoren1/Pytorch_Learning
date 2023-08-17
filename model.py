import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*14*14, 256)
        self.fc2 = nn.Linear(256, 10)
        self.sm = nn.Softmax(dim=1)
        self.flat = nn.Flatten(1, -1)
        self.layers = [self.conv1, self.relu,
                       self.conv2, self.relu, self.max_pool,
                       self.drop_out,
                       self.flat, self.fc1, self.relu, self.fc2, self.sm]
        self.model = nn.Sequential(*self.layers)

    def feed_forward(self, x):
        y = self.model(x)
        return y


if __name__ == '__main__':
    batch = torch.randn(32, 1, 28, 28)
    model = Model()
    ret = model.feed_forward(batch)
    print(ret.shape, ret)
