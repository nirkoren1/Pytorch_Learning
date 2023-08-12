import torch
import torch.nn as nn
import torch.functional as F


class Model(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.drop_out = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(8, 1, 3, padding=1)
        self.fc = nn.Linear(7*7, 10)
        self.sm = nn.Softmax(dim=1)
        self.flat = nn.Flatten(1, -1)
        self.layers = [self.conv1, self.relu, self.max_pool, self.drop_out, self.conv2,
                       self.relu, self.max_pool, self.drop_out, self.flat, self.fc, self.sm]
        self.model = nn.Sequential(*self.layers)

    def feed_forward(self, x):
        y = self.model(x)
        return y


if __name__ == '__main__':
    batch = torch.randn(32, 1, 28, 28)
    model = Model(32)
    ret = model.feed_forward(batch)
    print(ret.shape, ret)
