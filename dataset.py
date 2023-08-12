import pandas as pd
import numpy as np
import cv2
import torch

file_name = "./data/mnist_train_small.csv"


def preprocessing(file_name):

    data = pd.read_csv(file_name)

    data_array = data.to_numpy()

    # creating the y values

    column_index = 0

    first_column = data_array[:, column_index]

    def one_hot_encode(number, num_classes):
        assert 0 <= number < num_classes, f"Number should be between 0 and {num_classes - 1}"
        return torch.tensor([1.0 if i == number else 0.0 for i in range(num_classes)], dtype=torch.float32)

    y = torch.stack([one_hot_encode(i, num_classes=10) for i in first_column])

    # creating the x values

    new_data_array = np.delete(data_array, column_index, axis=1)

    all_columns = torch.tensor(new_data_array[:, column_index:])

    batch_size = all_columns.shape[0]

    width, height = 28, 28

    x = all_columns.reshape((batch_size, 1, width, height)) / 255

    print(x.shape, y.shape)

    return x, y


preprocessing(file_name)
