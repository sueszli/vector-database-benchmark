from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        if False:
            print('Hello World!')
        return self.len
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
for epoch in range(2):
    for (i, data) in enumerate(train_loader, 0):
        (inputs, labels) = data
        (inputs, labels) = (tensor(inputs), tensor(labels))
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')