import random
import numpy as np
import torch
from torch.utils.data import Dataset

class DummyData(Dataset):

    def __init__(self, max_val: int, sample_count: int, sample_length: int, sparsity_percentage: int):
        if False:
            i = 10
            return i + 15
        '\n        A data class that generates random data.\n        Args:\n            max_val (int): the maximum value for an element\n            sample_count (int): count of training samples\n            sample_length (int): number of elements in a sample\n            sparsity_percentage (int): the percentage of\n                embeddings used by the input data in each iteration\n        '
        self.max_val = max_val
        self.input_samples = sample_count
        self.input_dim = sample_length
        self.sparsity_percentage = sparsity_percentage

        def generate_input():
            if False:
                for i in range(10):
                    print('nop')
            precentage_of_elements = (100 - self.sparsity_percentage) / float(100)
            index_count = int(self.max_val * precentage_of_elements)
            elements = list(range(self.max_val))
            random.shuffle(elements)
            elements = elements[:index_count]
            data = [[elements[random.randint(0, index_count - 1)] for _ in range(self.input_dim)] for _ in range(self.input_samples)]
            return torch.from_numpy(np.array(data))
        self.input = generate_input()
        self.target = torch.randint(0, max_val, [sample_count])

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.input)

    def __getitem__(self, index):
        if False:
            return 10
        return (self.input[index], self.target[index])