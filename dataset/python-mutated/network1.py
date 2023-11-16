import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.linear = nn.Linear(10, 20)