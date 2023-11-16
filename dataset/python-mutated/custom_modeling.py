import torch
from transformers import PreTrainedModel
from .custom_configuration import CustomConfig, NoSuperInitConfig

class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.linear(x)

    def _init_weights(self, module):
        if False:
            print('Hello World!')
        pass

class NoSuperInitModel(PreTrainedModel):
    config_class = NoSuperInitConfig

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.linear = torch.nn.Linear(config.attribute, config.attribute)

    def forward(self, x):
        if False:
            return 10
        return self.linear(x)

    def _init_weights(self, module):
        if False:
            while True:
                i = 10
        pass