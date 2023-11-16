import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model

class FakeModelForTestingNormalizationBiasVerification(Model):

    def __init__(self, use_bias=True):
        if False:
            print('Hello World!')
        super().__init__(vocab=Vocabulary())
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=1, bias=use_bias)
        self.bn = torch.nn.BatchNorm2d(5)

    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.bn(self.conv(x))
        return {'loss': out.sum()}