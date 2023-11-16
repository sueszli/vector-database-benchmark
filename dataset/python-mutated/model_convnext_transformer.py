import torch
import torch.nn as nn
from .ocr_modules.convnext import convnext_tiny
from .ocr_modules.vitstr import vitstr_tiny

class OCRRecModel(nn.Module):

    def __init__(self, num_classes):
        if False:
            i = 10
            return i + 15
        super(OCRRecModel, self).__init__()
        self.cnn_model = convnext_tiny()
        self.num_classes = num_classes
        self.vitstr = vitstr_tiny(num_tokens=num_classes)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        ' Transformation stage '
        features = self.cnn_model(input)
        prediction = self.vitstr(features)
        prediction = torch.nn.functional.softmax(prediction, dim=-1)
        output = torch.argmax(prediction, -1)
        return output