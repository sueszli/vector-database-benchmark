import numpy as np
from autokeras.engine import analyser

class TargetAnalyser(analyser.Analyser):

    def __init__(self, name=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.name = name

class ClassificationAnalyser(TargetAnalyser):

    def __init__(self, num_classes=None, multi_label=False, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_encoder = None
        self.multi_label = multi_label
        self.labels = set()

    def update(self, data):
        if False:
            while True:
                i = 10
        super().update(data)
        if len(self.shape) > 2:
            raise ValueError('Expect the target data for {name} to have shape (batch_size, num_classes), but got {shape}.'.format(name=self.name, shape=self.shape))
        if len(self.shape) > 1 and self.shape[1] > 1:
            return
        self.labels = self.labels.union(set(np.unique(data.numpy())))

    def finalize(self):
        if False:
            i = 10
            return i + 15
        self.labels = sorted(list(self.labels))
        if not self.num_classes:
            if self.encoded:
                if len(self.shape) == 1 or self.shape[1:] == [1]:
                    self.num_classes = 2
                else:
                    self.num_classes = self.shape[1]
            else:
                self.num_classes = len(self.labels)
        if self.num_classes < 2:
            raise ValueError('Expect the target data for {name} to have at least 2 classes, but got {num_classes}.'.format(name=self.name, num_classes=self.num_classes))
        expected = self.get_expected_shape()
        actual = self.shape[1:]
        if len(actual) == 0:
            actual = [1]
        if self.encoded and actual != expected:
            raise ValueError('Expect the target data for {name} to have shape {expected}, but got {actual}.'.format(name=self.name, expected=expected, actual=self.shape[1:]))

    def get_expected_shape(self):
        if False:
            return 10
        if self.num_classes == 2 and (not self.multi_label):
            return [1]
        return [self.num_classes]

    @property
    def encoded(self):
        if False:
            i = 10
            return i + 15
        return self.encoded_for_sigmoid or self.encoded_for_softmax

    @property
    def encoded_for_sigmoid(self):
        if False:
            while True:
                i = 10
        if len(self.labels) != 2:
            return False
        return sorted(self.labels) == [0, 1]

    @property
    def encoded_for_softmax(self):
        if False:
            while True:
                i = 10
        return len(self.shape) > 1 and self.shape[1] > 1

class RegressionAnalyser(TargetAnalyser):

    def __init__(self, output_dim=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def finalize(self):
        if False:
            print('Hello World!')
        if self.output_dim and self.expected_dim() != self.output_dim:
            raise ValueError('Expect the target data for {name} to have shape (batch_size, {output_dim}), but got {shape}.'.format(name=self.name, output_dim=self.output_dim, shape=self.shape))

    def expected_dim(self):
        if False:
            return 10
        if len(self.shape) == 1:
            return 1
        return self.shape[1]