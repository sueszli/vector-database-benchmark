import re
import torch
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.nn.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator
from allennlp.common.testing import AllenNlpTestCase

class TestRegularizers(AllenNlpTestCase):

    def test_l1_regularization(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 5))
        constant_init = Initializer.from_params(Params({'type': 'constant', 'val': -1}))
        initializer = InitializerApplicator([('.*', constant_init)])
        initializer(model)
        value = RegularizerApplicator([('', L1Regularizer(1.0))])(model)
        assert value.data.numpy() == 115.0

    def test_l2_regularization(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 5))
        constant_init = Initializer.from_params(Params({'type': 'constant', 'val': 0.5}))
        initializer = InitializerApplicator([('.*', constant_init)])
        initializer(model)
        value = RegularizerApplicator([('', L2Regularizer(1.0))])(model)
        assert value.data.numpy() == 28.75

    def test_regularizer_applicator_respects_regex_matching(self):
        if False:
            print('Hello World!')
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 5))
        constant_init = Initializer.from_params(Params({'type': 'constant', 'val': 1.0}))
        initializer = InitializerApplicator([('.*', constant_init)])
        initializer(model)
        value = RegularizerApplicator([('weight', L2Regularizer(0.5)), ('bias', L1Regularizer(1.0))])(model)
        assert value.data.numpy() == 65.0

    def test_from_params(self):
        if False:
            return 10
        params = Params({'regexes': [('conv', 'l1'), ('linear', {'type': 'l2', 'alpha': 10})]})
        regularizer_applicator = RegularizerApplicator.from_params(params)
        regularizers = regularizer_applicator._regularizers
        conv = linear = None
        for (regex, regularizer) in regularizers:
            if regex == 'conv':
                conv = regularizer
            elif regex == 'linear':
                linear = regularizer
        assert isinstance(conv, L1Regularizer)
        assert isinstance(linear, L2Regularizer)
        assert linear.alpha == 10

    def test_frozen_params(self):
        if False:
            return 10
        model = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 5))
        constant_init = Initializer.from_params(Params({'type': 'constant', 'val': -1}))
        initializer = InitializerApplicator([('.*', constant_init)])
        initializer(model)
        for (name, param) in model.named_parameters():
            if re.search('0.*$', name):
                param.requires_grad = False
        value = RegularizerApplicator([('', L1Regularizer(1.0))])(model)
        assert value.data.numpy() == 55