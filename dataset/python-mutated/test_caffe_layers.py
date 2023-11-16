from bigdl.dllib.nn.layer import *
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.utils.common import *
import numpy as np
import pytest
import tempfile
from numpy import random
from numpy.testing import assert_allclose
import caffe
from caffe_layers import testlayers

class TestCaffeLayers:

    def test_caffe_layers(self):
        if False:
            while True:
                i = 10
        temp = tempfile.mkdtemp()
        for testlayer in testlayers:
            name = testlayer.name
            definition = testlayer.definition
            shapes = testlayer.shapes
            prototxtfile = temp + name + '.prototxt'
            weightfile = temp + name + '.caffemodel'
            with open(prototxtfile, 'w') as prototxt:
                prototxt.write(definition)
            caffe.set_mode_cpu()
            caffe.set_random_seed(100)
            net = caffe.Net(prototxtfile, caffe.TEST)
            inputs = []
            for shape in shapes:
                (inputName, size) = shape.items()[0]
                input = random.uniform(size=size)
                net.blobs[inputName].data[...] = input
                inputs.append(input)
            cafferesult = net.forward().get(name)
            net.save(weightfile)
            model = Model.load_caffe_model(prototxtfile, weightfile, bigdl_type='float')
            model.set_seed(100)
            if len(inputs) == 1:
                inputs = inputs[0]
            bigdlResult = model.forward(inputs)
            print(cafferesult)
            print(bigdlResult)
            assert_allclose(cafferesult, bigdlResult, atol=0.0001, rtol=0)
if __name__ == '__main__':
    pytest.main([__file__])