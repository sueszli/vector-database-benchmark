from neon.initializers.initializer import Uniform
from neon.transforms.activation import Rectlin
from neon.layers.layer import Linear, Convolution, Convolution_bias, Conv, Bias, Activation, Affine

def test_conv_wrapper(backend_default):
    if False:
        print('Hello World!')
    '\n    Verify that the Conv wrapper constructs the right layer objects.\n    '
    conv = Conv((4, 4, 3), Uniform())
    assert isinstance(conv, list)
    assert len(conv) == 1
    assert isinstance(conv[0], Convolution)
    conv = Conv((4, 4, 3), Uniform(), bias=Uniform())
    assert isinstance(conv, list)
    assert len(conv) == 1
    assert isinstance(conv[0], Convolution_bias)
    conv = Conv((4, 4, 3), Uniform(), activation=Rectlin())
    assert isinstance(conv, list)
    assert len(conv) == 2
    assert isinstance(conv[0], Convolution)
    assert isinstance(conv[1], Activation)
    conv = Conv((4, 4, 3), Uniform(), bias=Uniform(), activation=Rectlin())
    assert isinstance(conv, list)
    assert isinstance(conv[0], Convolution_bias)
    assert isinstance(conv[1], Activation)
    assert len(conv) == 2

def test_affine_wrapper(backend_default):
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify that the Affine wrapper constructs the right layer objects.\n    '
    nout = 11
    aff = Affine(nout, Uniform())
    assert isinstance(aff, list)
    assert len(aff) == 1
    assert isinstance(aff[0], Linear)
    assert aff[0].nout == nout
    aff = Affine(nout, Uniform(), bias=Uniform())
    assert isinstance(aff, list)
    assert len(aff) == 2
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Bias)
    aff = Affine(nout, Uniform(), activation=Rectlin())
    assert isinstance(aff, list)
    assert len(aff) == 2
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Activation)
    aff = Affine(nout, Uniform(), bias=Uniform(), activation=Rectlin())
    assert isinstance(aff, list)
    assert len(aff) == 3
    assert isinstance(aff[0], Linear)
    assert isinstance(aff[1], Bias)
    assert isinstance(aff[2], Activation)