"""
Unit tests for Generative Adversarial Networks. Tests the API of the
GenerativeAdversarial container and the GANCost cost function

"""
from neon.initializers.initializer import Gaussian
from neon.layers import Affine
from neon.transforms import GANCost
from neon.layers.container import Sequential, GenerativeAdversarial
from utils import tensors_allclose, allclose_with_out

def test_gan_container(backend_default):
    if False:
        i = 10
        return i + 15
    '\n    Set up a GenerativeAdversarial container and make sure generator\n    and discriminator layers get configured correctly.\n    '
    init_norm = Gaussian(loc=0.0, scale=0.01)
    generator = Sequential([Affine(nout=10, init=init_norm), Affine(nout=100, init=init_norm)])
    discriminator = Sequential([Affine(nout=100, init=init_norm), Affine(nout=1, init=init_norm)])
    layers = GenerativeAdversarial(generator, discriminator)
    assert len(layers.layers) == 4
    assert layers.layers[0].nout == 10
    assert layers.layers[1].nout == 100
    assert layers.layers[2].nout == 100
    assert layers.layers[3].nout == 1
    assert layers.generator.layers == layers.layers[0:2]
    assert layers.discriminator.layers == layers.layers[2:4]

def test_modified_gan_cost(backend_default):
    if False:
        while True:
            i = 10
    '\n    Set up a modified GANCost transform and make sure cost and errors are getting\n    computed correctly.\n    '
    be = backend_default
    cost = GANCost(cost_type='dis', func='modified')
    y_data = be.iobuf(5).fill(1.0)
    y_noise = be.iobuf(5).fill(2.0)
    output = be.iobuf(1)
    expected = be.iobuf(1)
    delta = be.iobuf(5)
    output[:] = cost(y_data, y_noise)
    expected[:] = -be.sum(be.safelog(y_data) + be.safelog(1 - y_noise), axis=0)
    tensors_allclose(output, expected)
    delta[:] = cost.bprop_data(y_data)
    assert allclose_with_out(delta.get(), -1.0 / 1)
    delta[:] = cost.bprop_noise(y_noise)
    assert allclose_with_out(delta.get(), 1.0 - 2.0)
    delta[:] = cost.bprop_generator(y_noise)
    assert allclose_with_out(delta.get(), -1.0 / 2.0)

def test_wgan_cost(backend_default):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set up a Wasserstein GANCost transform and make sure cost and errors are getting\n    computed correctly.\n    '
    be = backend_default
    cost = GANCost(func='wasserstein')
    y_data = be.iobuf(5).fill(1.0)
    y_noise = be.iobuf(5).fill(2.0)
    output = be.iobuf(1)
    expected = be.iobuf(1)
    delta = be.iobuf(5)
    output[:] = cost(y_data, y_noise)
    expected[:] = be.sum(y_data - y_noise, axis=0)
    tensors_allclose(output, expected)
    delta[:] = cost.bprop_data(y_data)
    assert allclose_with_out(delta.get(), 1.0)
    delta[:] = cost.bprop_noise(y_noise)
    assert allclose_with_out(delta.get(), -1.0)
    delta[:] = cost.bprop_generator(y_noise)
    assert allclose_with_out(delta.get(), 1.0)