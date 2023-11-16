from __future__ import print_function, division
from builtins import range, input
import os
import util
import scipy as sp
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from datetime import datetime
from theano.tensor.nnet.bn import batch_normalization_train, batch_normalization_test
from theano.tensor.nnet import conv2d
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
EPSILON = 1e-08
BATCH_SIZE = 64
EPOCHS = 2
BATCH_SIZE = 64
SAVE_SAMPLE_PERIOD = 50
if not os.path.exists('samples'):
    os.mkdir('samples')

def lrelu(x, alpha=0.2):
    if False:
        i = 10
        return i + 15
    return T.nnet.relu(x, alpha)

def adam(params, grads):
    if False:
        i = 10
        return i + 15
    updates = []
    time = theano.shared(0)
    new_time = time + 1
    updates.append((time, new_time))
    lr = LEARNING_RATE * T.sqrt(1 - BETA2 ** new_time) / (1 - BETA1 ** new_time)
    for (p, g) in zip(params, grads):
        m = theano.shared(p.get_value() * 0.0)
        v = theano.shared(p.get_value() * 0.0)
        new_m = BETA1 * m + (1 - BETA1) * g
        new_v = BETA2 * v + (1 - BETA2) * g * g
        new_p = p - lr * new_m / (T.sqrt(new_v) + EPSILON)
        updates.append((m, new_m))
        updates.append((v, new_v))
        updates.append((p, new_p))
    return updates

def batch_norm(input_, gamma, beta, running_mean, running_var, is_training, axes='per-activation'):
    if False:
        return 10
    if is_training:
        (out, _, _, new_running_mean, new_running_var) = batch_normalization_train(input_, gamma, beta, running_mean=running_mean, running_var=running_var, axes=axes, running_average_factor=0.9)
    else:
        new_running_mean = None
        new_running_var = None
        out = batch_normalization_test(input_, gamma, beta, running_mean, running_var, axes=axes)
    return (out, new_running_mean, new_running_var)

class ConvLayer:

    def __init__(self, mi, mo, apply_batch_norm, filtersz=5, stride=2, f=T.nnet.relu):
        if False:
            for i in range(10):
                print('nop')
        W = 0.02 * np.random.randn(mo, mi, filtersz, filtersz)
        self.W = theano.shared(W)
        self.b = theano.shared(np.zeros(mo))
        self.params = [self.W, self.b]
        self.updates = []
        if apply_batch_norm:
            self.gamma = theano.shared(np.ones(mo))
            self.beta = theano.shared(np.zeros(mo))
            self.params += [self.gamma, self.beta]
            self.running_mean = theano.shared(np.zeros(mo))
            self.running_var = theano.shared(np.zeros(mo))
        self.f = f
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm

    def forward(self, X, is_training):
        if False:
            print('Hello World!')
        conv_out = conv2d(input=X, filters=self.W, subsample=(self.stride, self.stride), border_mode='half')
        conv_out += self.b.dimshuffle('x', 0, 'x', 'x')
        if self.apply_batch_norm:
            (conv_out, new_running_mean, new_running_var) = batch_norm(conv_out, self.gamma, self.beta, self.running_mean, self.running_var, is_training, 'spatial')
            if is_training:
                self.updates = [(self.running_mean, new_running_mean), (self.running_var, new_running_var)]
        return self.f(conv_out)

class FractionallyStridedConvLayer:

    def __init__(self, mi, mo, output_shape, apply_batch_norm, filtersz=5, stride=2, f=T.nnet.relu):
        if False:
            i = 10
            return i + 15
        self.filter_shape = (mi, mo, filtersz, filtersz)
        W = 0.02 * np.random.randn(*self.filter_shape)
        self.W = theano.shared(W)
        self.b = theano.shared(np.zeros(mo))
        self.params = [self.W, self.b]
        self.updates = []
        if apply_batch_norm:
            self.gamma = theano.shared(np.ones(mo))
            self.beta = theano.shared(np.zeros(mo))
            self.params += [self.gamma, self.beta]
            self.running_mean = theano.shared(np.zeros(mo))
            self.running_var = theano.shared(np.zeros(mo))
        self.f = f
        self.stride = stride
        self.output_shape = output_shape
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, is_training):
        if False:
            i = 10
            return i + 15
        conv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(X, self.W, input_shape=self.output_shape, filter_shape=self.filter_shape, border_mode='half', subsample=(self.stride, self.stride))
        conv_out += self.b.dimshuffle('x', 0, 'x', 'x')
        if self.apply_batch_norm:
            (conv_out, new_running_mean, new_running_var) = batch_norm(conv_out, self.gamma, self.beta, self.running_mean, self.running_var, is_training, 'spatial')
            if is_training:
                self.updates = [(self.running_mean, new_running_mean), (self.running_var, new_running_var)]
        return self.f(conv_out)

class DenseLayer(object):

    def __init__(self, M1, M2, apply_batch_norm, f=T.nnet.relu):
        if False:
            print('Hello World!')
        W = 0.02 * np.random.randn(M1, M2)
        self.W = theano.shared(W)
        self.b = theano.shared(np.zeros(M2))
        self.params = [self.W, self.b]
        self.updates = []
        if apply_batch_norm:
            self.gamma = theano.shared(np.ones(M2))
            self.beta = theano.shared(np.zeros(M2))
            self.params += [self.gamma, self.beta]
            self.running_mean = theano.shared(np.zeros(M2))
            self.running_var = theano.shared(np.zeros(M2))
        self.f = f
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, is_training):
        if False:
            print('Hello World!')
        a = X.dot(self.W) + self.b
        if self.apply_batch_norm:
            (a, new_running_mean, new_running_var) = batch_norm(a, self.gamma, self.beta, self.running_mean, self.running_var, is_training, 'spatial')
            if is_training:
                self.updates = [(self.running_mean, new_running_mean), (self.running_var, new_running_var)]
        return self.f(a)

class DCGAN:

    def __init__(self, img_length, num_colors, d_sizes, g_sizes):
        if False:
            while True:
                i = 10
        self.img_length = img_length
        self.num_colors = num_colors
        self.latent_dims = g_sizes['z']
        self.X = T.tensor4('placeholderX')
        self.Z = T.matrix('placeholderZ')
        p_real_given_real = self.build_discriminator(self.X, d_sizes)
        self.sample_images = self.build_generator(self.Z, g_sizes)
        p_real_given_fake = self.d_forward(self.sample_images, True)
        self.sample_images_test = self.g_forward(self.Z, False)
        self.d_cost_real = T.nnet.binary_crossentropy(output=p_real_given_real, target=T.ones_like(p_real_given_real))
        self.d_cost_fake = T.nnet.binary_crossentropy(output=p_real_given_fake, target=T.zeros_like(p_real_given_fake))
        self.d_cost = T.mean(self.d_cost_real) + T.mean(self.d_cost_fake)
        self.g_cost = T.mean(T.nnet.binary_crossentropy(output=p_real_given_fake, target=T.ones_like(p_real_given_fake)))
        real_predictions = T.sum(T.eq(T.round(p_real_given_real), 1))
        fake_predictions = T.sum(T.eq(T.round(p_real_given_fake), 0))
        num_predictions = 2.0 * BATCH_SIZE
        num_correct = real_predictions + fake_predictions
        self.d_accuracy = num_correct / num_predictions
        d_grads = T.grad(self.d_cost, self.d_params)
        d_updates = adam(self.d_params, d_grads)
        for layer in self.d_convlayers + self.d_denselayers + [self.d_finallayer]:
            d_updates += layer.updates
        self.train_d = theano.function(inputs=[self.X, self.Z], outputs=[self.d_cost, self.d_accuracy], updates=d_updates)
        g_grads = T.grad(self.g_cost, self.g_params)
        g_updates = adam(self.g_params, g_grads)
        for layer in self.g_denselayers + self.g_convlayers:
            g_updates += layer.updates
        g_updates += self.g_bn_updates
        self.train_g = theano.function(inputs=[self.Z], outputs=self.g_cost, updates=g_updates)
        self.get_sample_images = theano.function(inputs=[self.Z], outputs=self.sample_images_test)

    def build_discriminator(self, X, d_sizes):
        if False:
            return 10
        self.d_params = []
        self.d_convlayers = []
        mi = self.num_colors
        dim = self.img_length
        print('*** conv layer image sizes:')
        for (mo, filtersz, stride, apply_batch_norm) in d_sizes['conv_layers']:
            layer = ConvLayer(mi, mo, apply_batch_norm, filtersz, stride, lrelu)
            self.d_convlayers.append(layer)
            self.d_params += layer.params
            mi = mo
            print('dim:', dim)
            dim = int(np.ceil(float(dim) / stride))
        print('final dim before flatten:', dim)
        mi = mi * dim * dim
        self.d_denselayers = []
        for (mo, apply_batch_norm) in d_sizes['dense_layers']:
            layer = DenseLayer(mi, mo, apply_batch_norm, lrelu)
            mi = mo
            self.d_denselayers.append(layer)
            self.d_params += layer.params
        self.d_finallayer = DenseLayer(mi, 1, False, T.nnet.sigmoid)
        self.d_params += self.d_finallayer.params
        p_real_given_x = self.d_forward(X, True)
        return p_real_given_x

    def d_forward(self, X, is_training):
        if False:
            for i in range(10):
                print('nop')
        output = X
        for layer in self.d_convlayers:
            output = layer.forward(output, is_training)
        output = output.flatten(ndim=2)
        for layer in self.d_denselayers:
            output = layer.forward(output, is_training)
        output = self.d_finallayer.forward(output, is_training)
        return output

    def build_generator(self, Z, g_sizes):
        if False:
            for i in range(10):
                print('nop')
        self.g_params = []
        dims = [self.img_length]
        dim = self.img_length
        for (_, filtersz, stride, _) in reversed(g_sizes['conv_layers']):
            dim = int(np.ceil(float(dim) / stride))
            dims.append(dim)
        dims = list(reversed(dims))
        print('dims:', dims)
        self.g_dims = dims
        mi = self.latent_dims
        self.g_denselayers = []
        for (mo, apply_batch_norm) in g_sizes['dense_layers']:
            layer = DenseLayer(mi, mo, apply_batch_norm)
            self.g_denselayers.append(layer)
            self.g_params += layer.params
            mi = mo
        mo = g_sizes['projection'] * dims[0] * dims[0]
        layer = DenseLayer(mi, mo, not g_sizes['bn_after_project'])
        self.g_denselayers.append(layer)
        self.g_params += layer.params
        mi = g_sizes['projection']
        self.g_convlayers = []
        num_relus = len(g_sizes['conv_layers']) - 1
        activation_functions = [T.nnet.relu] * num_relus + [g_sizes['output_activation']]
        for i in range(len(g_sizes['conv_layers'])):
            (mo, filtersz, stride, apply_batch_norm) = g_sizes['conv_layers'][i]
            f = activation_functions[i]
            output_shape = [BATCH_SIZE, mo, dims[i + 1], dims[i + 1]]
            print('mi:', mi, 'mo:', mo, 'outp shape:', output_shape)
            layer = FractionallyStridedConvLayer(mi, mo, output_shape, apply_batch_norm, filtersz, stride, f)
            self.g_convlayers.append(layer)
            self.g_params += layer.params
            mi = mo
        if g_sizes['bn_after_project']:
            self.gamma = theano.shared(np.ones(g_sizes['projection']))
            self.beta = theano.shared(np.zeros(g_sizes['projection']))
            self.running_mean = theano.shared(np.zeros(g_sizes['projection']))
            self.running_var = theano.shared(np.zeros(g_sizes['projection']))
            self.g_params += [self.gamma, self.beta]
        self.g_sizes = g_sizes
        return self.g_forward(Z, True)

    def g_forward(self, Z, is_training):
        if False:
            return 10
        output = Z
        for layer in self.g_denselayers:
            output = layer.forward(output, is_training)
        output = output.reshape([-1, self.g_sizes['projection'], self.g_dims[0], self.g_dims[0]])
        if self.g_sizes['bn_after_project']:
            (output, new_running_mean, new_running_var) = batch_norm(output, self.gamma, self.beta, self.running_mean, self.running_var, is_training, 'spatial')
            if is_training:
                self.g_bn_updates = [(self.running_mean, new_running_mean), (self.running_var, new_running_var)]
        else:
            self.g_bn_updates = []
        for layer in self.g_convlayers:
            output = layer.forward(output, is_training)
        return output

    def fit(self, X):
        if False:
            return 10
        d_costs = []
        g_costs = []
        N = len(X)
        n_batches = N // BATCH_SIZE
        total_iters = 0
        for i in range(EPOCHS):
            print('epoch:', i)
            np.random.shuffle(X)
            for j in range(n_batches):
                t0 = datetime.now()
                if type(X[0]) is str:
                    batch = util.files2images_theano(X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
                else:
                    batch = X[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dims))
                (d_cost, d_acc) = self.train_d(batch, Z)
                d_costs.append(d_cost)
                g_cost1 = self.train_g(Z)
                g_cost2 = self.train_g(Z)
                g_costs.append((g_cost1 + g_cost2) / 2)
                print('  batch: %d/%d - dt: %s - d_acc: %.2f' % (j + 1, n_batches, datetime.now() - t0, d_acc))
                total_iters += 1
                if total_iters % SAVE_SAMPLE_PERIOD == 0:
                    print('saving a sample...')
                    samples = self.sample(64)
                    d = self.img_length
                    if samples.shape[-1] == 1:
                        samples = samples.reshape(64, d, d)
                        flat_image = np.empty((8 * d, 8 * d))
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k].reshape(d, d)
                                k += 1
                    else:
                        flat_image = np.empty((8 * d, 8 * d, 3))
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k].transpose((1, 2, 0))
                                k += 1
                    sp.misc.imsave('samples/samples_at_iter_%d.png' % total_iters, flat_image)
        plt.clf()
        plt.plot(d_costs, label='discriminator cost')
        plt.plot(g_costs, label='generator cost')
        plt.legend()
        plt.savefig('cost_vs_iteration.png')

    def sample(self, n):
        if False:
            for i in range(10):
                print('nop')
        Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
        return self.get_sample_images(Z)

def celeb():
    if False:
        return 10
    X = util.get_celeb()
    dim = 64
    colors = 3
    d_sizes = {'conv_layers': [(64, 5, 2, False), (128, 5, 2, True), (256, 5, 2, True), (512, 5, 2, True)], 'dense_layers': []}
    g_sizes = {'z': 100, 'projection': 512, 'bn_after_project': True, 'conv_layers': [(256, 5, 2, True), (128, 5, 2, True), (64, 5, 2, True), (colors, 5, 2, False)], 'dense_layers': [], 'output_activation': T.tanh}
    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)

def mnist():
    if False:
        i = 10
        return i + 15
    (X, Y) = util.get_mnist()
    X = X.reshape(len(X), 1, 28, 28)
    dim = X.shape[2]
    colors = X.shape[1]
    d_sizes = {'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)], 'dense_layers': [(1024, True)]}
    g_sizes = {'z': 100, 'projection': 128, 'bn_after_project': False, 'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)], 'dense_layers': [(1024, True)], 'output_activation': T.nnet.sigmoid}
    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)
if __name__ == '__main__':
    celeb()