import autograd.numpy as np
from mla.neuralnet.layers import Layer, ParamMixin
from mla.neuralnet.parameters import Parameters

class Convolution(Layer, ParamMixin):

    def __init__(self, n_filters=8, filter_shape=(3, 3), padding=(0, 0), stride=(1, 1), parameters=None):
        if False:
            i = 10
            return i + 15
        'A 2D convolutional layer.\n        Input shape: (n_images, n_channels, height, width)\n\n        Parameters\n        ----------\n        n_filters : int, default 8\n            The number of filters (kernels).\n        filter_shape : tuple(int, int), default (3, 3)\n            The shape of the filters. (height, width)\n        parameters : Parameters instance, default None\n        stride : tuple(int, int), default (1, 1)\n            The step of the convolution. (height, width).\n        padding : tuple(int, int), default (0, 0)\n            The number of pixel to add to each side of the input. (height, weight)\n\n        '
        self.padding = padding
        self._params = parameters
        self.stride = stride
        self.filter_shape = filter_shape
        self.n_filters = n_filters
        if self._params is None:
            self._params = Parameters()

    def setup(self, X_shape):
        if False:
            while True:
                i = 10
        (n_channels, self.height, self.width) = X_shape[1:]
        W_shape = (self.n_filters, n_channels) + self.filter_shape
        b_shape = self.n_filters
        self._params.setup_weights(W_shape, b_shape)

    def forward_pass(self, X):
        if False:
            i = 10
            return i + 15
        (n_images, n_channels, height, width) = self.shape(X.shape)
        self.last_input = X
        self.col = image_to_column(X, self.filter_shape, self.stride, self.padding)
        self.col_W = self._params['W'].reshape(self.n_filters, -1).T
        out = np.dot(self.col, self.col_W) + self._params['b']
        out = out.reshape(n_images, height, width, -1).transpose(0, 3, 1, 2)
        return out

    def backward_pass(self, delta):
        if False:
            print('Hello World!')
        delta = delta.transpose(0, 2, 3, 1).reshape(-1, self.n_filters)
        d_W = np.dot(self.col.T, delta).transpose(1, 0).reshape(self._params['W'].shape)
        d_b = np.sum(delta, axis=0)
        self._params.update_grad('b', d_b)
        self._params.update_grad('W', d_W)
        d_c = np.dot(delta, self.col_W.T)
        return column_to_image(d_c, self.last_input.shape, self.filter_shape, self.stride, self.padding)

    def shape(self, x_shape):
        if False:
            return 10
        (height, width) = convoltuion_shape(self.height, self.width, self.filter_shape, self.stride, self.padding)
        return (x_shape[0], self.n_filters, height, width)

class MaxPooling(Layer):

    def __init__(self, pool_shape=(2, 2), stride=(1, 1), padding=(0, 0)):
        if False:
            for i in range(10):
                print('nop')
        'Max pooling layer.\n        Input shape: (n_images, n_channels, height, width)\n\n        Parameters\n        ----------\n        pool_shape : tuple(int, int), default (2, 2)\n        stride : tuple(int, int), default (1,1)\n        padding : tuple(int, int), default (0,0)\n        '
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

    def forward_pass(self, X):
        if False:
            print('Hello World!')
        self.last_input = X
        (out_height, out_width) = pooling_shape(self.pool_shape, X.shape, self.stride)
        (n_images, n_channels, _, _) = X.shape
        col = image_to_column(X, self.pool_shape, self.stride, self.padding)
        col = col.reshape(-1, self.pool_shape[0] * self.pool_shape[1])
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        self.arg_max = arg_max
        return out.reshape(n_images, out_height, out_width, n_channels).transpose(0, 3, 1, 2)

    def backward_pass(self, delta):
        if False:
            return 10
        delta = delta.transpose(0, 2, 3, 1)
        pool_size = self.pool_shape[0] * self.pool_shape[1]
        y_max = np.zeros((delta.size, pool_size))
        y_max[np.arange(self.arg_max.size), self.arg_max.flatten()] = delta.flatten()
        y_max = y_max.reshape(delta.shape + (pool_size,))
        dcol = y_max.reshape(y_max.shape[0] * y_max.shape[1] * y_max.shape[2], -1)
        return column_to_image(dcol, self.last_input.shape, self.pool_shape, self.stride, self.padding)

    def shape(self, x_shape):
        if False:
            while True:
                i = 10
        (h, w) = convoltuion_shape(x_shape[2], x_shape[3], self.pool_shape, self.stride, self.padding)
        return (x_shape[0], x_shape[1], h, w)

class Flatten(Layer):
    """Flattens multidimensional input into 2D matrix."""

    def forward_pass(self, X):
        if False:
            i = 10
            return i + 15
        self.last_input_shape = X.shape
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, delta):
        if False:
            return 10
        return delta.reshape(self.last_input_shape)

    def shape(self, x_shape):
        if False:
            i = 10
            return i + 15
        return (x_shape[0], np.prod(x_shape[1:]))

def image_to_column(images, filter_shape, stride, padding):
    if False:
        i = 10
        return i + 15
    'Rearrange image blocks into columns.\n\n    Parameters\n    ----------\n\n    filter_shape : tuple(height, width)\n    images : np.array, shape (n_images, n_channels, height, width)\n    padding: tuple(height, width)\n    stride : tuple (height, width)\n\n    '
    (n_images, n_channels, height, width) = images.shape
    (f_height, f_width) = filter_shape
    (out_height, out_width) = convoltuion_shape(height, width, (f_height, f_width), stride, padding)
    images = np.pad(images, ((0, 0), (0, 0), padding, padding), mode='constant')
    col = np.zeros((n_images, n_channels, f_height, f_width, out_height, out_width))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            col[:, :, y, x, :, :] = images[:, :, y:y_bound:stride[0], x:x_bound:stride[1]]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n_images * out_height * out_width, -1)
    return col

def column_to_image(columns, images_shape, filter_shape, stride, padding):
    if False:
        print('Hello World!')
    'Rearrange columns into image blocks.\n\n    Parameters\n    ----------\n    columns\n    images_shape : tuple(n_images, n_channels, height, width)\n    filter_shape : tuple(height, _width)\n    stride : tuple(height, width)\n    padding : tuple(height, width)\n    '
    (n_images, n_channels, height, width) = images_shape
    (f_height, f_width) = filter_shape
    (out_height, out_width) = convoltuion_shape(height, width, (f_height, f_width), stride, padding)
    columns = columns.reshape(n_images, out_height, out_width, n_channels, f_height, f_width).transpose(0, 3, 4, 5, 1, 2)
    img_h = height + 2 * padding[0] + stride[0] - 1
    img_w = width + 2 * padding[1] + stride[1] - 1
    img = np.zeros((n_images, n_channels, img_h, img_w))
    for y in range(f_height):
        y_bound = y + stride[0] * out_height
        for x in range(f_width):
            x_bound = x + stride[1] * out_width
            img[:, :, y:y_bound:stride[0], x:x_bound:stride[1]] += columns[:, :, y, x, :, :]
    return img[:, :, padding[0]:height + padding[0], padding[1]:width + padding[1]]

def convoltuion_shape(img_height, img_width, filter_shape, stride, padding):
    if False:
        print('Hello World!')
    'Calculate output shape for convolution layer.'
    height = (img_height + 2 * padding[0] - filter_shape[0]) / float(stride[0]) + 1
    width = (img_width + 2 * padding[1] - filter_shape[1]) / float(stride[1]) + 1
    assert height % 1 == 0
    assert width % 1 == 0
    return (int(height), int(width))

def pooling_shape(pool_shape, image_shape, stride):
    if False:
        return 10
    'Calculate output shape for pooling layer.'
    (n_images, n_channels, height, width) = image_shape
    height = (height - pool_shape[0]) / float(stride[0]) + 1
    width = (width - pool_shape[1]) / float(stride[1]) + 1
    assert height % 1 == 0
    assert width % 1 == 0
    return (int(height), int(width))