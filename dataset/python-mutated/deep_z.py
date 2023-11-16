"""
This module implements DeepZ proposed in Fast and Effective Robustness Certification.

| Paper link: https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
"""
from typing import Tuple, Union
import numpy as np
import torch

class ZonoDenseLayer(torch.nn.Module):
    """
    Class implementing a dense layer on a zonotope.
    Bias is only added to the zeroth term.
    """

    def __init__(self, in_features: int, out_features: int):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros(out_features, in_features), std=torch.ones(out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros(out_features), std=torch.ones(out_features)))

    def __call__(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        return self.forward(x)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract forward pass through the dense layer.\n\n        :param x: input zonotope to the dense layer.\n        :return: zonotope after being pushed through the dense layer.\n        '
        x = self.zonotope_matmul(x)
        x = self.zonotope_add(x)
        return x

    def concrete_forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            return 10
        '\n        Concrete forward pass through the dense layer.\n\n        :param x: concrete input to the dense layer.\n        :return: concrete dense layer outputs.\n        '
        x = torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias
        return x

    def zonotope_matmul(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            for i in range(10):
                print('nop')
        '\n        Matrix multiplication for dense layer.\n\n        :param x: input to the dense layer.\n        :return: zonotope after weight multiplication.\n        '
        return torch.matmul(x, torch.transpose(self.weight, 0, 1))

    def zonotope_add(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            print('Hello World!')
        '\n        Modification required compared to the normal torch dense layer.\n        The bias is added only to the central zonotope term and not the error terms.\n\n        :param x: zonotope input to have the bias added.\n        :return: zonotope with the bias added to the central (first) term.\n        '
        x[0] = x[0] + self.bias
        return x

class ZonoBounds:
    """
    Class providing functionality for computing operations related to getting lower and upper bounds on zonotopes.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def compute_lb(cent: 'torch.Tensor', eps: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Compute the lower bound on a feature.\n\n        :param eps: tensor with the eps terms\n        :param cent: tensor with the zero zonotope term\n        :return: lower bound on the given feature\n        '
        return torch.sum(-1 * torch.abs(eps), dim=0) + cent

    @staticmethod
    def compute_ub(cent: 'torch.Tensor', eps: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            return 10
        '\n        Compute the upper bound on a feature.\n\n        :param eps: tensor with the eps terms\n        :param cent: tensor with the zero zonotope term\n        :return: upper bound on the given feature\n        '
        return torch.sum(torch.abs(eps), dim=0) + cent

    @staticmethod
    def certify_via_subtraction(predicted_class: int, class_to_consider: int, cent: np.ndarray, eps: np.ndarray) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        To perform the certification we subtract the zonotope of "class_to_consider"\n        from the zonotope of the predicted class.\n\n        :param predicted_class: class the model predicted.\n        :param class_to_consider: class to check if the model could have classified to it.\n        :param cent: center/zeroth zonotope term.\n        :param eps: zonotope error terms.\n        :return: True/False if the point has been certified\n        '
        diff_in_bias = cent[class_to_consider] - cent[predicted_class]
        diff_in_eps = eps[:, class_to_consider] - eps[:, predicted_class]
        lbs = np.sum(-1 * np.abs(diff_in_eps)) + diff_in_bias
        ubs = np.sum(np.abs(diff_in_eps)) + diff_in_bias
        return np.sign(lbs) < 0 and np.sign(ubs) < 0

    def zonotope_get_bounds(self, cent: 'torch.Tensor', eps: 'torch.Tensor') -> Tuple[list, list]:
        if False:
            i = 10
            return i + 15
        '\n        Compute the upper and lower bounds for the final zonotopes\n\n        :param cent: center/zeroth zonotope term.\n        :param eps: zonotope error terms.\n        :return: lists with the upper and lower bounds.\n        '
        upper_bounds_output = []
        lower_bounds_output = []
        for j in range(cent.shape[0]):
            lbs = self.compute_lb(eps=eps[:, j], cent=cent[j])
            ubs = self.compute_ub(eps=eps[:, j], cent=cent[j])
            upper_bounds_output.append(ubs)
            lower_bounds_output.append(lbs)
        return (upper_bounds_output, lower_bounds_output)

    @staticmethod
    def adjust_to_within_bounds(cent: np.ndarray, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Simple helper function to pre-process and adjust zonotope values to be within 0 - 1 range.\n        This is written with image data from MNIST and CIFAR10 in mind using L-infty bounds.\n        Each feature here starts with a single eps term.\n        Users can implement custom pre-processors tailored to their data if it does not conform to these requirements.\n\n        :param cent: original feature values between 0 - 1\n        :param eps: the zonotope error terms.\n        :return: adjusted center and eps values if center + eps exceed 1 or if center - eps falls below 0.\n        '
        for j in range(cent.shape[1]):
            row_of_eps = np.argmax(eps[:, j])
            if cent[:, j] < eps[row_of_eps, j]:
                eps[row_of_eps, j] = (eps[row_of_eps, j] + cent[:, j]) / 2
                cent[:, j] = eps[row_of_eps, j]
            elif cent[:, j] > 1 - eps[row_of_eps, j]:
                eps[row_of_eps, j] = (eps[row_of_eps, j] + (1 - cent[:, j])) / 2
                cent[:, j] = 1 - eps[row_of_eps, j]
        return (cent, eps)

    def pre_process(self, cent: np.ndarray, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Simple helper function to reshape and adjust the zonotope values before pushing through the neural network.\n        This is written with image data from MNIST and CIFAR10 in mind using L-infty bounds.\n        Each feature here starts with a single eps term.\n        Users can implement custom pre-processors tailored to their data if it does not conform to these requirements.\n\n        :param cent: original feature values between 0 - 1\n        :param eps: the zonotope error terms.\n        :return: adjusted center and eps values if center + eps exceed 1 or if center - eps falls below 0.\n        '
        original_shape = cent.shape
        cent = np.reshape(np.copy(cent), (1, -1))
        num_of_error_terms = eps.shape[0]
        (cent, eps) = self.adjust_to_within_bounds(cent, np.copy(eps))
        cent = np.reshape(cent, original_shape)
        reshape_dim = (num_of_error_terms,) + original_shape
        eps = np.reshape(eps, reshape_dim)
        return (cent, eps)

class ZonoConv(torch.nn.Module):
    """
    Wrapper around pytorch's convolutional layer.
    We only add the bias to the zeroth element of the zonotope
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dilation: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int]]=0):
        if False:
            print('Hello World!')
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def __call__(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            print('Hello World!')
        return self.forward(x)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract forward pass through the convolutional layer\n\n        :param x: input zonotope to the convolutional layer.\n        :return x: zonotope after being pushed through the convolutional layer.\n        '
        x = self.conv(x)
        x = self.zonotope_add(x)
        return x

    def concrete_forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            while True:
                i = 10
        '\n        Concrete forward pass through the convolutional layer\n\n        :param x: concrete input to the convolutional layer.\n        :return: concrete convolutional layer outputs.\n        '
        x = self.conv(x)
        bias = torch.unsqueeze(self.bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=0)
        return x + bias

    def zonotope_add(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Modification required compared to the normal torch conv layers.\n        The bias is added only to the central zonotope term and not the error terms.\n\n        :param x: zonotope input to have the bias added.\n        :return: zonotope with the bias added to the central (first) term.\n        '
        bias = torch.unsqueeze(self.bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=-1)
        x[0] = x[0] + bias
        return x

class ZonoReLU(torch.nn.Module, ZonoBounds):
    """
    Implements "DeepZ" for relu.

    | Paper link:  https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
    """

    def __init__(self, device='cpu'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.device = device
        self.concrete_activation = torch.nn.ReLU()

    def __call__(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            print('Hello World!')
        return self.forward(x)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Forward pass through the relu\n\n        :param x: input zonotope to the dense layer.\n        :return x: zonotope after being pushed through the dense layer.\n        '
        return self.zonotope_relu(x)

    def concrete_forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Concrete pass through the ReLU function\n\n        :param x: concrete input to the activation function.\n        :return: concrete outputs from the ReLU.\n        '
        return self.concrete_activation(x)

    def zonotope_relu(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            return 10
        '\n        Implements "DeepZ" for relu.\n\n        :param x: input zonotope\n        :return x: zonotope after application of the relu. May have grown in dimension if crossing relus occur.\n        '
        original_shape = x.shape
        x = x.reshape((x.shape[0], -1))
        lbs = self.compute_lb(cent=x[0], eps=x[1:])
        ubs = self.compute_ub(cent=x[0], eps=x[1:])
        slope = torch.div(ubs, ubs - lbs)
        index_cent_vector = torch.zeros((x.shape[0], 1)).to(self.device)
        index_cent_vector[0] = 1
        cent_update = slope * lbs / 2
        cent_update = torch.tile(cent_update, (x.shape[0], 1))
        bools = torch.logical_and(lbs < 0, ubs > 0)
        x = torch.where(bools, x * slope - cent_update * index_cent_vector, x)
        zeros = torch.from_numpy(np.zeros(1).astype('float32')).to(self.device)
        x = torch.where(ubs < 0, zeros, x)
        new_vector = torch.unsqueeze(-1 * (slope * lbs / 2), dim=0)
        indexing_matrix = np.zeros((torch.sum(bools), x.shape[1]))
        tmp_crossing_relu = torch.logical_and(lbs < 0, ubs > 0)
        crossing_relu_index = 0
        for (j, crossing_relu) in enumerate(tmp_crossing_relu):
            if crossing_relu:
                indexing_matrix[crossing_relu_index, j] = 1
                crossing_relu_index += 1
        indexing_matrix_tensor = torch.from_numpy(indexing_matrix.astype('float32')).to(self.device)
        new_vector = torch.where(bools, new_vector, zeros)
        new_vector = torch.tile(new_vector, (crossing_relu_index, 1))
        new_vector = new_vector * indexing_matrix_tensor
        x = torch.cat((x, new_vector))
        if len(original_shape) > 2:
            x = x.reshape((-1, original_shape[1], original_shape[2], original_shape[3]))
        return x