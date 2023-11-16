"""
Automatic differentiation of optrees
Supports:
    - elementwise operations
      - unary
      - binary
    - dot
    - reductions (<2d, same as nervanagpu)
To support:
    - batched_dot
    - zero-operand operations
    - slicing (need to modify tensor view)
TODO:
    - make use of empty_like
    - intrinsic key for caching
"""
from __future__ import division
from builtins import object, zip
from neon.backends.backend import OpTreeNode, Tensor
import numpy as np
from functools import wraps
_scalar_types = {int, float, np.float16, np.float32, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32}

class GradUtil(object):
    """
    Utility class for calculating gradients.
    """

    @staticmethod
    def get_grad_back(grad_node):
        if False:
            print('Hello World!')
        '\n        Get left and right gradient increments from back-propagation.\n\n        Arguments:\n            grad_node (GradNode): The GradNode to perform gradient\n                                  back-propagation on.\n        '
        if not grad_node:
            return None
        x = grad_node.left.op_tree if grad_node.left else None
        y = grad_node.right.op_tree if grad_node.right else None
        z = grad_node.op_tree
        dz = grad_node.grad_op_tree
        op_dict = z[0]
        be = grad_node.ad.be
        op = grad_node.op_tree[0]['op']
        grad_increments = grad_map[op](x, y, z, dz, op_dict, be)
        left_increment = GradUtil._unbroadcast(grad_increments[0], x, be)
        right_increment = GradUtil._unbroadcast(grad_increments[1], y, be)
        return (left_increment, right_increment)

    @staticmethod
    def _unbroadcast(grad_op_tree, x, be):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reverse broadcast from shape(grad_op_tree) to shape(x)\n\n        Arguments:\n            grad_op_tree (OpTreeNode or Tensor): The OpTreeNode to broadcast.\n            x (OpTreeNode or Tensor): Provides the dimension to be broadcasted to.\n            be: (Backend): The backend to be used.\n\n        Returns:\n            OpTreeNode or Tensor: The broadcasted result.\n        '
        if not grad_op_tree or not x:
            return grad_op_tree
        if type(x) in _scalar_types:
            return 0.0
        in_shape = x.shape
        out_shape = grad_op_tree.shape
        if in_shape == out_shape:
            return grad_op_tree
        elif len(in_shape) == 2 and len(out_shape) == 2:
            if in_shape == (1, 1):
                return be.sum(grad_op_tree)
            elif in_shape[0] == out_shape[0] and in_shape[1] == 1:
                return be.sum(grad_op_tree, axis=1)
            elif in_shape[0] == 1 and in_shape[1] == out_shape[1]:
                return be.sum(grad_op_tree, axis=0)
            elif out_shape[0] == in_shape[0] and out_shape[1] == 1 or (out_shape[0] == 1 and out_shape[1] == in_shape[1]):
                return 0 * x + grad_op_tree
            else:
                return NotImplemented
        else:
            return NotImplemented

    @staticmethod
    def is_invalid(grad_op_tree, be):
        if False:
            while True:
                i = 10
        '\n        Test if the result of grad_op_tree contains Nan, inf, -inf, or\n        abnormally large or small numbers. Only for debug purpose.\n\n        Arguments:\n            grad_op_tree (OpTreeNode or Tensor): The tensor or op-tree to test.\n            be (Backend): The backend to be used.\n\n        Returns:\n            bool: Whether the result contains Nan, inf, -inf, or abnormally\n                  large or small numbers\n        '
        grad_op_tree_val = be.empty(grad_op_tree.shape)
        grad_op_tree_val[:] = grad_op_tree
        grad_op_tree_val_np = grad_op_tree_val.get().reshape(-1)
        for val in grad_op_tree_val_np:
            if not -50000.0 < val < 50000.0:
                return True
        else:
            return False
    '\n    (Applies to the following grad functions)\n    Return gradients for these operations.\n\n    Arguments:\n        x (Tensor, int, float, OpTreeNode): Left operand.\n        y (Tensor, int, float, OpTreeNode): Right operand.\n        z (Tensor, int, float, OpTreeNode): `z = x op y`\n        dz (Tensor, int, float, OpTreeNode): Gradient w.r.t.`z`\n        op_dict (dict): Dictionary specifying the operation.\n        be (Backend): The backend of the tensors.\n\n    Returns:\n        Tuple: (left_increment, right_increment)\n    '

    @staticmethod
    def _zero_grad_unary(x, y, z, dz, op_dict, be):
        if False:
            while True:
                i = 10
        return (dz * 0.0, None)

    @staticmethod
    def _zero_grad_binary(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz * 0.0, dz * 0.0)

    @staticmethod
    def _add_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz, dz)

    @staticmethod
    def _mul_grad(x, y, z, dz, op_dict, be):
        if False:
            return 10
        return (dz * y, dz * x)

    @staticmethod
    def _sub_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz, -dz)

    @staticmethod
    def _neg_grad(x, y, z, dz, op_dict, be):
        if False:
            while True:
                i = 10
        return (-dz, None)

    @staticmethod
    def _pow_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz * y * x ** (y - 1.0), dz * z * be.log(x))

    @staticmethod
    def _div_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz / y, -dz * x / be.square(y))

    @staticmethod
    def _dot_grad(x, y, z, dz, op_dict, be):
        if False:
            print('Hello World!')
        return (be.dot(dz, y.T), be.dot(x.T, dz))

    @staticmethod
    def _abs_grad(x, y, z, dz, op_dict, be):
        if False:
            return 10
        return (dz * be.sgn(x), None)

    @staticmethod
    def _sqrt_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz * 0.5 / z, None)

    @staticmethod
    def _sqr_grad(x, y, z, dz, op_dict, be):
        if False:
            print('Hello World!')
        return (dz * 2.0 * x, None)

    @staticmethod
    def _exp_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz * z, None)

    @staticmethod
    def _exp2_grad(x, y, z, dz, op_dict, be):
        if False:
            print('Hello World!')
        return (dz * z * be.log(2.0), None)

    @staticmethod
    def _log_grad(x, y, z, dz, op_dict, be):
        if False:
            while True:
                i = 10
        return (dz / x, None)

    @staticmethod
    def _log2_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz / x / be.log(2.0), None)

    @staticmethod
    def _sig_grad(x, y, z, dz, op_dict, be):
        if False:
            return 10
        return (dz * z * (1.0 - z), None)

    @staticmethod
    def _sig2_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz * z * (1.0 - z) * be.log(2.0), None)

    @staticmethod
    def _tanh_grad(x, y, z, dz, op_dict, be):
        if False:
            return 10
        return (dz * (1.0 - be.square(z)), None)

    @staticmethod
    def _tanh2_grad(x, y, z, dz, op_dict, be):
        if False:
            for i in range(10):
                print('nop')
        return (dz * (1.0 - be.square(z)) * be.log(2.0), None)

    @staticmethod
    def _max_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz * (x == z), None)

    @staticmethod
    def _min_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz * (x == z), None)

    @staticmethod
    def _maximum_grad(x, y, z, dz, op_dict, be):
        if False:
            while True:
                i = 10
        return (dz * be.greater_equal(x, y), dz * be.greater_equal(y, x))

    @staticmethod
    def _minimum_grad(x, y, z, dz, op_dict, be):
        if False:
            return 10
        return (dz * be.less_equal(x, y), dz * be.less_equal(y, x))

    @staticmethod
    def _sum_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        assert 'axis' in op_dict and op_dict['axis'] in (0, 1)
        return (dz, None)

    @staticmethod
    def _transpose_grad(x, y, z, dz, op_dict, be):
        if False:
            i = 10
            return i + 15
        return (dz.T, None)
grad_map = {'eq': GradUtil._zero_grad_binary, 'lt': GradUtil._zero_grad_binary, 'le': GradUtil._zero_grad_binary, 'gt': GradUtil._zero_grad_binary, 'ge': GradUtil._zero_grad_binary, 'sgn': GradUtil._zero_grad_unary, 'finite': GradUtil._zero_grad_unary, 'argmax': GradUtil._zero_grad_unary, 'argmin': GradUtil._zero_grad_unary, 'add': GradUtil._add_grad, 'mul': GradUtil._mul_grad, 'sub': GradUtil._sub_grad, 'pow': GradUtil._pow_grad, 'div': GradUtil._div_grad, 'dot': GradUtil._dot_grad, 'neg': GradUtil._neg_grad, 'abs': GradUtil._abs_grad, 'sqrt': GradUtil._sqrt_grad, 'sqr': GradUtil._sqr_grad, 'exp': GradUtil._exp_grad, 'exp2': GradUtil._exp2_grad, 'log': GradUtil._log_grad, 'log2': GradUtil._log2_grad, 'sig': GradUtil._sig_grad, 'sig2': GradUtil._sig2_grad, 'tanh': GradUtil._tanh_grad, 'tanh2': GradUtil._tanh2_grad, 'max': GradUtil._max_grad, 'min': GradUtil._min_grad, 'maximum': GradUtil._maximum_grad, 'minimum': GradUtil._minimum_grad, 'sum': GradUtil._sum_grad, 'transpose': GradUtil._transpose_grad}

def memoize_autodiff(func):
    if False:
        print('Hello World!')
    '\n    Memoize to avoid rebuilding of the gradient tree.\n\n    Arguments:\n        func (Function): Function to memoize.\n    '
    cache = {}

    @wraps(func)
    def memoizer(op_tree, be, next_error=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        If params in the caches, return results directly. Othewise, add to cache\n        and return the results.\n\n        Arguments:\n            op_tree (OpTreeNode): the op-tree to supply to the func.\n            be (Backend): computation backend to supply to the func.\n            next_error (Tensor or OpTreeNode, optional): next layer's error to\n                                                         supply to the func.\n        "
        key = (op_tree.key(), be, next_error)
        if key not in cache:
            cache[key] = func(op_tree, be, next_error)
        return cache[key]
    return memoizer

@memoize_autodiff
class Autodiff(object):
    """
    Automatic differentiation given an op-tree.

    Arguments:
        op_tree (OpTreeNode): the op-tree to take gradient of
        be (Backend): computation backend used
        next_error (Tensor or OpTreeNode, optional): next layer's error, usually
                                                     self.delta in a layer. If
                                                     set to None, then automatically
                                                     the default value is tensor
                                                     ones() in output shape
    """
    __slots__ = ['op_tree', 'be', 'dtype', 'next_error', 'map_tensor_grad_node', 'map_tensor_grad_op_tree', 'grad_node']

    def __init__(self, op_tree, be, next_error=None):
        if False:
            for i in range(10):
                print('nop')
        assert type(op_tree) in _scalar_types or type(op_tree) == OpTreeNode or isinstance(op_tree, Tensor), 'op_tree type not supported'
        assert be is not None
        self.op_tree = op_tree
        self.be = be
        self.dtype = be.default_dtype
        if next_error is not None:
            assert next_error.shape == op_tree.shape, 'next_error.shape %s must be consistant with op_tree.shape %s' % (next_error.shape, op_tree.shape)
            self.next_error = next_error
        else:
            self.next_error = self.be.ones(op_tree.shape)
        self.map_tensor_grad_node = {}
        self.map_tensor_grad_op_tree = {}
        self.grad_node = GradNode(op_tree, self)
        if self.next_error:
            self.grad_node.grad_op_tree = self.next_error
        else:
            self.grad_node.grad_op_tree = self.be.ones(self.op_tree.shape)
        self.grad_node.build_grad()

    def __del__(self):
        if False:
            print('Hello World!')
        self.cleanup()

    def cleanup(self):
        if False:
            return 10
        '\n        Perform cleanup on object deletion.\n        '
        if self.grad_node is not None:
            self.grad_node.cleanup()
        self.grad_node = None
        self.dtype = None
        self.next_error = None
        self.op_tree = None
        self.be = None

    def back_prop_grad(self, tensors, gradients):
        if False:
            return 10
        '\n        Back-propagate the gradient of the `tensors` to `gradients`.\n\n        Arguments:\n            Tensors (list): List of Tensors to compute gradients.\n            Gradient (list): List of Tensors, as output buffers of the\n                             Gradients.\n        '
        for grad_buffer in gradients:
            assert grad_buffer._original_base not in self.map_tensor_grad_op_tree
        skipped_tensor = None
        for (tensor, grad_buffer) in zip(tensors, gradients):
            if grad_buffer is self.next_error:
                skipped_tensor = tensor
            else:
                grad_buffer[:] = self.map_tensor_grad_op_tree.get(tensor._original_base, grad_buffer * 0.0)
        if skipped_tensor:
            self.next_error[:] = self.map_tensor_grad_op_tree.get(skipped_tensor._original_base, self.next_error * 0.0)

    def get_grad_op_tree(self, tensors):
        if False:
            return 10
        '\n        Get gradient op_trees w.r.t the list of `tensors`. If a tensor is not\n        used, its gradient will be set to zero.\n\n        Arguments:\n            Tensors (list): List of Tensors to compute gradients.\n\n        Returns\n            list: A list of op_trees, each of them is the gradent of the input\n                  tensor.\n        '
        grad_op_trees = []
        for tensor in tensors:
            grad_op_trees.append(self.map_tensor_grad_op_tree.get(tensor._original_base, tensor * 0.0))
        return grad_op_trees

    def get_grad_tensor(self, tensors):
        if False:
            return 10
        '\n        Get gradient values in type Tensor w.r.t the list of `tensors`. If a\n        tensor is not used, its gradient will be set to zero.\n\n        Arguments:\n            Tensors (list): List of Tensors to compute gradients on.\n\n        Returns\n            list: A list of Tensors, each of them is the gradent of the input\n                  tensor.\n        '
        grad_op_trees = self.get_grad_op_tree(tensors)
        grad_vals = []
        for grad_op_tree in grad_op_trees:
            grad_val = self.be.empty(grad_op_tree.shape)
            grad_val[:] = grad_op_tree
            grad_vals.append(grad_val)
        return grad_vals

    def get_grad_asnumpyarray(self, tensors):
        if False:
            return 10
        '\n        Get gradient values as numpy array w.r.t the list of `tensors`. If a\n        tensor is not used, its gradient will be set to zero.\n\n        Arguments:\n            Tensors (list): List of Tensors to compute gradients.\n\n        Returns\n            list: A list of numpy.ndarray, each of them is the gradient of the\n                  input tensor.\n        '
        grad_vals = self.get_grad_tensor(tensors)
        for i in range(len(grad_vals)):
            grad_vals[i] = grad_vals[i].get().astype(self.dtype)
        return grad_vals

class GradNode(object):
    """
    A node in grad_tree. A GradNode contains the op_optree and the grad_op_tree
    at this location of the grad_tree, and it also has pointers to the left and
    right child in the grad_tree.
    """
    __slots__ = ['op_tree', 'grad_op_tree', 'ad', 'left', 'right']

    def __init__(self, op_tree, ad):
        if False:
            return 10
        '\n        Arguments:\n            op_tree (OpTreeNode or Tensor): the op_tree at this grad_node\n            ad (Autodiff): the autodiff object with global op_tree, next_error and dicts\n        '
        assert op_tree is not None
        self.op_tree = op_tree
        self.grad_op_tree = None
        self.ad = ad
        self.left = None
        self.right = None
        if isinstance(op_tree, Tensor):
            if op_tree._original_base not in ad.map_tensor_grad_node:
                ad.map_tensor_grad_node[op_tree._original_base] = self
        elif type(op_tree) == OpTreeNode:
            if op_tree[1] is not None:
                if isinstance(op_tree[1], Tensor) and op_tree[1]._original_base in ad.map_tensor_grad_node:
                    self.left = ad.map_tensor_grad_node[op_tree[1]._original_base]
                else:
                    self.left = GradNode(op_tree[1], ad)
            if op_tree[2] is not None:
                if isinstance(op_tree[2], Tensor) and op_tree[2]._original_base in ad.map_tensor_grad_node:
                    self.right = ad.map_tensor_grad_node[op_tree[2]._original_base]
                else:
                    self.right = GradNode(op_tree[2], ad)

    def __del__(self):
        if False:
            return 10
        self.cleanup()

    def cleanup(self):
        if False:
            return 10
        '\n        Perform cleanup on object deletion.\n        '
        self.op_tree = None
        self.grad_op_tree = None
        self.ad = None
        if self.left is not None:
            self.left.cleanup()
        self.left = None
        if self.right is not None:
            self.right.cleanup()
        self.right = None

    def build_grad(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Actually back-propagate the gradient.\n        '
        assert self.grad_op_tree is not None
        if type(self.op_tree) == OpTreeNode:
            (left_increment, right_increment) = GradUtil.get_grad_back(self)
            if self.left.grad_op_tree is None:
                self.left.grad_op_tree = left_increment
            else:
                self.left.grad_op_tree = self.left.grad_op_tree + left_increment
            self.left.build_grad()
            if right_increment is None:
                return
            if self.right.grad_op_tree is None:
                self.right.grad_op_tree = right_increment
            else:
                self.right.grad_op_tree = self.right.grad_op_tree + right_increment
            self.right.build_grad()
        elif isinstance(self.op_tree, Tensor):
            self.ad.map_tensor_grad_op_tree[self.op_tree._original_base] = self.grad_op_tree