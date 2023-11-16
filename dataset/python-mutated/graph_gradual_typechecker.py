from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var
import sympy
_INFERENCE_RULES: Dict[Target, Callable] = {}
_REFINEMENT_RULES: Dict[Target, Callable] = {}
_RULES: Dict[Target, Callable] = {}

def expand_to_tensor_dim(t, n):
    if False:
        i = 10
        return i + 15
    '\n    Expand a type to the desired tensor dimension if possible\n    Raise an error otherwise.\n    - t is the given type\n    - n is a number of dimensions to expand to\n    '
    if t == Dyn:
        dims = [Dyn] * n
        return TensorType(tuple(dims))
    elif isinstance(t, TensorType):
        if len(t.__args__) != n:
            raise TypeError(f'Cannot extend tensor. Tensor {t} has rank {len(t.__args__)}. It should have rank {n}')
        return t
    else:
        raise TypeError(f'Cannot match the type {t}')

def broadcast_types(t1, t2):
    if False:
        i = 10
        return i + 15
    '\n    Applies broadcasting to both given types such that they\n    become consistent with eachother and returns two new\n    resulting types\n    '
    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return (t1, t2)
    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        s1 = len(t1.__args__)
        s2 = len(t2.__args__)
        new_t1 = list(t1.__args__)
        new_t2 = list(t2.__args__)
        if s1 > s2:
            for i in range(s1 - s2):
                new_t2.insert(0, 1)
        elif s2 > s1:
            for i in range(s2 - s1):
                new_t1.insert(0, 1)
        for (i, (x, y)) in enumerate(zip(new_t1, new_t2)):
            if x == 1:
                new_t1[i] = y
            elif y == 1:
                new_t2[i] = x
        (t1, t2) = (TensorType(tuple(new_t1)), TensorType(tuple(new_t2)))
        return (t1, t2)
    else:
        raise TypeError(f'Cannot broadcast types {t1} and {t2}')

def register_inference_rule(call_target):
    if False:
        print('Hello World!')

    def register(fn):
        if False:
            return 10
        if call_target in _INFERENCE_RULES:
            raise RuntimeError(f'Inference rule already registered for {call_target}!')
        _INFERENCE_RULES[call_target] = fn
        return fn
    return register

def register_refinement_rule(call_target):
    if False:
        return 10

    def register(fn):
        if False:
            while True:
                i = 10
        if call_target in _REFINEMENT_RULES:
            raise RuntimeError(f'Refinement rule already registered for {call_target}!')
        _REFINEMENT_RULES[call_target] = fn
        return fn
    return register

def register_algebraic_expressions_inference_rule(call_target):
    if False:
        for i in range(10):
            print('nop')

    def register(fn):
        if False:
            return 10
        if call_target in _RULES:
            raise RuntimeError(f'Rule already registered for {call_target}!')
        _RULES[call_target] = fn
        return fn
    return register

@register_inference_rule(torch.add)
@register_inference_rule(operator.add)
def add_inference_rule(n: Node):
    if False:
        for i in range(10):
            print('nop')
    '\n    Apply the addition inference rule. This includes:\n    - scalar addition\n    - broadcasting semantics\n\n    Note that we always return the least precise type between\n    the operands (after applying broadcasting) to be the final type of the operation\n\n    Note that we do not modify the operand types themselves after applying broadcasting\n    to them. We only use them to calculate the final type\n    '
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    t1 = n.args[0].type
    t2 = n.args[1].type
    if t1 == int and isinstance(t2, TensorType):
        n.type = t2
        return n.type
    elif t2 == int and isinstance(t1, TensorType):
        n.type = t1
        return n.type
    (new_t1, new_t2) = broadcast_types(t1, t2)
    if new_t1 != t1 or new_t2 != t2:
        n.meta['broadcast'] = True
        n.meta[str(n.args[0])] = new_t1
        n.meta[str(n.args[1])] = new_t2
    else:
        n.meta['broadcast'] = False
    new_t1 = t1 if not n.meta['broadcast'] else new_t1
    new_t2 = t2 if not n.meta['broadcast'] else new_t2
    if is_consistent(new_t1, new_t2):
        if is_more_precise(new_t1, new_t2):
            n.type = new_t2
        else:
            n.type = new_t1
        return n.type
    else:
        raise TypeError(f'Cannot add arguments {n.args[0]} ({n.args[0].type}) and {n.args[1]} ({n.args[1].type}) in node {n}. Types should match ')

@register_inference_rule(getattr)
def get_attr_inference_rule(n: Node, traced):
    if False:
        return 10
    '\n    The current getattr rule only handles the shape attribute\n    Can be extended to other attributes\n    The most representitive type we have is "Dyn" but the system\n    can be extended with more types, such as a type to represent shapes\n    '
    attr_node = n.args[0]
    attr_name = n.args[1]
    if attr_name == 'shape':
        n.type = Dyn
    else:
        raise TypeError('Not yet implemented')
    return n.type

@register_inference_rule(torch.transpose)
def transpose_inference_rule(n: Node):
    if False:
        i = 10
        return i + 15
    '\n    We check that dimensions for the transpose operations\n    are within range of the tensor type of the node\n    '
    if n.target == torch.transpose:
        assert isinstance(n.args[0], Node)
        t = n.args[0].type
        assert isinstance(n.args[1], int)
        assert isinstance(n.args[2], int)
        (dim1, dim2) = (n.args[1], n.args[2])
        if t == Dyn:
            n.type = Dyn
            return n.type
        elif isinstance(t, TensorType):
            if 0 <= dim1 < len(t.__args__) and 0 <= dim2 < len(t.__args__):
                new_type = list(t.__args__)
                (new_type[dim1], new_type[dim2]) = (new_type[dim2], new_type[dim1])
                final = TensorType(new_type)
                n.type = get_greatest_upper_bound(n.type, final)
                return n.type
            else:
                raise TypeError(f'Cannot transpose {dim1} and {dim2} in type {t} for node {n}')
        else:
            raise TypeError(f'Cannot transpose {dim1} and {dim2} in type {t} for node {n}')

@register_inference_rule(torch.reshape)
def reshape_inference_rule(n: Node):
    if False:
        i = 10
        return i + 15
    '\n    Without dynamism, the rule checks that the\n    product of the elements of the argument tensor\n    type is equal to the product of the elements\n    of the required shape. We gradualize this rule\n    by adding a case to handle fully dynamic input\n    as well as input where some of the tensor dimensions\n    are unknown. In this case we check for divisibility\n    '
    assert isinstance(n.args[0], Node)
    t1 = n.args[0].type
    assert isinstance(n.args[1], list)
    t2 = n.args[1]
    t2_type = TensorType([Dyn if elem == -1 else elem for elem in t2])
    if t1 == Dyn:
        n.type = t2_type
        return t2_type
    elif isinstance(t1, TensorType):
        assert isinstance(t1, TensorType)
        a = [e if e != Dyn else 1 for e in t1.__args__]
        p1 = reduce(lambda x, y: x * y, a)
        p2 = reduce(lambda x, y: x * y, t2)
        if p1 % p2 == 0 or p2 % p1 == 0:
            n.type = t2_type
            return t2_type
        else:
            raise TypeError(f'Cannot reshape in node {n} from {t1} to {t2_type}')
    else:
        raise TypeError(f'Cannot reshape in node {n} from {t1} to {t2_type}')

@register_inference_rule(BatchNorm2d)
def bn2d_inference_rule(n: Node, module_instance):
    if False:
        while True:
            i = 10
    "\n    Given a BatchNorm2D instance and a node check the following conditions:\n    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, x_3, x_4)\n    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')\n    - t is consistent with t'\n    - x_2 is consistent with the module's num_features\n    - x_2' is consistent with the module's num_features\n    output type: the more precise type of t and t'\n    "
    assert isinstance(n.args[0], Node)
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    n.type = expand_to_tensor_dim(n.type, 4)
    if is_consistent(arg_type.__args__[1], module_instance.num_features) and is_consistent(n.type.__args__[1], module_instance.num_features) and is_consistent(arg_type, n.type):
        n.type = get_greatest_upper_bound(arg_type, n.type)
        return n.type
    else:
        raise TypeError(f'Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}')

def calculate_out_dimension(d_in, module_instance, index):
    if False:
        while True:
            i = 10
    '\n    For calculating h_in and w_out according to the conv2D documentation\n    '
    padding = (module_instance.padding, module_instance.padding) if isinstance(module_instance.padding, int) else module_instance.padding
    kernel_size = (module_instance.kernel_size, module_instance.kernel_size) if isinstance(module_instance.kernel_size, int) else module_instance.kernel_size
    stride = (module_instance.stride, module_instance.stride) if isinstance(module_instance.stride, int) else module_instance.stride
    dilation = (module_instance.dilation, module_instance.dilation) if isinstance(module_instance.dilation, int) else module_instance.dilation
    DIMENSION_TYPES = (int, sympy.Symbol)
    if d_in == Dyn:
        return Dyn
    elif isinstance(d_in, DIMENSION_TYPES):
        n = d_in + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1
        return n // stride[0] + 1
    else:
        raise TypeError(f'{d_in} in {module_instance} must be a number or Dyn. Received {type(d_in)}')

def get_greatest_upper_bound(type1, type2):
    if False:
        return 10
    "\n    Get the most precise type that's consistent with the given types\n    "
    if type1 == Dyn:
        return type2
    elif type2 == Dyn:
        return type1
    elif isinstance(type1, TensorType) and isinstance(type2, TensorType):
        if not is_consistent(type1, type2):
            raise TypeError(f'Inconsistent types {type1}, {type2}')
        gub = [t1 if is_more_precise(t1, t2) else t2 for (t1, t2) in zip(type1.__args__, type2.__args__)]
        return TensorType(tuple(gub))

@register_inference_rule(Conv2d)
def conv2d_inference_rule(n: Node, module_instance):
    if False:
        for i in range(10):
            print('nop')
    "\n    Given a Conv2D instance and a node check the following conditions:\n    - the input type can be expanded to a size 4 tensor: t =  (x_1, x_2, H, W)\n    - the current node type can be expanded to a size 4 tensor: t' =  (x_1', x_2', x_3', x_4')\n    - x_2 is consistent with the module's in_channels\n    - let o = (x_1, out_channels, H_out, W_out)\n    then the output is the greatest upper bound of o and the existing node type t'.\n    "
    assert isinstance(n.args[0], Node)
    n.args[0].type = expand_to_tensor_dim(n.args[0].type, 4)
    arg_type = n.args[0].type
    curr_node_type = expand_to_tensor_dim(n.type, 4)
    if is_consistent(arg_type.__args__[1], module_instance.in_channels):
        w_in = arg_type.__args__[3]
        h_in = arg_type.__args__[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType((arg_type.__args__[0], module_instance.out_channels, h_out, w_out))
        gub = get_greatest_upper_bound(new_type, curr_node_type)
        n.type = gub
        return n.type
    else:
        raise TypeError(f'Cannot apply {module_instance} with input type {arg_type} and existing type {n.type} on {n}')

@register_inference_rule(torch.nn.ReLU)
def relu_inference_rule(n: Node, module_instance):
    if False:
        for i in range(10):
            print('nop')
    '\n    Input and output shapes should be equal.\n    '
    assert isinstance(n.args[0], Node)
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    if isinstance(n.args[0].type, TensorType):
        n.type = get_greatest_upper_bound(n.args[0].type, n.type)
    return n.type

def maxpool2d_check(typ, module_instance):
    if False:
        print('Hello World!')
    '\n    Applies the maxpool2d shape information to the input\n    this affects the last two dimensions\n    '
    new_type_list = list(typ.__args__)
    if len(new_type_list) == 4 or len(new_type_list) == 3:
        w_in = new_type_list[-1]
        h_in = new_type_list[-2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type_list[-1] = w_out
        new_type_list[-2] = h_out
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(f'Wrong size {typ} for {module_instance}')

@register_inference_rule(torch.nn.MaxPool2d)
def maxpool2d_inference_rule(n: Node, module_instance):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a MaxPool2D instance and a node check the following conditions:\n    - Input size matches size 3 or 4\n    - Current node type is consistent with the output type we will calculate\n    - Input size matches output size and the last two dimensions of the output\n      are w_out and h_out. The remaining dimensions are the same as the input\n    - Our final result is the greatest upper bound of the output we calculate\n      and the current node type.\n    '
    assert isinstance(n.args[0], Node)
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    if isinstance(n.args[0].type, TensorType):
        output = maxpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output, n.type)
    return n.type

def linear_check(tensor_type, module_instance):
    if False:
        while True:
            i = 10
    '\n    Checks that an input tensor type satisfies the conditions for linear operation\n    and returns the output type based on in and out features given by module_instance\n    '
    if len(tensor_type.__args__) >= 2:
        if is_consistent(module_instance.in_features, tensor_type.__args__[-1]):
            new_type_args = list(tensor_type.__args__)
            new_type_args[-1] = module_instance.out_features
            return TensorType(tuple(new_type_args))
        else:
            raise TypeError(f'Inconsistent {module_instance.in_features} and {tensor_type.__args__[-1]} in {module_instance}')
    else:
        raise TypeError(f'Type {tensor_type} must have rank 2 or more.')

@register_inference_rule(torch.nn.Linear)
def linear_inference_rule(n: Node, module_instance):
    if False:
        return 10
    '\n    Applies the shape information to the input then gets the greatest upper bound\n    of the resulting type and the existing type\n    '
    assert isinstance(n.args[0], Node)
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    if isinstance(n.args[0].type, TensorType):
        output_type = linear_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(output_type, n.type)
    return n.type

def adaptiveavgpool2d_check(tensor_type, module_instance):
    if False:
        print('Hello World!')
    output_size = module_instance.output_size
    if isinstance(output_size, int):
        output_size = [output_size, output_size]
    elif isinstance(output_size, tuple):
        output_size = list(output_size)
        if output_size[0] is None:
            output_size[0] = output_size[1]
        if output_size[1] is None:
            output_size[1] = output_size[0]
    new_type_list = list(tensor_type.__args__)
    if len(tensor_type.__args__) == 4 or len(tensor_type.__args__) == 3:
        new_type_list[-1] = output_size[1]
        new_type_list[-2] = output_size[0]
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(f'Tensor ranks must be 3 or 4. Got {tensor_type}')

@register_inference_rule(torch.nn.AdaptiveAvgPool2d)
def adaptiveavgpool2d_inference_rule(n: Node, module_instance):
    if False:
        for i in range(10):
            print('nop')
    '\n    The input and output sizes should be the same except for the last\n    two dimensions taken from the input, which represent width and height\n    '
    assert isinstance(n.args[0], Node)
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    if isinstance(n.args[0].type, TensorType):
        output_type = adaptiveavgpool2d_check(n.args[0].type, module_instance)
        n.type = get_greatest_upper_bound(n.type, output_type)
    return n.type

def flatten_check(tensor_type, start_dim, end_dim):
    if False:
        while True:
            i = 10
    l = len(tensor_type.__args__)
    start_dim = l if start_dim == -1 else abs(start_dim)
    end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1
    if 0 <= start_dim <= l - 1 and 0 <= end_dim <= l and (start_dim < end_dim):
        my_args = list(tensor_type.__args__)
        lhs = my_args[0:start_dim]
        rhs = my_args[end_dim:]
        mid = my_args[start_dim:end_dim]
        if Dyn in mid:
            mid = [Dyn]
        else:
            mid = [reduce(lambda x, y: x * y, my_args[start_dim:end_dim])]
        new_type_list = lhs + mid + rhs
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(f'Incompatible dimensions {start_dim}, {end_dim - 1} in type {tensor_type}')

@register_inference_rule(torch.flatten)
def flatten_inference_rule(n: Node):
    if False:
        print('Hello World!')
    '\n    Applies the flatten shape information to the input then gets the\n    greatest upper bound of the resulting type and the existing type\n    '
    assert isinstance(n.args[0], Node)
    start_dim = 1
    end_dim = -1
    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]
    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]
    if n.args[0].type == Dyn and isinstance(n.type, TensorType):
        n.args[0].type = expand_to_tensor_dim(n.args[0].type, len(n.type.__args__))
    if isinstance(n.args[0].type, TensorType):
        output_type = flatten_check(n.args[0].type, start_dim, end_dim)
        n.type = get_greatest_upper_bound(output_type, n.type)
    return n.type

class GraphTypeChecker:

    def __init__(self, env, traced):
        if False:
            for i in range(10):
                print('nop')
        self.env = env
        self.traced = traced

    def type_check(self):
        if False:
            i = 10
            return i + 15
        "\n        A gradual type checker for graphs\n        Effect: every node's field type will be\n        populated with a type after type-checking is done\n        "
        graph = self.traced.graph
        for n in graph.nodes:
            self.type_check_node(n)
        return True

    def type_check_node(self, n: Node):
        if False:
            return 10
        '\n        Type check a given fx node.\n        Current operations:\n        - Reshape\n        - Transpose\n        - Add\n        - Relu\n        - conv2d\n        - batchnorm2d\n        - flatten\n        - maxpool2d\n        - adaptiveavgpool2d\n        - linear\n        '
        if n.type is None:
            n.type = Dyn
        if n.op == 'placeholder':
            return n.type
        elif n.op == 'get_attr':
            t = get_parameter(self.traced, n.target)
            if isinstance(t.data, torch.Tensor):
                n.type = TensorType(t.data.shape)
            return n.type
        elif n.op == 'call_function':
            if n.target == getattr:
                assert getattr in _INFERENCE_RULES
                return _INFERENCE_RULES[n.target](n, self.traced)
            elif n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n)
            else:
                raise RuntimeError(f'No inference rule registered for target {n.target}!')
        elif n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance)
            else:
                raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')
        elif n.op == 'output':

            def get_node_type(a):
                if False:
                    for i in range(10):
                        print('nop')
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type
        else:
            raise NotImplementedError(f'Method {n.op} not yet implemented')

@register_refinement_rule(Conv2d)
def conv_refinement_rule(n: Node):
    if False:
        return 10
    '\n    The equality constraints are between the first dimension of\n    the input and output\n    '
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.__args__[0], n.type.__args__[0])]
        return res

@register_refinement_rule(torch.nn.Linear)
def linear_refinement_rule(n: Node):
    if False:
        i = 10
        return i + 15
    '\n    The equality constraints are between the first dimension of\n    the input and output\n    '
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        res = [Equality(arg_type.__args__[0], n.type.__args__[0])]
    return res

@register_refinement_rule(BatchNorm2d)
@register_refinement_rule(torch.nn.ReLU)
def all_eq(n: Node):
    if False:
        print('Hello World!')
    '\n    For operations where the input shape is equal to the output shape\n    '
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.__args__
        args2 = n.type.__args__
        res = [Equality(args1[i], args2[i]) for i in range(len(args1))]
    return res

@register_refinement_rule(torch.nn.AdaptiveAvgPool2d)
@register_refinement_rule(torch.nn.MaxPool2d)
def first_two_eq(n: Node):
    if False:
        print('Hello World!')
    '\n    For operations where the first two dimensions of the input and output shape\n    are equal\n    '
    res = []
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        args1 = arg_type.__args__
        args2 = n.type.__args__
        res = [Equality(args1[0], args2[0]), Equality(args1[1], args2[1])]
    return res

@register_refinement_rule(torch.add)
@register_refinement_rule(operator.add)
def element_wise_eq(n: Node):
    if False:
        while True:
            i = 10
    '\n    For element-wise operations and handles broadcasting.\n    Note that after applying broadcasting to the arguments\n    we are able to determine if certain dimensions have not been broadcast\n    if they are symbolicallu equal.\n\n    in this case, we can establish equality between those dimensions and the\n    corresponding output dimensions.\n\n    Note that it takes two iterations for this result. One iteration to establish\n    equality between certain dimensions of the operands (requiring the whole solver\n    including unification) and another iteration to establish equality between the operands\n    and the resulting type, requiring another round of constraint generation and unificaiton.\n    '
    res = []
    if isinstance(n.args[0], Node) and isinstance(n.args[1], Node):
        arg_type1 = n.args[0].type
        arg_type2 = n.args[1].type
        if isinstance(arg_type1, TensorType) and isinstance(arg_type2, TensorType) and isinstance(n.type, TensorType):
            (args1, args2) = broadcast_types(arg_type1, arg_type2)
            a1 = args1.__args__
            a2 = args2.__args__
            a3 = n.type.__args__
            r = []
            for (x, y, z) in zip(a1, a2, a3):
                if x == y:
                    r.append(Equality(x, z))
            res = r
    return res

@register_refinement_rule(torch.flatten)
def flatten_refinement_rule(n: Node):
    if False:
        print('Hello World!')
    '\n    Generates equality constraints between the dimensions of the input and output\n    that will not be involved in the flatten operation\n    '
    assert isinstance(n.args[0], Node)
    eq_const = []
    start_dim = 1
    end_dim = -1
    if len(n.args) > 1:
        assert isinstance(n.args[1], int)
        start_dim = n.args[1]
    if len(n.args) > 2:
        assert isinstance(n.args[2], int)
        end_dim = n.args[2]
    if isinstance(n.type, TensorType) and isinstance(n.args[0].type, TensorType):
        l = len(n.type.__args__)
        arg_type = n.args[0].type
        start_dim = l if start_dim == -1 else start_dim
        end_dim = l + end_dim + 1 if end_dim < 0 else end_dim + 1
        for (t1, t2) in zip(n.type.__args__[0:start_dim], arg_type.__args__[0:start_dim]):
            eq_const.append(Equality(t1, t2))
        for (t1, t2) in zip(n.type.__args__[end_dim:], arg_type.__args__[end_dim:]):
            eq_const.append(Equality(t1, t2))
    return eq_const

@register_algebraic_expressions_inference_rule(Conv2d)
def conv_rule(n: Node, module_instance):
    if False:
        print('Hello World!')
    '\n    Represents the outout in terms of an algrbraic expression w.r.t\n    the input when possible\n    '
    assert isinstance(n.args[0], Node)
    arg_type = n.args[0].type
    if isinstance(arg_type, TensorType) and isinstance(n.type, TensorType):
        w_in = arg_type.__args__[3]
        h_in = arg_type.__args__[2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type = TensorType((n.type.__args__[0], n.type.__args__[1], h_out, w_out))
        n.type = new_type
        return new_type

class Refine:
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """

    def __init__(self, traced):
        if False:
            while True:
                i = 10
        self.constraints = []
        self.traced = traced
        self.symbol_iter = itertools.count(start=0, step=1)

    def refine(self):
        if False:
            i = 10
            return i + 15
        '\n        Generates constraints for\n        every node in the graph based on\n        the operation.\n        '
        graph = self.traced.graph
        for n in graph.nodes:
            self.refine_node(n)
        return True

    def symbolic_relations(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Infers algebraic relations\n        '
        graph = self.traced.graph
        for n in graph.nodes:
            self.infer_symbolic_relations(n)
        return True

    def replace_dyn_with_fresh_var(self, typ):
        if False:
            for i in range(10):
                print('nop')
        '\n        Replace all unknown types with fresh type variables.\n        '
        if typ == Dyn:
            new_symbol = Var(next(self.symbol_iter))
            return new_symbol
        elif isinstance(typ, TensorType):
            new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.replace_dyn_with_fresh_var(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.replace_dyn_with_fresh_var(t) for t in typ)
        else:
            return typ

    def convert_to_sympy_symbols(self, typ):
        if False:
            print('Hello World!')
        '\n        Replace all unknown types with fresh type variables.\n        '
        if isinstance(typ, Var):
            return sympy.symbols(str(typ))
        elif isinstance(typ, TensorType):
            new_args = [self.convert_to_sympy_symbols(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.convert_to_sympy_symbols(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.convert_to_sympy_symbols(t) for t in typ)
        else:
            return typ

    def refine_node(self, n: Node):
        if False:
            print('Hello World!')
        '\n        Returns a list of equality constraints for\n        call_module and call_function nodes.\n        Models the relation between input and output dimensions\n        using constraints in case they are both tensors.\n        All operations used in resnet50 are defined.\n        '
        if n.type is None:
            n.type = Dyn
        n.type = self.replace_dyn_with_fresh_var(n.type)
        if n.op == 'call_function':
            if n.target in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[n.target](n)
            else:
                pass
        if n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[type(module_instance)](n)
            else:
                pass
        if n.op == 'output':

            def get_node_type(a):
                if False:
                    i = 10
                    return i + 15
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type
        else:
            pass

    def infer_symbolic_relations(self, n: Node):
        if False:
            return 10
        n.type = self.convert_to_sympy_symbols(n.type)
        if n.op == 'call_function':
            if n.target in _RULES:
                return _RULES[n.target](n)
            else:
                pass
        if n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _RULES:
                return _RULES[type(module_instance)](n, module_instance)
            else:
                pass
        if n.op == 'output':

            def get_node_type(a):
                if False:
                    while True:
                        i = 10
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type
        else:
            pass

def get_parameter(traced, target: str):
    if False:
        return 10
    "\n    Returns the parameter given by ``target`` if it exists,\n    otherwise throws an error.\n\n    See the docstring for ``get_submodule`` for a more detailed\n    explanation of this method's functionality as well as how to\n    correctly specify ``target``.\n\n    Args:\n        target: The fully-qualified string name of the Parameter\n            to look for. (See ``get_submodule`` for how to specify a\n            fully-qualified string.)\n\n    Returns:\n        torch.nn.Parameter: The Parameter referenced by ``target``\n\n    Raises:\n        AttributeError: If the target string references an invalid\n            path or resolves to something that is not an\n            ``nn.Parameter``\n    "
    (module_path, _, param_name) = target.rpartition('.')
    mod: torch.nn.Module = traced.get_submodule(module_path)
    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + ' has no attribute `' + param_name + '`')
    param: torch.nn.Parameter = getattr(mod, param_name)
    return param