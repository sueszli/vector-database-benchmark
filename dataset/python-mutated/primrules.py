import functools
import math
import operator
import typing
import paddle
from . import primops
from .primops import add, bernoulli, broadcast, concat, cos, div, eq, erf, exp, fill_const, gather, ge, gt, log, matmul, mul, ne, neg, reduce_sum, reshape, rsqrt, scatter_add, select, set_value, sin, slice_assign, slice_select, split, sqrt, sub, tanh, transpose, uniform_random
from .primreg import REGISTER_JVP, REGISTER_ORIG2PRIM, REGISTER_PRIM2ORIG, REGISTER_TRANSPOSE, lookup_fn, lookup_jvp, lookup_orig2prim, lookup_prim2orig, lookup_transpose, op_position_inputs, op_position_output
from .utils import INT_DTYPE_2_STRING, get_output_var_list

def _orig2prim(op, *args):
    if False:
        while True:
            i = 10
    _lowerrule = lookup_orig2prim(op.type)
    return _lowerrule(op, *args)

def _prim2orig(op, *args):
    if False:
        print('Hello World!')
    _lowerrule = lookup_prim2orig(op.type)
    return _lowerrule(op, *args)

def _jvp(op, *args):
    if False:
        for i in range(10):
            print('nop')
    _jvprule = lookup_jvp(op.type)
    return _jvprule(op, *args)

def _transpose(op, dot_checker, *args):
    if False:
        i = 10
        return i + 15
    _transposerule = lookup_transpose(op.type)
    return _transposerule(op, dot_checker, *args)

def linear_jvp(op, *args, **kwargs):
    if False:
        print('Hello World!')
    fn = lookup_fn(op.type)
    out_dot = fn(*args, **kwargs)
    return out_dot
'\nThese original ops are fully supported:\n\nelementwise_add\nelementwise_sub\nelementwise_mul\ntanh\nfill_zeros_like\nfill_any_like\nsum\nindex_select\nscale\nassign\nsqrt\nlog\nselect\nequal\nelementwise_pow\ndropout\nuniform_random\n\nThese original ops are partially supported:\n\nmatmul_v2\nreshape2\nconcat\nslice\np_norm\n'

@REGISTER_ORIG2PRIM('elementwise_add')
def elementwise_add_orig2prim(op, x, y):
    if False:
        while True:
            i = 10
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return add(x, y)

@REGISTER_ORIG2PRIM('elementwise_sub')
def elementwise_sub_orig2prim(op, x, y):
    if False:
        for i in range(10):
            print('nop')
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return sub(x, y)

@REGISTER_ORIG2PRIM('elementwise_mul')
def elementwise_mul_orig2prim(op, x, y):
    if False:
        return 10
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return mul(x, y)

@REGISTER_ORIG2PRIM('elementwise_div')
def elementwise_div_orig2prim(op, x, y):
    if False:
        return 10
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return primops.div(x, y)

@REGISTER_ORIG2PRIM('tanh')
def tanh_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return tanh(x)

@REGISTER_ORIG2PRIM('sin')
def sin_orig2prim(op, x):
    if False:
        while True:
            i = 10
    return sin(x)

@REGISTER_ORIG2PRIM('cos')
def cos_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return cos(x)

@REGISTER_ORIG2PRIM('exp')
def exp_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return exp(x)

@REGISTER_ORIG2PRIM('erf')
def erf_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return erf(x)

@REGISTER_ORIG2PRIM('abs')
def abs_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return primops.abs(x)

@REGISTER_ORIG2PRIM('log')
def log_orig2prim(op, x):
    if False:
        print('Hello World!')
    return log(x)

@REGISTER_ORIG2PRIM('fill_zeros_like')
def fill_zeros_like_orig2prim(op, x):
    if False:
        for i in range(10):
            print('nop')
    return fill_const(value=0.0, shape=x.shape, dtype=x.dtype)

@REGISTER_ORIG2PRIM('fill_any_like')
def fill_any_like_orig2prim(op, x):
    if False:
        while True:
            i = 10
    if op.attr('dtype') == -1:
        return fill_const(value=op.attr('value'), shape=x.shape, dtype=x.dtype)
    return fill_const(value=op.attr('value'), shape=x.shape, dtype=paddle.dtype(op.attr('dtype')))

@REGISTER_ORIG2PRIM('fill_constant')
def fill_const_orig2prim(op, shape_tensor=None, shape_tensor_list=None, value_tensor=None):
    if False:
        for i in range(10):
            print('nop')
    if shape_tensor or shape_tensor_list or value_tensor:
        raise TypeError('fill_const_orig2prim currently not support Tensor input of shape and value.')
    return fill_const(value=op.attr('value'), shape=op.attr('shape'), dtype=paddle.dtype(op.attr('dtype')))

@REGISTER_ORIG2PRIM('sum')
def sum_orig2prim(op, xs):
    if False:
        for i in range(10):
            print('nop')
    x0 = xs[0]
    for x in xs[1:]:
        x0 = add(x0, x)
    return x0

@REGISTER_ORIG2PRIM('index_select')
def index_select_orig2prim(op, index_t, x):
    if False:
        return 10
    return gather(x, indextensor=index_t, axis=op.attr('dim'))

@REGISTER_ORIG2PRIM('scale')
def scale_orig2prim(op, scale_t, x):
    if False:
        while True:
            i = 10
    if scale_t is None:
        scale_t = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('scale'))
    bias_t = fill_const(shape=x.shape, dtype=x.dtype, value=op.attr('bias'))
    if op.attr('bias_after_scale'):
        return add(mul(x, scale_t), bias_t)
    else:
        return mul(add(x, bias_t), scale_t)

@REGISTER_ORIG2PRIM('assign')
def assign_orig2prim(op, x):
    if False:
        while True:
            i = 10
    zero_t = fill_const(shape=x.shape, dtype=x.dtype, value=0.0)
    return add(x, zero_t)

@REGISTER_ORIG2PRIM('sqrt')
def sqrt_orig2prim(op, x):
    if False:
        for i in range(10):
            print('nop')
    return sqrt(x)

@REGISTER_ORIG2PRIM('rsqrt')
def rsqrt_orig2prim(op, x):
    if False:
        while True:
            i = 10
    return rsqrt(x)

@REGISTER_ORIG2PRIM('matmul_v2')
def matmul_v2_orig2prim(op, x, y):
    if False:
        i = 10
        return i + 15

    def trans(shape):
        if False:
            i = 10
            return i + 15
        ret = list(range(len(shape)))
        (ret[-1], ret[-2]) = (ret[-2], ret[-1])
        return ret
    assert len(x.shape) < 4 and len(y.shape) < 4, 'Do not support multi batchsize dimensions currently.'
    if len(x.shape) == 1:
        x = broadcast(x, shape=[1, x.shape[0]])
    if len(y.shape) == 1:
        y = broadcast(y, shape=[y.shape[0], 1])
    if op.attr('trans_x'):
        x = transpose(x, axis=trans(x.shape))
    if op.attr('trans_y'):
        y = transpose(y, axis=trans(y.shape))
    return matmul(x, y)

@REGISTER_ORIG2PRIM('reshape2')
def reshape2_orig2prim(op, shape_t, shape_tl, x):
    if False:
        i = 10
        return i + 15
    assert shape_t is None, 'Can not lower reshape2 into prim ops with shapetensor.'
    assert shape_tl is None, 'Can not lower reshape2 into prim ops with shapetensorlist.'
    (y, xshape) = get_output_var_list(op)
    return (reshape(x, shape=y.shape), fill_const(shape=xshape.shape, dtype=xshape.dtype, value=0.0))

@REGISTER_ORIG2PRIM('concat')
def concat_orig2prim(op, axis_t, xs):
    if False:
        for i in range(10):
            print('nop')
    assert axis_t is None, 'Can not lower concat into prim ops with axistensor.'
    return concat(xs, axis=op.attr('axis'))

@REGISTER_ORIG2PRIM('slice')
def slice_orig2prim(op, ends_t, ends_tl, x, starts_t, starts_tl):
    if False:
        while True:
            i = 10
    assert starts_t is None, 'Can not lower concat into prim ops with startstensor.'
    assert ends_t is None, 'Can not lower concat into prim ops with endstensor.'
    assert starts_tl is None, 'Can not lower concat into prim ops with startstensorlist.'
    assert ends_tl is None, 'Can not lower concat into prim ops with endstensorlist.'
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = [1 for _ in starts]
    axis = op.attr('axes')
    y = slice_select(x, starts=starts, ends=ends, strides=strides, axis=axis)
    if op.attr('decrease_axis'):
        y = reshape(y, shape=get_output_var_list(op)[0].shape)
    return y

@REGISTER_ORIG2PRIM('sigmoid')
def sigmoid_orig2prim(op, x):
    if False:
        print('Hello World!')
    return div(fill_const(value=1.0, shape=x.shape, dtype=x.dtype), add(fill_const(value=1.0, shape=x.shape, dtype=x.dtype), exp(neg(x))))

@REGISTER_ORIG2PRIM('p_norm')
def p_norm_orig2prim(op, x):
    if False:
        while True:
            i = 10

    def num_el(shape):
        if False:
            for i in range(10):
                print('nop')
        n = 1
        for s in shape:
            n = n * s
        return n
    assert op.attr('asvector'), 'Only support lower pnorm when asvector=True currently'
    if len(x.shape) > 1:
        x = reshape(x, shape=[num_el(x.shape)])
    if abs(op.attr('porder') - 2.0) < 1e-05:
        return sqrt(reduce_sum(mul(x, x), axis=[0]))
    elif abs(op.attr('porder') - 1.0) < 1e-05:
        return reduce_sum(primops.abs(x), axis=[0])
    else:
        raise RuntimeError('Only support lower l2/l1 norm currently')

@REGISTER_ORIG2PRIM('cast')
def cast_orig2prim(op, x):
    if False:
        while True:
            i = 10
    return primops.cast(x, paddle.dtype(op.attr('out_dtype')))

@REGISTER_ORIG2PRIM('where')
def select_orig2prim(op, condition, x, y):
    if False:
        for i in range(10):
            print('nop')
    return select(condition, x, y)

@REGISTER_ORIG2PRIM('equal')
def equal_orig2prim(op, x, y):
    if False:
        i = 10
        return i + 15
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return eq(x, y)

@REGISTER_ORIG2PRIM('not_equal')
def ne_orig2prim(op, x, y):
    if False:
        i = 10
        return i + 15
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return ne(x, y)

@REGISTER_ORIG2PRIM('greater_than')
def gt_orig2prim(op, x, y):
    if False:
        while True:
            i = 10
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return gt(x, y)

@REGISTER_ORIG2PRIM('greater_equal')
def ge_orig2prim(op, x, y):
    if False:
        for i in range(10):
            print('nop')
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return ge(x, y)

@REGISTER_ORIG2PRIM('elementwise_pow')
def elementwise_pow_orig2prim(op, x, y):
    if False:
        print('Hello World!')
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    z = primops.pow(x, y)
    return z

@REGISTER_ORIG2PRIM('pow')
def pow_orig2prim(op, x, y):
    if False:
        while True:
            i = 10
    return primops.pow(y, fill_const(op.attr('factor'), y.shape, y.dtype))

@REGISTER_ORIG2PRIM('square')
def square_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    return primops.square(x)

@REGISTER_ORIG2PRIM('elementwise_max')
def elementwise_max_orig2prim(op, x, y):
    if False:
        return 10
    if x.shape != y.shape:
        y = broadcast(y, shape=x.shape)
    return primops.max(x, y)

@REGISTER_ORIG2PRIM('gelu')
def gelu_orig2prim(op, x):
    if False:
        print('Hello World!')
    if op.attr('approximate'):
        cdf = mul(fill_const(0.5, x.shape, x.dtype), add(fill_const(1.0, x.shape, x.dtype), tanh(mul(fill_const(math.sqrt(2 / math.pi), x.shape, x.dtype), add(x, mul(fill_const(0.044715, x.shape, x.dtype), primops.pow(x, fill_const(3.0, x.shape, x.dtype))))))))
        return mul(x, cdf)
    else:
        return mul(mul(fill_const(0.5, x.shape, x.dtype), x), add(fill_const(1.0, x.shape, x.dtype), erf(mul(x, fill_const(1 / math.sqrt(2.0), x.shape, x.dtype)))))

@REGISTER_ORIG2PRIM('dropout')
def dropout_orig2prim(op, seed_t, x):
    if False:
        return 10
    assert seed_t is None, 'Can not lower dropout into prim ops with seedtensor.'
    mask = bernoulli(shape=x.shape, dtype=x.dtype, p=op.attr('dropout_prob'))
    if op.attr('dropout_implementation') == 'upscale_in_train':
        if not op.attr('is_test'):
            out = div(mul(x, mask), fill_const(1.0 - op.attr('dropout_prob'), x.shape, x.dtype))
            return (primops.cast(mask, dtype=paddle.uint8), out)
        else:
            return (primops.cast(mask, dtype=paddle.uint8), x)
    elif op.attr('dropout_implementation') == 'downgrade_in_infer':
        if not op.attr('is_test'):
            return (primops.cast(mask, dtype=paddle.uint8), mul(x, mask))
        else:
            return (primops.cast(mask, dtype=paddle.uint8), mul(x, fill_const(1.0 - op.attr('dropout_prob'), x.shape, x.dtype)))
    else:
        raise RuntimeError('Unsupported dropout_implementation, only support upscale_in_train and downgrade_in_infer')

@REGISTER_ORIG2PRIM('uniform_random')
def uniform_random_orig2prim(op, shape_t, shape_tl):
    if False:
        while True:
            i = 10
    if shape_t or shape_tl:
        raise TypeError('uniform_random_orig2prim currently not support ShapeTensor input or ShapeTensorList input.')
    min_value = op.attr('min')
    max_value = op.attr('max')
    seed = op.attr('seed')
    dtype = paddle.dtype(op.attr('dtype'))
    shape = op.attr('shape')
    return uniform_random(dtype, min_value, max_value, seed, shape=shape)

@REGISTER_ORIG2PRIM('reduce_sum')
def reduce_sum_orig2prim(op, x):
    if False:
        i = 10
        return i + 15
    axes = tuple(range(0, len(x.shape))) if op.attr('reduce_all') else op.attr('dim')
    return reduce_sum(x, axis=axes, keepdim=op.attr('keep_dim'))

@REGISTER_ORIG2PRIM('reduce_mean')
def reduce_mean_orig2prim(op, x):
    if False:
        return 10
    axes = tuple(range(0, len(x.shape))) if op.attr('reduce_all') else op.attr('dim')
    return primops.mean(x, axes, op.attr('keep_dim'))

@REGISTER_ORIG2PRIM('batch_norm')
def batch_norm_orig2prim(op, bias, run_mean, momentum_tensor, scale, run_var, x):
    if False:
        for i in range(10):
            print('nop')
    momentum = op.attr('momentum')
    eps = op.attr('epsilon')
    is_test = op.attr('is_test')
    data_layout = op.attr('data_layout')
    use_global_stats = op.attr('use_global_stats')
    trainable_statistics = op.attr('trainable_statistics')
    reserve_space = None if len(op.output_names) == 5 else get_output_var_list(op)[1]
    feature_axis = 1 if data_layout in ('NC', 'NCL', 'NCHW', 'NCHWD') else len(x.shape) - 1
    use_run_stat = is_test and (not trainable_statistics) or use_global_stats
    return primops.batch_norm(x, feature_axis, scale, bias, run_mean, run_var, eps=eps, momentum=momentum, use_run_stat=use_run_stat, reserve_space=reserve_space)

@REGISTER_ORIG2PRIM('size')
def size_orig2prim(op, x):
    if False:
        while True:
            i = 10
    return fill_const(functools.reduce(operator.mul, x.shape), (), paddle.int64)

@REGISTER_PRIM2ORIG('add_p')
def add_prim2orig(op, x, y):
    if False:
        return 10
    return paddle.add(x, y)

@REGISTER_PRIM2ORIG('sub_p')
def sub_prim2orig(op, x, y):
    if False:
        i = 10
        return i + 15
    return paddle.subtract(x, y)

@REGISTER_PRIM2ORIG('rsqrt_p')
def rsqrt_prim2orig(op, x):
    if False:
        i = 10
        return i + 15
    return paddle.rsqrt(x)

@REGISTER_PRIM2ORIG('mul_p')
def mul_prim2orig(op, x, y):
    if False:
        return 10
    return paddle.multiply(x, y)

@REGISTER_PRIM2ORIG('div_p')
def div_prim2orig(op, x, y):
    if False:
        i = 10
        return i + 15
    return paddle.divide(x, y)

@REGISTER_PRIM2ORIG('sqrt_p')
def sqrt_prim2orig(op, x):
    if False:
        while True:
            i = 10
    return paddle.sqrt(x)

@REGISTER_PRIM2ORIG('tanh_p')
def tanh_prim2orig(op, x):
    if False:
        i = 10
        return i + 15
    return paddle.tanh(x)

@REGISTER_PRIM2ORIG('sin_p')
def sin_prim2orig(op, x):
    if False:
        for i in range(10):
            print('nop')
    return paddle.sin(x)

@REGISTER_PRIM2ORIG('cos_p')
def cos_prim2orig(op, x):
    if False:
        while True:
            i = 10
    return paddle.cos(x)

@REGISTER_PRIM2ORIG('exp_p')
def exp_prim2orig(op, x):
    if False:
        print('Hello World!')
    return paddle.exp(x)

@REGISTER_PRIM2ORIG('erf_p')
def erf_prim2orig(op, x):
    if False:
        return 10
    return paddle.erf(x)

@REGISTER_PRIM2ORIG('abs_p')
def abs_prim2orig(op, x):
    if False:
        i = 10
        return i + 15
    return paddle.abs(x)

@REGISTER_PRIM2ORIG('log_p')
def log_prim2orig(op, x):
    if False:
        return 10
    return paddle.log(x)

@REGISTER_PRIM2ORIG('reshape_p')
def reshape_prim2orig(op, x):
    if False:
        for i in range(10):
            print('nop')
    return paddle.reshape(x, shape=op.attr('shape'))

@REGISTER_PRIM2ORIG('broadcast_p')
def broadcast_prim2orig(op, x):
    if False:
        while True:
            i = 10
    return paddle.broadcast_to(x, shape=op.attr('shape'))

@REGISTER_PRIM2ORIG('transpose_p')
def transpose_prim2orig(op, x):
    if False:
        print('Hello World!')
    return paddle.transpose(x, perm=op.attr('axis'))

@REGISTER_PRIM2ORIG('split_p')
def split_prim2orig(op, x):
    if False:
        print('Hello World!')
    num_or_sections = op.attr('num_or_sections')
    if len(num_or_sections) == 1:
        num_or_sections = num_or_sections[0]
    return paddle.split(x, num_or_sections=num_or_sections, axis=op.attr('axis'))

@REGISTER_PRIM2ORIG('concat_p')
def concat_prim2orig(op, xs):
    if False:
        return 10
    return paddle.concat(xs, axis=op.attr('axis'))

@REGISTER_PRIM2ORIG('reduce_sum_p')
def reduce_prim2orig(op, x):
    if False:
        for i in range(10):
            print('nop')
    return paddle.sum(x, axis=op.attr('axis'), keepdim=op.attr('keepdim'))

@REGISTER_PRIM2ORIG('matmul_p')
def matmul_prim2orig(op, x, y):
    if False:
        i = 10
        return i + 15
    return paddle.matmul(x, y)

@REGISTER_PRIM2ORIG('slice_select_p')
def slice_select_prim2orig(op, x):
    if False:
        while True:
            i = 10
    return paddle.strided_slice(x, axes=op.attr('axis'), starts=op.attr('starts'), ends=op.attr('ends'), strides=op.attr('strides'))

@REGISTER_PRIM2ORIG('slice_assign_p')
def slice_assign_prim2orig(op, x, y):
    if False:
        for i in range(10):
            print('nop')
    x_copy = paddle.assign(x)
    return set_value(x_copy, y, axis=op.attr('axis'), starts=op.attr('starts'), ends=op.attr('ends'), strides=op.attr('strides'), out=x_copy)

@REGISTER_PRIM2ORIG('gather_p')
def gather_prim2orig(op, index_t, x):
    if False:
        print('Hello World!')
    return paddle.gather(x, index_t, axis=op.attr('axis'))

@REGISTER_PRIM2ORIG('scatter_add_p')
def scatter_add_prim2orig(op, index_t, x, y):
    if False:
        while True:
            i = 10
    assert op.attr('axis') == 0, 'Only support axis==0 currently'
    zeros = paddle.zeros_like(x=x, dtype=x.dtype)
    tmp = paddle.scatter(x=zeros, index=index_t, updates=y, overwrite=False)
    return paddle.add(x, tmp)

@REGISTER_PRIM2ORIG('fill_constant_p')
def fill_constant_prim2orig(op):
    if False:
        while True:
            i = 10
    return paddle.full(shape=op.attr('shape'), fill_value=op.attr('value'), dtype=INT_DTYPE_2_STRING[op.attr('dtype')])

@REGISTER_PRIM2ORIG('bernoulli_p')
def bernoulli_prim2orig(op):
    if False:
        i = 10
        return i + 15
    t = paddle.full(shape=op.attr('shape'), fill_value=op.attr('p'), dtype=INT_DTYPE_2_STRING[op.attr('dtype')])
    return paddle.bernoulli(t)

@REGISTER_PRIM2ORIG('uniform_random_p')
def uniform_random_prim2orig(op):
    if False:
        return 10
    return paddle.uniform(shape=op.attr('shape'), dtype=INT_DTYPE_2_STRING[op.attr('dtype')], min=op.attr('min'), max=op.attr('max'), seed=op.attr('seed'))

@REGISTER_PRIM2ORIG('select_p')
def select_prim2orig(op, condition, x, y):
    if False:
        for i in range(10):
            print('nop')
    return paddle.where(condition, x, y)

@REGISTER_PRIM2ORIG('eq_p')
def eq_prim2orig(op, x, y):
    if False:
        for i in range(10):
            print('nop')
    return paddle.equal(x, y)

@REGISTER_PRIM2ORIG('gt_p')
def gt_prim2orig(op, x, y):
    if False:
        i = 10
        return i + 15
    return paddle.greater_than(x, y)

@REGISTER_PRIM2ORIG('ge_p')
def ge_prim2orig(op, x, y):
    if False:
        for i in range(10):
            print('nop')
    return paddle.greater_equal(x, y)

@REGISTER_PRIM2ORIG('ne_p')
def ne_prim2orig(op, x, y):
    if False:
        return 10
    return paddle.not_equal(x, y)

@REGISTER_PRIM2ORIG('pow_p')
def pow_prim2orig(op, x, y):
    if False:
        return 10
    return paddle.pow(x, y)

@REGISTER_PRIM2ORIG('max_p')
def max_prim2orig(op, x, y):
    if False:
        i = 10
        return i + 15
    return paddle.maximum(x, y)

@REGISTER_PRIM2ORIG('cast_p')
def cast_prim2orig(op, x):
    if False:
        i = 10
        return i + 15
    return paddle.cast(x, paddle.dtype(op.attr('dtype')))

@REGISTER_JVP('add_p')
def add_jvp(op, x_dot, y_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return y_dot
    elif y_dot is None:
        return x_dot
    else:
        return linear_jvp(op, x_dot, y_dot)

@REGISTER_JVP('sub_p')
def sub_jvp(op, x_dot, y_dot):
    if False:
        return 10
    if x_dot is None:
        return neg(y_dot)
    elif y_dot is None:
        return x_dot
    else:
        return linear_jvp(op, x_dot, y_dot)

@REGISTER_JVP('mul_p')
def mul_jvp(op, x_dot, y_dot):
    if False:
        i = 10
        return i + 15
    if x_dot is None and y_dot is None:
        return None
    (x, y) = op_position_inputs(op)
    if x_dot is None:
        return mul(x, y_dot)
    elif y_dot is None:
        return mul(x_dot, y)
    else:
        (t1, t2) = (mul(x_dot, y), mul(x, y_dot))
        z_dot = add(t1, t2)
        return z_dot

@REGISTER_JVP('div_p')
def div_jvp(op, x_dot, y_dot):
    if False:
        for i in range(10):
            print('nop')
    if x_dot is None and y_dot is None:
        return None
    (x, y) = op_position_inputs(op)
    if y_dot is None:
        return div(x_dot, y)
    elif x_dot is None:
        return neg(div(mul(x, y_dot), mul(y, y)))
    else:
        t1 = div(x_dot, y)
        t2 = div(mul(x, y_dot), mul(y, y))
        return sub(t1, t2)

@REGISTER_JVP('sqrt_p')
def sqrt_jvp(op, x_dot):
    if False:
        i = 10
        return i + 15
    if x_dot is None:
        return None
    y = op_position_output(op)
    c2 = fill_const(value=2.0, shape=y.shape, dtype=y.dtype)
    y_dot = div(x_dot, mul(c2, y))
    return y_dot

@REGISTER_JVP('tanh_p')
def tanh_jvp(op, x_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    y = op_position_output(op)
    c1 = fill_const(value=1.0, shape=y.shape, dtype=y.dtype)
    y_dot = mul(x_dot, sub(c1, mul(y, y)))
    return y_dot

@REGISTER_JVP('sin_p')
def sin_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    (x,) = op_position_inputs(op)
    return mul(x_dot, cos(x))

@REGISTER_JVP('cos_p')
def cos_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    (x,) = op_position_inputs(op)
    return mul(x_dot, neg(sin(x)))

@REGISTER_JVP('exp_p')
def exp_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    y = op_position_output(op)
    return mul(x_dot, y)

@REGISTER_JVP('erf_p')
def erf_jvp(op, x_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    (x,) = op_position_inputs(op)
    return mul(fill_const(2.0 / math.sqrt(math.pi), x.shape, x.dtype), mul(x_dot, exp(neg(primops.pow(x, fill_const(2.0, x.shape, x.dtype))))))

@REGISTER_JVP('abs_p')
def abs_jvp(op, x_dot):
    if False:
        i = 10
        return i + 15
    if x_dot is None:
        return None
    (x,) = op_position_inputs(op)
    return select(ge(x, fill_const(0.0, x.shape, x.dtype)), x_dot, neg(x_dot))

@REGISTER_JVP('log_p')
def log_jvp(op, x_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    (x,) = op_position_inputs(op)
    return div(x_dot, x)

@REGISTER_JVP('reshape_p')
def reshape_jvp(op, x_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)

@REGISTER_JVP('broadcast_p')
def broadcast_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    shape = op.attr('shape')
    return linear_jvp(op, x_dot, shape=shape)

@REGISTER_JVP('transpose_p')
def transpose_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, axis=axis)

@REGISTER_JVP('split_p')
def split_jvp(op, x_dot):
    if False:
        return 10
    if x_dot is None:
        return None
    num_or_sections = op.attr('num_or_sections')
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, num_or_sections=num_or_sections, axis=axis)

@REGISTER_JVP('concat_p')
def concat_jvp(op, xs_dot):
    if False:
        i = 10
        return i + 15
    if xs_dot is None:
        return None
    axis = op.attr('axis')
    return linear_jvp(op, xs_dot, axis=axis)

@REGISTER_JVP('reduce_sum_p')
def reduce_sum_jvp(op, x_dot):
    if False:
        return 10
    if x_dot is None:
        return None
    axis = op.attr('axis')
    keepdim = op.attr('keepdim')
    return linear_jvp(op, x_dot, axis=axis, keepdim=keepdim)

@REGISTER_JVP('matmul_p')
def matmul_jvp(op, x_dot, y_dot):
    if False:
        print('Hello World!')
    if x_dot is None and y_dot is None:
        return None
    (x, y) = op_position_inputs(op)
    if x_dot is None:
        return matmul(x, y_dot)
    elif y_dot is None:
        return matmul(x_dot, y)
    else:
        t1 = matmul(x, y_dot)
        t2 = matmul(x_dot, y)
        return add(t1, t2)

@REGISTER_JVP('slice_select_p')
def slice_select_jvp(op, x_dot):
    if False:
        return 10
    if x_dot is None:
        return x_dot
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return linear_jvp(op, x_dot, axis=axis, starts=starts, ends=ends, strides=strides)

@REGISTER_JVP('slice_assign_p')
def slice_assign_jvp(op, x_dot, y_dot):
    if False:
        return 10
    (x, y) = op_position_inputs(op)
    assert x_dot is not None or y_dot is not None, "x_dot and y_dot can't be None at the same time. "
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    if x_dot is None:
        return linear_jvp(op, fill_const(value=0.0, shape=x.shape, dtype=x.dtype), y_dot, axis=axis, starts=starts, ends=ends, strides=strides)
    elif y_dot is None:
        return linear_jvp(op, x_dot, fill_const(value=0.0, shape=y.shape, dtype=y.dtype), axis=axis, starts=starts, ends=ends, strides=strides)
    return add(linear_jvp(op, fill_const(value=0.0, shape=x.shape, dtype=x.dtype), y_dot, axis=axis, starts=starts, ends=ends, strides=strides), linear_jvp(op, x_dot, fill_const(value=0.0, shape=y.shape, dtype=y.dtype), axis=axis, starts=starts, ends=ends, strides=strides))

@REGISTER_JVP('gather_p')
def gather_jvp(op, x_dot, indextensor):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    (_, indextensor) = op_position_inputs(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, indextensor, axis=axis)

@REGISTER_JVP('scatter_add_p')
def scatter_add_jvp(op, x_dot, y_dot):
    if False:
        print('Hello World!')
    if x_dot is None:
        return None
    (_, _, indextensor) = op_position_inputs(op)
    axis = op.attr('axis')
    return linear_jvp(op, x_dot, y_dot, indextensor, axis=axis)

@REGISTER_JVP('select_p')
def select_jvp(op, cond_dot, x_dot, y_dot):
    if False:
        i = 10
        return i + 15
    if x_dot is None and y_dot is None:
        return None
    (cond, x, y) = op_position_inputs(op)
    if x_dot is None:
        x_dot = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    if y_dot is None:
        y_dot = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    return select(cond, x_dot, y_dot)

@REGISTER_JVP('eq_p')
def eq_jvp(op, x_dot, y_dot):
    if False:
        while True:
            i = 10
    if x_dot is None and y_dot is None:
        return None
    (x, _) = op_position_inputs(op)
    z_dot = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    return z_dot

@REGISTER_JVP('gt_p')
def gt_jvp(op, x_dot, y_dot):
    if False:
        for i in range(10):
            print('nop')
    if x_dot is None and y_dot is None:
        return None
    (x, _) = op_position_inputs(op)
    z_dot = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    return z_dot

@REGISTER_JVP('ge_p')
def ge_jvp(op, x_dot, y_dot):
    if False:
        return 10
    if x_dot is None and y_dot is None:
        return None
    (x, _) = op_position_inputs(op)
    z_dot = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    return z_dot

@REGISTER_JVP('ne_p')
def ne_jvp(op, x_dot, y_dot):
    if False:
        return 10
    if x_dot is None and y_dot is None:
        return None
    (x, _) = op_position_inputs(op)
    z_dot = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    return z_dot

@REGISTER_JVP('pow_p')
def pow_jvp(op, x_dot, y_dot):
    if False:
        return 10

    def _compute_t1(x, y):
        if False:
            return 10
        zero_y = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
        one_y = fill_const(value=1.0, shape=y.shape, dtype=y.dtype)
        cond = eq(y, zero_y)
        new_y = select(cond, one_y, sub(y, one_y))
        t1 = mul(x_dot, mul(y, primops.pow(x, new_y)))
        return t1
    if x_dot is None and y_dot is None:
        return None
    (x, y) = op_position_inputs(op)
    z = op_position_output(op)
    if y_dot is None:
        return _compute_t1(x, y)
    elif x_dot is None:
        return mul(y_dot, mul(log(x), z))
    else:
        (t1, t2) = (_compute_t1(x, y), mul(y_dot, mul(log(x), z)))
        z_dot = add(t1, t2)
        return z_dot

@REGISTER_JVP('max_p')
def max_jvp(op, x_dot, y_dot):
    if False:
        for i in range(10):
            print('nop')
    if x_dot is None and y_dot is None:
        return None
    (x, y) = op_position_inputs(op)
    z = op_position_output(op)
    z_zeros = fill_const(value=0.0, shape=z.shape, dtype=z.dtype)
    if y_dot is None:
        return select(eq(y, z), z_zeros, x_dot)
    elif x_dot is None:
        return select(eq(y, z), y_dot, z_zeros)
    else:
        return select(eq(y, z), y_dot, x_dot)

@REGISTER_JVP('cast_p')
def cast_jvp(op, x_dot):
    if False:
        return 10
    y = op_position_output(op)
    return primops.cast(x_dot, y.dtype)

@REGISTER_JVP('rsqrt_p')
def rsqrt_jvp(op, x_dot):
    if False:
        while True:
            i = 10
    if x_dot is None:
        return None
    y = op_position_output(op)
    x = op_position_inputs(op)
    c2 = fill_const(value=-2.0, shape=y.shape, dtype=y.dtype)
    y_dot = mul(x_dot, div(div(y, x), c2))
    return y_dot

@REGISTER_TRANSPOSE('add_p')
def add_transpose(op, check_dot, z_bar):
    if False:
        return 10
    (x, y) = op_position_inputs(op)
    assert check_dot(x) or check_dot(y), f'(check_dot(x) or check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    x_bar = z_bar if check_dot(x) else None
    y_bar = z_bar if check_dot(y) else None
    return (x_bar, y_bar)

@REGISTER_TRANSPOSE('sub_p')
def sub_transpose(op, check_dot, z_bar):
    if False:
        return 10
    (x, y) = op_position_inputs(op)
    assert check_dot(x) or check_dot(y), f'(check_dot(x) or check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    x_bar = z_bar if check_dot(x) else None
    y_bar = neg(z_bar) if check_dot(y) else None
    return (x_bar, y_bar)

@REGISTER_TRANSPOSE('mul_p')
def mul_transpose(op, check_dot, z_bar):
    if False:
        while True:
            i = 10
    (x, y) = op_position_inputs(op)
    assert check_dot(x) ^ check_dot(y), f'(check_dot(x) ^ check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    if check_dot(x):
        return (mul(z_bar, y), None)
    else:
        return (None, mul(x, z_bar))

@REGISTER_TRANSPOSE('div_p')
def div_transpose(op, check_dot, z_bar):
    if False:
        print('Hello World!')
    (x, y) = op_position_inputs(op)
    assert not check_dot(y), 'check_dot(y) must be False'
    x_bar = div(z_bar, y) if check_dot(x) else None
    return (x_bar, None)

@REGISTER_TRANSPOSE('reshape_p')
def reshape_transpose(op, check_dot, y_bar):
    if False:
        for i in range(10):
            print('nop')
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    return reshape(y_bar, shape=x.shape)

@REGISTER_TRANSPOSE('broadcast_p')
def broadcast_transpose(op, check_dot, y_bar):
    if False:
        for i in range(10):
            print('nop')
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    bat = len(y_bar.shape) - len(x.shape)
    axis = list(range(bat))
    keepdim = [bat + i for (i, s) in enumerate(x.shape) if s == 1]
    axis += keepdim
    out = reduce_sum(y_bar, axis=axis, keepdim=False)
    return reshape(out, x.shape)

@REGISTER_TRANSPOSE('transpose_p')
def transpose_transpose(op, check_dot, y_bar):
    if False:
        print('Hello World!')
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    axis = op.attr('axis')
    reordered = sorted(((k, i) for (i, k) in enumerate(axis)))
    axis = [i for (k, i) in reordered]
    return transpose(y_bar, axis=axis)

@REGISTER_TRANSPOSE('split_p')
def split_transpose(op, check_dot, ys_bar):
    if False:
        return 10
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    return concat(ys_bar, axis=op.attr('axis'))

@REGISTER_TRANSPOSE('concat_p')
def concat_transpose(op, check_dot, y_bar):
    if False:
        while True:
            i = 10
    (xs,) = op_position_inputs(op)
    if not isinstance(xs, typing.Sequence):
        xs = [xs]
    for x in xs:
        assert check_dot(x), 'check_dot(x) must be True'
    axis = op.attr('axis')
    sections = [x.shape[axis] for x in xs]
    if len(sections) == 1:
        return y_bar
    return split(y_bar, num_or_sections=sections, axis=axis)

@REGISTER_TRANSPOSE('reduce_sum_p')
def reduce_sum_transpose(op, check_dot, y_bar):
    if False:
        for i in range(10):
            print('nop')
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    axes = op.attr('axis')
    shape = tuple((1 if i in axes else size for (i, size) in enumerate(x.shape)))
    t = reshape(y_bar, shape=shape)
    return broadcast(t, shape=x.shape)

@REGISTER_TRANSPOSE('matmul_p')
def matmul_transpose(op, check_dot, z_bar):
    if False:
        return 10
    (x, y) = op_position_inputs(op)
    assert check_dot(x) ^ check_dot(y), f'(check_dot(x) ^ check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    axis = [1, 0] if len(x.shape) == 2 else [0, 2, 1]
    if check_dot(x):
        return (matmul(z_bar, transpose(y, axis=axis)), None)
    else:
        return (None, matmul(transpose(x, axis=axis), z_bar))

@REGISTER_TRANSPOSE('slice_select_p')
def slice_select_transpose(op, check_dot, y_bar):
    if False:
        print('Hello World!')
    (x,) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    zeros = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    return slice_assign(zeros, y_bar, axis=axis, starts=starts, ends=ends, strides=strides)

@REGISTER_TRANSPOSE('slice_assign_p')
def slice_assign_transpose(op, check_dot, z_bar):
    if False:
        i = 10
        return i + 15
    (x, y) = op_position_inputs(op)
    assert check_dot(x) ^ check_dot(y), f'(check_dot(x) ^ check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    zeros = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    axis = op.attr('axis')
    starts = op.attr('starts')
    ends = op.attr('ends')
    strides = op.attr('strides')
    if check_dot(x):
        return (slice_assign(z_bar, zeros, axis=axis, starts=starts, ends=ends, strides=strides), None)
    return (None, slice_select(z_bar, axis=axis, starts=starts, ends=ends, strides=strides))

@REGISTER_TRANSPOSE('gather_p')
def gather_transpose(op, check_dot, y_bar):
    if False:
        for i in range(10):
            print('nop')
    (x, indextensor) = op_position_inputs(op)
    assert check_dot(x), 'check_dot(x) must be True'
    axis = op.attr('axis')
    zeros = fill_const(0.0, x.shape, x.dtype)
    x_bar = scatter_add(zeros, y_bar, indextensor, axis=axis)
    indextensor_bar = None
    return (x_bar, indextensor_bar)

@REGISTER_TRANSPOSE('scatter_add_p')
def scatter_add_transpose(op, check_dot, z_bar):
    if False:
        i = 10
        return i + 15
    (x, y, indextensor) = op_position_inputs(op)
    assert check_dot(x) and check_dot(y), f'(check_dot(x) and check_dot(y)) must be True, but check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    axis = op.attr('axis')
    zeros = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    x_bar = scatter_add(z_bar, zeros, indextensor, axis=axis)
    y_bar = gather(z_bar, indextensor, axis=axis)
    indextensor_bar = None
    return (x_bar, y_bar, indextensor_bar)

@REGISTER_TRANSPOSE('select_p')
def select_transpose(op, check_dot, z_bar):
    if False:
        for i in range(10):
            print('nop')
    (cond, x, y) = op_position_inputs(op)
    assert check_dot(cond) or check_dot(x) or check_dot(y), f'check_dot(cond) ^ (check_dot(x) ^ check_dot(y)) must be True, but check_dot(cond)={check_dot(cond)}, check_dot(x)={check_dot(x)} and check_dot(y)={check_dot(y)}.'
    zeros_x = fill_const(value=0.0, shape=x.shape, dtype=x.dtype)
    zeros_y = fill_const(value=0.0, shape=y.shape, dtype=y.dtype)
    cond_bar = fill_const(value=0.0, shape=y.shape, dtype=cond.dtype) if check_dot(cond) else None
    x_bar = select(cond, z_bar, zeros_x) if check_dot(x) else None
    y_bar = select(cond, zeros_y, z_bar) if check_dot(y) else None
    return (cond_bar, x_bar, y_bar)

@REGISTER_TRANSPOSE('cast_p')
def cast_transpose(op, check_dot, y_bar):
    if False:
        for i in range(10):
            print('nop')
    (x,) = op_position_inputs(op)
    return primops.cast(y_bar, x.dtype)