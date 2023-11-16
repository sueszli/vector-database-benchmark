import paddle
from paddle import base

def add_fn(x):
    if False:
        return 10
    x = x + 1
    return x

def loss_fn(x, label):
    if False:
        print('Hello World!')
    loss = paddle.nn.functional.cross_entropy(x, label, reduction='none', use_softmax=False)
    return loss

def dyfunc_empty_nonlocal(x):
    if False:
        for i in range(10):
            print('nop')
    flag = True
    if flag:
        print("It's a test for empty nonlocal stmt")
    if paddle.mean(x) < 0:
        x + 1
    out = x * 2
    return out

def dyfunc_with_if_else(x_v, label=None):
    if False:
        print('Hello World!')
    if paddle.mean(x_v).numpy() > 5:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    if label is not None:
        loss = paddle.nn.functional.cross_entropy(x_v, label, reduction='none', use_softmax=False)
        return loss
    return x_v

def dyfunc_with_if_else2(x, col=100):
    if False:
        while True:
            i = 10
    row = 0
    if abs(col) > x.shape[-1]:
        col = paddle.tensor.fill_constant(shape=[1], value=-1, dtype='int64')
    if paddle.mean(x).numpy() > x.numpy()[row][col]:
        y = paddle.nn.functional.relu(x)
    else:
        x_pow = paddle.pow(x, 2)
        y = paddle.tanh(x_pow)
    return y

def dyfunc_with_if_else3(x):
    if False:
        return 10
    "\n    q = paddle.jit.dy2static.UndefinedVar('q')\n    z = paddle.jit.dy2static.UndefinedVar('z')\n\n    def true_fn_0(q, x, y):\n        x = x + 1\n        z = x + 2\n        q = x + 3\n        return q, x, y, z\n\n    def false_fn_0(q, x, y):\n        y = y + 1\n        z = x - 2\n        m = x + 2\n        n = x + 3\n        return q, x, y, z\n    q, x, y, z = paddle.static.nn.cond(paddle.mean(x) < 5, lambda :\n        paddle.jit.dy2static.convert_call(true_fn_0)(q, x, y),\n        lambda : paddle.jit.dy2static.convert_call(false_fn_0)(q,\n        x, y))\n    "
    y = x + 1
    if paddle.mean(x).numpy() < 5:
        x = x + 1
        z = x + 2
        q = x + 3
    else:
        y = y + 1
        z = x - 2
        m = x + 2
        n = x + 3
    q = q + 1
    n = q + 2
    x = n
    return x

def dyfunc_with_if_else_early_return1():
    if False:
        for i in range(10):
            print('nop')
    x = paddle.to_tensor([10])
    if x == 0:
        a = paddle.zeros([2, 2])
        b = paddle.zeros([3, 3])
        return (a, b)
    a = paddle.zeros([2, 2]) + 1
    return (a, None)

def dyfunc_with_if_else_early_return2():
    if False:
        for i in range(10):
            print('nop')
    x = paddle.to_tensor([10])
    if x == 0:
        a = paddle.zeros([2, 2])
        b = paddle.zeros([3, 3])
        return (a, b)
    elif x == 1:
        c = paddle.zeros([2, 2]) + 1
        d = paddle.zeros([3, 3]) + 1
        return (c, d)
    e = paddle.zeros([2, 2]) + 3
    return (e, None)

def dyfunc_with_if_else_with_list_generator(x):
    if False:
        while True:
            i = 10
    if 10 > 5:
        y = paddle.add_n([paddle.full(shape=[2], fill_value=v) for v in range(5)])
    else:
        y = x
    return y

def nested_if_else(x_v):
    if False:
        while True:
            i = 10
    batch_size = 16
    feat_size = x_v.shape[-1]
    bias = paddle.tensor.fill_constant([feat_size], dtype='float32', value=1)
    if x_v.shape[0] != batch_size:
        batch_size = paddle.shape(x_v)[0]
    if paddle.mean(x_v).numpy() < 0:
        y = x_v + bias
        w = paddle.tensor.fill_constant([feat_size], dtype='float32', value=10)
        if y.numpy()[0] < 10:
            tmp = y * w
            y = paddle.nn.functional.relu(tmp)
            if paddle.mean(y).numpy() < batch_size:
                y = paddle.abs(y)
            else:
                tmp = paddle.tensor.fill_constant(y.shape, dtype='float32', value=-1)
                y = y - tmp
    else:
        y = x_v - bias
    return y

def nested_if_else_2(x):
    if False:
        return 10
    y = paddle.reshape(x, [-1, 1])
    b = 2
    if b < 1:
        z = y
    x_shape_0 = x.shape[0]
    if x_shape_0 < 1:
        if paddle.shape(y).numpy()[0] < 1:
            res = paddle.tensor.fill_constant(value=2, shape=x.shape, dtype='int32')
            z = y + 1
        else:
            res = paddle.tensor.fill_constant(value=3, shape=x.shape, dtype='int32')
    else:
        res = x
    return res

def nested_if_else_3(x):
    if False:
        for i in range(10):
            print('nop')
    y = paddle.reshape(x, [-1, 1])
    b = 2
    if b < 1:
        z = y
    else:
        z = x
    if b < 1:
        res = x
        if b > 1:
            out = x + 1
        else:
            out = x - 1
    else:
        y_shape = paddle.shape(y)
        if y_shape.numpy()[0] < 1:
            res = paddle.tensor.fill_constant(value=2, shape=x.shape, dtype='int32')
            z = y + 1
        else:
            res = paddle.tensor.fill_constant(value=3, shape=x.shape, dtype='int32')
            out = x + 1
    return res

class NetWithControlFlowIf(paddle.nn.Layer):

    def __init__(self, hidden_dim=16):
        if False:
            while True:
                i = 10
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = paddle.nn.Linear(in_features=hidden_dim, out_features=5, weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.99)), bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.5)))
        self.alpha = 10.0
        self.constant_vars = {}

    @paddle.jit.to_static
    def forward(self, input):
        if False:
            i = 10
            return i + 15
        hidden_dim = input.shape[-1]
        if hidden_dim != self.hidden_dim:
            raise ValueError('hidden_dim {} of input is not equal to FC.weight[0]: {}'.format(hidden_dim, self.hidden_dim))
        self.constant_vars['bias'] = paddle.tensor.fill_constant([5], dtype='float32', value=1)
        fc_out = self.fc(input)
        if paddle.mean(fc_out).numpy() < 0:
            y = fc_out + self.constant_vars['bias']
            self.constant_vars['w'] = paddle.tensor.fill_constant([5], dtype='float32', value=10)
            if y.numpy()[0] < self.alpha:
                x = 10
                tmp = y * self.constant_vars['w']
                y = paddle.nn.functional.relu(tmp)
                if y.numpy()[-1] < self.alpha:
                    self.constant_vars['w'] = paddle.tensor.fill_constant([hidden_dim], dtype='float32', value=9)
                    y = paddle.abs(y)
                else:
                    tmp = paddle.tensor.fill_constant(y.shape, dtype='float32', value=-1)
                    y = y - tmp
        else:
            y = fc_out - self.constant_vars['bias']
        loss = paddle.mean(y)
        return loss

def if_with_and_or(x_v, label=None):
    if False:
        return 10
    batch_size = paddle.shape(x_v)
    if x_v is not None and (paddle.mean(x_v).numpy() > 0 or label is not None) and (batch_size[0] > 1) and True:
        x_v = x_v - 1
    else:
        x_v = x_v + 1
    if label is not None:
        loss = paddle.nn.functional.cross_entropy(x_v, label, reduction='none', use_softmax=False)
        return loss
    return x_v

def if_with_and_or_1(x, y=None):
    if False:
        return 10
    batch_size = paddle.shape(x)
    if batch_size[0] > 1 and y is not None:
        x = x + 1
    if y is not None or batch_size[0] > 1:
        x = x - 1
    return x

def if_with_and_or_2(x, y=None):
    if False:
        while True:
            i = 10
    batch_size = paddle.shape(x)
    if x is not None and batch_size[0] > 1 and (y is not None):
        x = x + 1
    if batch_size[0] > 1 or y is not None or x is not None:
        x = x - 1
    return x

def if_with_and_or_3(x, y=None):
    if False:
        return 10
    batch_size = paddle.shape(x)
    mean_res = paddle.mean(x)
    if x is not None and batch_size[0] > 1 and (y is not None) and (mean_res.numpy() > 0):
        x = x + 1
    if mean_res.numpy() > 0 and (x is not None and batch_size[0] > 1) and y:
        x = x - 1
    return x

def if_with_and_or_4(x, y=None):
    if False:
        i = 10
        return i + 15
    batch_size = paddle.shape(x)
    mean_res = paddle.mean(x)
    if x is not None and batch_size[0] > 1 or (y is not None and mean_res.numpy() > 0):
        x = x + 1
    if (x is not None or batch_size[0] > 1) and (y is not None or mean_res.numpy() > 0):
        x = x - 1
    return x

def if_with_class_var(x, y=None):
    if False:
        print('Hello World!')

    class Foo:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.a = 1
            self.b = 2
    foo = Foo()
    batch_size = paddle.shape(x)
    mean_res = paddle.mean(x)
    if batch_size[0] > foo.a:
        x = x + foo.b
    else:
        x = x - foo.b
    return x

def if_tensor_case(x):
    if False:
        print('Hello World!')
    x = base.dygraph.to_variable(x)
    mean = paddle.mean(x)
    if mean:
        for i in range(0, 10):
            if i > 5:
                x += 1
                break
            x += 1
    else:
        for i in range(0, 37):
            x += 1
            break
            x += i
    if paddle.mean(x) + 1 and mean > 1 and (x is not None) or 2 > 1:
        x -= 1
    if not (x[0][0] and (mean * x)[0][0]):
        x += 1
    return x

def dyfunc_ifelse_ret_int1(x):
    if False:
        for i in range(10):
            print('nop')
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return (y, index)
    else:
        y = x[index] + 2
        index = index + 1
        return (y, index)

def dyfunc_ifelse_ret_int2(x):
    if False:
        i = 10
        return i + 15
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return (y, index)
    else:
        y = x[index] + 2
        index = index + 1
        return y

def dyfunc_ifelse_ret_int3(x):
    if False:
        i = 10
        return i + 15
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return index
    else:
        y = x[index] + 2
        return y

def dyfunc_ifelse_ret_int4(x):
    if False:
        print('Hello World!')
    index = 0
    pred = paddle.to_tensor([1])
    if pred:
        y = x[index] + 1
        index = index + 1
        return 'unsupport ret'
    else:
        y = x[index] + 2
        return y