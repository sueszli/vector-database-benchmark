import os
import numpy as np
os.environ['FLAGS_check_nan_inf'] = '1'
import paddle
from paddle import base
from paddle.base import core
paddle.enable_static()
np.random.seed(0)

def generator():
    if False:
        while True:
            i = 10
    batch_size = 5
    for i in range(5):
        curr_train_x = np.random.randint(batch_size, size=(batch_size, 3)).astype('float32')
        if i >= 2:
            curr_train_x[0, :] = np.nan
            curr_train_x[-1, :] = np.inf
        res = []
        for i in range(batch_size):
            y = i % 3
            res.append([y])
        y_label = np.array(res).astype('int64')
        yield [curr_train_x, y_label]

def net():
    if False:
        for i in range(10):
            print('nop')
    x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
    y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
    zero = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
    fp16_zero = paddle.cast(zero, dtype='float16')
    y = y + zero
    hidden = x
    hidden = paddle.static.nn.fc(x=hidden, size=400, activation='sigmoid')
    hidden = paddle.static.nn.fc(x=hidden, size=3)
    (cost, y_predict) = paddle.nn.functional.softmax_with_cross_entropy(hidden, y, return_softmax=True)
    acc_top1 = paddle.static.accuracy(input=y_predict, label=y, k=1)
    avg_cost = paddle.mean(cost)
    sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.05)
    sgd_optimizer.minimize(avg_cost)
    return (y_predict, avg_cost, acc_top1)

def check(use_cuda):
    if False:
        i = 10
        return i + 15
    main = base.Program()
    startup = base.Program()
    scope = base.core.Scope()
    with base.scope_guard(scope):
        with base.program_guard(main, startup):
            (y_predict, avg_cost, acc_top1) = net()
            place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
            exe = base.Executor(place)
            exe.run(startup)
            step = 0.0
            for (train_data, y_label) in generator():
                outs = exe.run(main, feed={'x': train_data, 'y': y_label}, fetch_list=[y_predict.name, avg_cost.name, acc_top1.name])
                step += 1
                print(f'iter={step:.0f},cost={outs[1]},acc1={outs[2]}')
if __name__ == '__main__':
    try:
        check(use_cuda=False)
        raise AssertionError()
    except Exception as e:
        print(e)
        print(type(e))
        assert type(e) == RuntimeError
    if core.is_compiled_with_cuda():
        try:
            check(use_cuda=True)
            raise AssertionError()
        except Exception as e:
            print(e)
            print(type(e))
            assert type(e) == OSError or type(e) == RuntimeError