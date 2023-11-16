from functools import reduce
import nets
from test_dist_base import TestDistRunnerBase, runtime_main
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()
DTYPE = 'float32'
paddle.dataset.mnist.fetch()
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1

def cnn_model(data):
    if False:
        for i in range(10):
            print('nop')
    conv_pool_1 = nets.simple_img_conv_pool(input=data, filter_size=5, num_filters=20, pool_size=2, pool_stride=2, act='relu', param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    conv_pool_2 = nets.simple_img_conv_pool(input=conv_pool_1, filter_size=5, num_filters=50, pool_size=2, pool_stride=2, act='relu', param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    SIZE = 10
    input_shape = conv_pool_2.shape
    param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
    scale = (2.0 / (param_shape[0] ** 2 * SIZE)) ** 0.5
    predict = paddle.static.nn.fc(x=conv_pool_2, size=SIZE, activation='softmax', weight_attr=base.param_attr.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)))
    return predict

class TestFleetMetaOptimizerFuseAllReducePrecision(TestDistRunnerBase):

    def get_model(self, batch_size=2, single_device=False):
        if False:
            print('Hello World!')
        images = paddle.static.data(name='pixel', shape=[-1, 1, 28, 28], dtype=DTYPE)
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        predict = cnn_model(images)
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        batch_size_tensor = paddle.tensor.create_tensor(dtype='int64')
        batch_acc = paddle.static.accuracy(input=predict, label=label, total=batch_size_tensor)
        test_program = base.default_main_program().clone(for_test=True)
        train_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=batch_size)
        optimizer = paddle.optimizer.Adam(0.01)
        if single_device:
            optimizer.minimize(avg_cost)
        else:
            role = role_maker.PaddleCloudRoleMaker(is_collective=True)
            fleet.init(role)
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.without_graph_optimization = True
            strategy.fuse_all_reduce_ops = True
            strategy._calc_comm_same_stream = False
            strategy.fuse_grad_size_in_num = 8
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)
        return (test_program, avg_cost, train_reader, test_reader, batch_acc, predict)
if __name__ == '__main__':
    runtime_main(TestFleetMetaOptimizerFuseAllReducePrecision)