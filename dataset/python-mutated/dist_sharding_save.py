import os
from dist_mnist import cnn_model
import paddle
from paddle import base
from paddle.distributed.fleet.base import role_maker
from paddle.distributed.fleet.meta_optimizers import sharding
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1

def runtime_main():
    if False:
        return 10
    from test_dist_base import dump_output
    from paddle.distributed import fleet
    paddle.enable_static()
    train_prog = paddle.base.Program()
    startup_prog = paddle.base.Program()
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)
    with base.program_guard(train_prog, startup_prog):
        with base.unique_name.guard():
            input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
            input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
            fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
            fc_2 = paddle.static.nn.fc(x=fc_1, size=256, activation='tanh')
            prediction = paddle.static.nn.fc(x=[fc_2], size=2, activation='softmax')
            cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
            avg_cost = paddle.mean(x=cost)
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.sharding = True
            strategy.sharding_configs = {'sharding_segment_strategy': 'segment_broadcast_MB', 'segment_broadcast_MB': 0.2, 'sharding_degree': 2}
            optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
            optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)
    device_id = int(os.getenv('FLAGS_selected_gpus', '0'))
    place = base.CUDAPlace(device_id)
    exe = base.Executor(place)
    exe.run(startup_prog)
    dirname = './ut_sharding_save_model'
    sharding.utils.save_persistables(exe, dirname, main_program=train_prog, filename=None)
    out_losses = []
    dump_output(out_losses)
if __name__ == '__main__':
    runtime_main()