import argparse
import logging
import time
from caffe2.python import workspace, data_parallel_model
from caffe2.python import cnn
import caffe2.python.models.resnet as resnet
'\nSimple benchmark that creates a data-parallel resnet-50 model\nand measures the time.\n'
logging.basicConfig()
log = logging.getLogger('net_construct_bench')
log.setLevel(logging.DEBUG)

def AddMomentumParameterUpdate(train_model, LR):
    if False:
        while True:
            i = 10
    '\n    Add the momentum-SGD update.\n    '
    params = train_model.GetParams()
    assert len(params) > 0
    ONE = train_model.param_init_net.ConstantFill([], 'ONE', shape=[1], value=1.0)
    NEGONE = train_model.param_init_net.ConstantFill([], 'NEGONE', shape=[1], value=-1.0)
    for param in params:
        param_grad = train_model.param_to_grad[param]
        param_momentum = train_model.param_init_net.ConstantFill([param], param + '_momentum', value=0.0)
        train_model.net.MomentumSGD([param_grad, param_momentum, LR], [param_grad, param_momentum], momentum=0.9, nesterov=1)
        train_model.WeightedSum([param, ONE, param_grad, NEGONE], param)

def Create(args):
    if False:
        i = 10
        return i + 15
    gpus = list(range(args.num_gpus))
    log.info('Running on gpus: {}'.format(gpus))
    train_model = cnn.CNNModelHelper(order='NCHW', name='resnet50', use_cudnn=True, cudnn_exhaustive_search=False)

    def create_resnet50_model_ops(model, loss_scale):
        if False:
            while True:
                i = 10
        [softmax, loss] = resnet.create_resnet50(model, 'data', num_input_channels=3, num_labels=1000, label='label')
        model.Accuracy([softmax, 'label'], 'accuracy')
        return [loss]

    def add_parameter_update_ops(model):
        if False:
            for i in range(10):
                print('nop')
        model.AddWeightDecay(0.0001)
        ITER = model.Iter('ITER')
        stepsz = int(30)
        LR = model.net.LearningRate([ITER], 'LR', base_lr=0.1, policy='step', stepsize=stepsz, gamma=0.1)
        AddMomentumParameterUpdate(model, LR)

    def add_image_input(model):
        if False:
            print('Hello World!')
        pass
    start_time = time.time()
    data_parallel_model.Parallelize_GPU(train_model, input_builder_fun=add_image_input, forward_pass_builder_fun=create_resnet50_model_ops, param_update_builder_fun=add_parameter_update_ops, devices=gpus)
    ct = time.time() - start_time
    train_model.net._CheckLookupTables()
    log.info('Model create for {} gpus took: {} secs'.format(len(gpus), ct))

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Caffe2: Benchmark for net construction')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs.')
    args = parser.parse_args()
    Create(args)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    import cProfile
    cProfile.run('main()', sort='cumulative')