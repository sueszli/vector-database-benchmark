import numpy as np
import jittor as jt
import torch
import time
import jittor.models as jtmodels
import torchvision.models as tcmodels
import os
jt.flags.use_cuda = 1
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
jt.cudnn.set_algorithm_cache_size(10000)
threshold = 0.001
models = ['squeezenet1_1', 'alexnet', 'resnet50', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'vgg11', 'wide_resnet50_2', 'wide_resnet101_2']

def to_cuda(x):
    if False:
        return 10
    if jt.has_cuda:
        return x.cuda()
    return x

def test_allmodels(bs=1):
    if False:
        return 10
    test_img = np.random.random((bs, 3, 224, 224)).astype('float32')
    pytorch_test_img = to_cuda(torch.Tensor(test_img))
    jittor_test_img = jt.array(test_img)
    for model in models:
        if model == 'inception_v3':
            test_img = np.random.random((bs, 3, 300, 300)).astype('float32')
            pytorch_test_img = to_cuda(torch.Tensor(test_img))
            jittor_test_img = jt.array(test_img)
        jittor_test_img.stop_grad()
        pytorch_test_img.requires_grad = False
        pytorch_model = to_cuda(tcmodels.__dict__[model]())
        jittor_model = jtmodels.__dict__[model]()
        pytorch_model.eval()
        jittor_model.eval()
        jittor_model.load_parameters(pytorch_model.state_dict())
        total = 512
        warmup = max(2, total // bs // 8)
        rerun = max(2, total // bs)
        print('=' * 20 + model + '=' * 20)
        for i in range(warmup):
            jittor_result = jittor_model(jittor_test_img)
        jt.sync_all(True)
        sta = time.time()
        for i in range(rerun):
            jittor_result = jittor_model(jittor_test_img)
            jittor_result.sync()
        jt.sync_all(True)
        end = time.time()
        print(f'- Jittor {model} forward average time cost: {round((time.time() - sta) / rerun, 5)}, Batch Size: {bs}, FPS: {round(bs * rerun / (end - sta), 2)}')
        for i in range(warmup):
            pytorch_result = pytorch_model(pytorch_test_img)
        torch.cuda.synchronize()
        sta = time.time()
        for i in range(rerun):
            pytorch_result = pytorch_model(pytorch_test_img)
        torch.cuda.synchronize()
        end = time.time()
        print(f'- Pytorch {model} forward average time cost: {round((end - sta) / rerun, 5)}, Batch Size: {bs}, FPS: {round(bs * rerun / (end - sta), 2)}')
        x = pytorch_result.detach().cpu().numpy() + 1
        y = jittor_result.numpy() + 1
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < threshold, f'[*] {model} forward fails..., Relative Error: {diff}'
        print(f'[*] {model} forword passes with Relative Error {diff}')
        torch.cuda.empty_cache()
        jt.clean()
        jt.gc()
with torch.no_grad():
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        test_allmodels(bs)