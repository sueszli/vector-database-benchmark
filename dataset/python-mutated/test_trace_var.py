import unittest
import jittor as jt
import numpy as np
from jittor import Module
from jittor.models import resnet
import pickle
from PIL import Image
import platform
f32 = jt.float32

def matmul(a, b):
    if False:
        i = 10
        return i + 15
    ((n, m), k) = (a.shape, b.shape[-1])
    a = a.broadcast([n, m, k], dims=[2])
    b = b.broadcast([n, m, k], dims=[0])
    return (a * b).sum(dim=1)

def relu(x):
    if False:
        i = 10
        return i + 15
    return jt.maximum(x, 0.0)
Relu = jt.make_module(relu)

class Model(Module):

    def __init__(self, input_size):
        if False:
            print('Hello World!')
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)

    def execute(self, x):
        if False:
            while True:
                i = 10
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

def print_stack_tree(data):
    if False:
        for i in range(10):
            print('nop')
    tree = {}
    for n in data['node_data'].values():
        p = tree
        for s in n['stacks']:
            name = s['name']
            if name not in p:
                p[name] = {}
            p = p[name]
    from pprint import pprint
    pprint(tree)

class Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        if False:
            print('Hello World!')
        self.w = (jt.random((in_features, out_features)) - 0.5) / in_features ** 0.5
        self.b = jt.random((out_features,)) - 0.5 if bias else None

    def execute(self, x):
        if False:
            while True:
                i = 10
        x = matmul(x, self.w)
        if self.b is not None:
            return x + self.b
        return x

class TestTraceVar(unittest.TestCase):

    def test_simple_model(self):
        if False:
            i = 10
            return i + 15
        with jt.flag_scope(trace_py_var=2):
            model = Model(input_size=1)
            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            y.sync()
            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f'{jt.flags.cache_path}/simple_model.pkl', 'wb') as f:
                pickle.dump(data, f)

    def test_simple_model_train(self):
        if False:
            for i in range(10):
                print('nop')
        with jt.flag_scope(trace_py_var=2):
            model = Model(input_size=1)
            opt = jt.optim.SGD(model.parameters(), 0.1)
            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            opt.step(y ** 2)
            jt.sync_all()
            data = jt.dump_trace_data()
            jt.clear_trace_data()
            for (k, v) in data['execute_op_info'].items():
                for i in v['fused_ops']:
                    if i not in data['node_data']:
                        assert 0, (i, 'not found')
            for (k, v) in list(data['node_data'].items()):
                if v['attrs']['name'] == 'unname':
                    assert 0
            print(len(data['node_data']))
            with open(f'{jt.flags.cache_path}/simple_model_train.pkl', 'wb') as f:
                pickle.dump(data, f)

    def test_resnet_infer(self):
        if False:
            i = 10
            return i + 15
        with jt.flag_scope(trace_py_var=2):
            resnet18 = resnet.Resnet18()
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            y.sync()
            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f'{jt.flags.cache_path}/resnet.pkl', 'wb') as f:
                pickle.dump(data, f)
            for (k, v) in data['execute_op_info'].items():
                for i in v['fused_ops']:
                    if i not in data['node_data']:
                        assert 0, (i, 'not found')

    def test_resnet_infer_with_feature(self):
        if False:
            print('Hello World!')
        cat_url = 'https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=3782485413,1118109468&fm=26&gp=0.jpg'
        import jittor_utils
        cat_path = f'{jt.flags.cache_path}/cat.jpg'
        print('download')
        jittor_utils.download(cat_url, cat_path)
        with open(cat_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = jt.array(np.array(img))
            print(img.shape, img.dtype)
            img = ((img.float() - 128) / 255).transpose(2, 0, 1)
        with jt.flag_scope(trace_py_var=2, trace_var_data=1):
            img = img[None, ...]
            resnet18 = resnet.Resnet18(pretrained=True)
            x = jt.float32(img)
            y = resnet18(x)
            y.sync()
            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f'{jt.flags.cache_path}/resnet_with_feature.pkl', 'wb') as f:
                pickle.dump(data, f)
            for (k, v) in data['execute_op_info'].items():
                for i in v['fused_ops']:
                    if i not in data['node_data']:
                        assert 0, (i, 'not found')

    def test_resnet_trainx(self):
        if False:
            return 10
        with jt.flag_scope(trace_py_var=2):
            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            opt.step(y ** 2)
            jt.sync_all()
            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f'{jt.flags.cache_path}/resnet_train.pkl', 'wb') as f:
                pickle.dump(data, f)
            for (k, v) in data['execute_op_info'].items():
                for i in v['fused_ops']:
                    if i not in data['node_data']:
                        assert 0, (i, 'not found')
            for (k, v) in data['node_data'].items():
                if 'name' not in v['attrs']:
                    print(v)

    def test_resnet_train_profile(self):
        if False:
            while True:
                i = 10
        with jt.profile_scope(trace_py_var=1):
            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            opt.step(y ** 2)
            jt.sync_all()
if __name__ == '__main__':
    unittest.main()