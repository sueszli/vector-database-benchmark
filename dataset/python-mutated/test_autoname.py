import torch
import pyro
import pyro.distributions.torch as dist
import pyro.poutine as poutine
from pyro.contrib.autoname import autoname, sample

def test_basic_scope():
    if False:
        for i in range(10):
            print('nop')

    @autoname
    def f1():
        if False:
            print('Hello World!')
        sample(dist.Normal(0, 1))
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            for i in range(10):
                print('nop')
        sample('x', dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))
    tr1 = poutine.trace(f1).get_trace()
    assert 'f1/Normal' in tr1.nodes
    assert 'f1/Bernoulli' in tr1.nodes
    tr2 = poutine.trace(f2).get_trace()
    assert 'model/x' in tr2.nodes
    assert 'model/Normal' in tr2.nodes

def test_repeat_names():
    if False:
        for i in range(10):
            print('nop')

    @autoname
    def f1():
        if False:
            while True:
                i = 10
        sample(dist.Normal(0, 1))
        sample(dist.Normal(0, 1))
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            while True:
                i = 10
        sample('x', dist.Bernoulli(0.5))
        sample('x', dist.Bernoulli(0.5))
        sample('x', dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))
    tr1 = poutine.trace(f1).get_trace()
    assert 'f1/Normal' in tr1.nodes
    assert 'f1/Normal1' in tr1.nodes
    assert 'f1/Bernoulli' in tr1.nodes
    tr2 = poutine.trace(f2).get_trace()
    assert 'model/x' in tr2.nodes
    assert 'model/x1' in tr2.nodes
    assert 'model/x2' in tr2.nodes
    assert 'model/Normal' in tr2.nodes

def test_compose_scopes():
    if False:
        i = 10
        return i + 15

    @autoname
    def f1():
        if False:
            print('Hello World!')
        return sample(dist.Bernoulli(0.5))

    @autoname
    def f2():
        if False:
            for i in range(10):
                print('nop')
        f1()
        return sample(dist.Bernoulli(0.5))

    @autoname
    def f3():
        if False:
            i = 10
            return i + 15
        f1()
        f1()
        f1()
        f2()
        return sample(dist.Normal(0, 1))
    tr1 = poutine.trace(f1).get_trace()
    assert 'f1/Bernoulli' in tr1.nodes
    tr2 = poutine.trace(f2).get_trace()
    assert 'f2/f1/Bernoulli' in tr2.nodes
    assert 'f2/Bernoulli' in tr2.nodes
    tr3 = poutine.trace(f3).get_trace()
    assert 'f3/f1/Bernoulli' in tr3.nodes
    assert 'f3/f1__1/Bernoulli' in tr3.nodes
    assert 'f3/f1__2/Bernoulli' in tr3.nodes
    assert 'f3/f2/f1/Bernoulli' in tr3.nodes
    assert 'f3/f2/Bernoulli' in tr3.nodes
    assert 'f3/Normal' in tr3.nodes

def test_basic_loop():
    if False:
        for i in range(10):
            print('nop')

    @autoname
    def f1():
        if False:
            print('Hello World!')
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            return 10
        f1()
        for i in range(3):
            f1()
            sample('x', dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))
    tr = poutine.trace(f2).get_trace()
    assert 'model/f1/Bernoulli' in tr.nodes
    assert 'model/f1__1/Bernoulli' in tr.nodes
    assert 'model/f1__2/Bernoulli' in tr.nodes
    assert 'model/f1__3/Bernoulli' in tr.nodes
    assert 'model/x' in tr.nodes
    assert 'model/x1' in tr.nodes
    assert 'model/x2' in tr.nodes
    assert 'model/Normal' in tr.nodes

def test_named_loop():
    if False:
        return 10

    @autoname
    def f1():
        if False:
            for i in range(10):
                print('nop')
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            while True:
                i = 10
        f1()
        for i in autoname(range(3), name='loop'):
            f1()
            sample('x', dist.Bernoulli(0.5))
        return sample(dist.Normal(0.0, 1.0))
    tr = poutine.trace(f2).get_trace()
    assert 'model/f1/Bernoulli' in tr.nodes
    assert 'model/loop/f1/Bernoulli' in tr.nodes
    assert 'model/loop__1/f1/Bernoulli' in tr.nodes
    assert 'model/loop__2/f1/Bernoulli' in tr.nodes
    assert 'model/loop/x' in tr.nodes
    assert 'model/loop__1/x' in tr.nodes
    assert 'model/loop__2/x' in tr.nodes
    assert 'model/Normal' in tr.nodes

def test_sequential_plate():
    if False:
        print('Hello World!')

    @autoname
    def f1():
        if False:
            while True:
                i = 10
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            print('Hello World!')
        for i in autoname(pyro.plate(name='data', size=3)):
            f1()
        return sample(dist.Bernoulli(0.5))
    expected_names = ['model/data/f1/Bernoulli', 'model/data__1/f1/Bernoulli', 'model/data__2/f1/Bernoulli', 'model/Bernoulli']
    tr = poutine.trace(f2).get_trace()
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] == 'sample' and type(node['fn']).__name__ != '_Subsample']
    assert expected_names == actual_names

def test_nested_plate():
    if False:
        i = 10
        return i + 15

    @autoname
    def f1():
        if False:
            return 10
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def f2():
        if False:
            while True:
                i = 10
        for i in autoname(pyro.plate(name='data', size=3)):
            for j in autoname(range(2), name='xy'):
                f1()
        return sample(dist.Bernoulli(0.5))
    expected_names = ['model/data/xy/f1/Bernoulli', 'model/data/xy__1/f1/Bernoulli', 'model/data__1/xy/f1/Bernoulli', 'model/data__1/xy__1/f1/Bernoulli', 'model/data__2/xy/f1/Bernoulli', 'model/data__2/xy__1/f1/Bernoulli', 'model/Bernoulli']
    tr = poutine.trace(f2).get_trace()
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] == 'sample' and type(node['fn']).__name__ != '_Subsample']
    assert expected_names == actual_names

def test_model_guide():
    if False:
        for i in range(10):
            print('nop')

    @autoname
    def model():
        if False:
            while True:
                i = 10
        sample('x', dist.HalfNormal(1))
        return sample(dist.Bernoulli(0.5))

    @autoname(name='model')
    def guide():
        if False:
            while True:
                i = 10
        sample('x', dist.Gamma(1, 1))
        return sample(dist.Bernoulli(0.5))
    model_tr = poutine.trace(model).get_trace()
    guide_tr = poutine.trace(guide).get_trace()
    assert 'model/x' in model_tr.nodes
    assert 'model/x' in guide_tr.nodes
    assert 'model/Bernoulli' in model_tr.nodes
    assert 'model/Bernoulli' in guide_tr.nodes

def test_context_manager():
    if False:
        return 10

    @autoname
    def f1():
        if False:
            return 10
        return sample(dist.Bernoulli(0.5))

    def f2():
        if False:
            print('Hello World!')
        with autoname(name='prefix'):
            f1()
            f1()
    tr2 = poutine.trace(f2).get_trace()
    assert 'prefix/f1/Bernoulli' in tr2.nodes
    assert 'prefix/f1__1/Bernoulli' in tr2.nodes

def test_multi_nested():
    if False:
        return 10

    @autoname
    def model1(r=True):
        if False:
            for i in range(10):
                print('nop')
        model2()
        model2()
        with autoname(name='inter'):
            model2()
            if r:
                model1(r=False)
        model2()

    @autoname
    def model2():
        if False:
            i = 10
            return i + 15
        return sample('y', dist.Normal(0.0, 1.0))
    expected_names = ['model1/model2/y', 'model1/model2__1/y', 'model1/inter/model2/y', 'model1/inter/model1/model2/y', 'model1/inter/model1/model2__1/y', 'model1/inter/model1/inter/model2/y', 'model1/inter/model1/model2__2/y', 'model1/model2__2/y']
    tr = poutine.trace(model1).get_trace(r=True)
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] == 'sample' and type(node['fn']).__name__ != '_Subsample']
    assert expected_names == actual_names

def test_recur_multi():
    if False:
        return 10

    @autoname
    def model1(r=True):
        if False:
            return 10
        model2()
        with autoname(name='inter'):
            model2()
            if r:
                model1(r=False)
        model2()

    @autoname
    def model2():
        if False:
            i = 10
            return i + 15
        return sample('y', dist.Normal(0.0, 1.0))
    expected_names = ['model1/model2/y', 'model1/inter/model2/y', 'model1/inter/model1/model2/y', 'model1/inter/model1/inter/model2/y', 'model1/inter/model1/model2__1/y', 'model1/model2__1/y']
    tr = poutine.trace(model1).get_trace()
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] == 'sample' and type(node['fn']).__name__ != '_Subsample']
    assert expected_names == actual_names

def test_only_withs():
    if False:
        i = 10
        return i + 15

    def model1():
        if False:
            return 10
        with autoname(name='a'):
            with autoname(name='b'):
                sample('x', dist.Bernoulli(0.5))
    tr1 = poutine.trace(model1).get_trace()
    assert 'a/b/x' in tr1.nodes
    tr2 = poutine.trace(autoname(model1)).get_trace()
    assert 'model1/a/b/x' in tr2.nodes

def test_mutual_recur():
    if False:
        print('Hello World!')

    @autoname
    def model1(n):
        if False:
            for i in range(10):
                print('nop')
        sample('a', dist.Bernoulli(0.5))
        if n <= 0:
            return
        else:
            return model2(n - 1)

    @autoname
    def model2(n):
        if False:
            print('Hello World!')
        sample('b', dist.Bernoulli(0.5))
        if n <= 0:
            return
        else:
            model1(n)
    expected_names = ['model2/b', 'model2/model1/a', 'model2/model1/model2/b']
    tr = poutine.trace(model2).get_trace(1)
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] == 'sample' and type(node['fn']).__name__ != '_Subsample']
    assert expected_names == actual_names

def test_simple_recur():
    if False:
        while True:
            i = 10

    @autoname
    def geometric(p):
        if False:
            i = 10
            return i + 15
        x = sample('x', dist.Bernoulli(p))
        if x.item() == 1.0:
            return x + geometric(p)
        else:
            return x
    prev_name = 'x'
    for (name, node) in poutine.trace(geometric).get_trace(0.9).nodes.items():
        if node['type'] == 'sample':
            assert name == 'geometric/' + prev_name
            prev_name = 'geometric/' + prev_name

def test_no_param():
    if False:
        return 10
    pyro.clear_param_store()

    @autoname
    def model():
        if False:
            for i in range(10):
                print('nop')
        a = pyro.param('a', torch.tensor(0.5))
        return sample('b', dist.Bernoulli(a))
    expected_names = ['a', 'model/b']
    tr = poutine.trace(model).get_trace()
    actual_names = [name for (name, node) in tr.nodes.items() if node['type'] in ('param', 'sample')]
    assert expected_names == actual_names