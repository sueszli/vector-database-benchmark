import pytest
import torch
from pyroapi import pyro_backend
from torch.distributions import constraints
from pyro.ops.indexing import Vindex
try:
    import funsor
    from funsor.testing import assert_close
    import pyro.contrib.funsor
    from pyro.contrib.funsor.infer.traceenum_elbo import terms_from_trace
    funsor.set_backend('torch')
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro
except ImportError:
    pytestmark = pytest.mark.skip(reason='funsor is not installed')

def model_0(data, history, vectorized):
    if False:
        i = 10
        return i + 15
    x_dim = 3
    init = pyro.param('init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    trans = pyro.param('trans', lambda : torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param('locs', lambda : torch.rand(x_dim))
    with pyro.plate('sequences', data.shape[0], dim=-3) as sequences:
        sequences = sequences[:, None]
        x_prev = None
        markov_loop = pyro.vectorized_markov(name='time', size=data.shape[1], dim=-2, history=history) if vectorized else pyro.markov(range(data.shape[1]), history=history)
        for i in markov_loop:
            x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(init if isinstance(i, int) and i < 1 else trans[x_prev]))
            with pyro.plate('tones', data.shape[2], dim=-1):
                pyro.sample('y_{}'.format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.0), obs=Vindex(data)[sequences, i])
            x_prev = x_curr

def model_1(data, history, vectorized):
    if False:
        for i in range(10):
            print('nop')
    x_dim = 3
    init = pyro.param('init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    trans = pyro.param('trans', lambda : torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param('locs', lambda : torch.rand(x_dim))
    x_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(init if isinstance(i, int) and i < 1 else trans[x_prev]))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.0), obs=data[i])
        x_prev = x_curr

def model_2(data, history, vectorized):
    if False:
        i = 10
        return i + 15
    (x_dim, y_dim) = (3, 2)
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_init = pyro.param('y_init', lambda : torch.rand(x_dim, y_dim), constraint=constraints.simplex)
    y_trans = pyro.param('y_trans', lambda : torch.rand((x_dim, y_dim, y_dim)), constraint=constraints.simplex)
    x_prev = y_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_init if isinstance(i, int) and i < 1 else x_trans[x_prev]))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            y_curr = pyro.sample('y_{}'.format(i), dist.Categorical(y_init[x_curr] if isinstance(i, int) and i < 1 else Vindex(y_trans)[x_curr, y_prev]), obs=data[i])
        (x_prev, y_prev) = (x_curr, y_curr)

def model_3(data, history, vectorized):
    if False:
        print('Hello World!')
    (w_dim, x_dim, y_dim) = (2, 3, 2)
    w_init = pyro.param('w_init', lambda : torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param('w_trans', lambda : torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param('y_probs', lambda : torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)
    w_prev = x_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample('w_{}'.format(i), dist.Categorical(w_init if isinstance(i, int) and i < 1 else w_trans[w_prev]))
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_init if isinstance(i, int) and i < 1 else x_trans[x_prev]))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Categorical(Vindex(y_probs)[w_curr, x_curr]), obs=data[i])
        (x_prev, w_prev) = (x_curr, w_curr)

def model_4(data, history, vectorized):
    if False:
        i = 10
        return i + 15
    (w_dim, x_dim, y_dim) = (2, 3, 2)
    w_init = pyro.param('w_init', lambda : torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param('w_trans', lambda : torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param('x_init', lambda : torch.rand(w_dim, x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((w_dim, x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param('y_probs', lambda : torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)
    w_prev = x_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample('w_{}'.format(i), dist.Categorical(w_init if isinstance(i, int) and i < 1 else w_trans[w_prev]))
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_init[w_curr] if isinstance(i, int) and i < 1 else x_trans[w_curr, x_prev]))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Categorical(Vindex(y_probs)[w_curr, x_curr]), obs=data[i])
        (x_prev, w_prev) = (x_curr, w_curr)

def model_5(data, history, vectorized):
    if False:
        return 10
    (x_dim, y_dim) = (3, 2)
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_init_2 = pyro.param('x_init_2', lambda : torch.rand(x_dim, x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((x_dim, x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param('y_probs', lambda : torch.rand(x_dim, y_dim), constraint=constraints.simplex)
    x_prev = x_prev_2 = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        if isinstance(i, int) and i == 0:
            x_probs = x_init
        elif isinstance(i, int) and i == 1:
            x_probs = Vindex(x_init_2)[x_prev]
        else:
            x_probs = Vindex(x_trans)[x_prev_2, x_prev]
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_probs))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Categorical(Vindex(y_probs)[x_curr]), obs=data[i])
        (x_prev_2, x_prev) = (x_prev, x_curr)

def model_6(data, history, vectorized):
    if False:
        return 10
    x_dim = 3
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((len(data) - 1, x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param('locs', lambda : torch.rand(x_dim))
    x_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        if isinstance(i, int) and i < 1:
            x_probs = x_init
        elif isinstance(i, int):
            x_probs = x_trans[i - 1, x_prev]
        else:
            x_probs = Vindex(x_trans)[(i - 1)[:, None], x_prev]
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_probs))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.0), obs=data[i])
        x_prev = x_curr

def model_7(data, history, vectorized):
    if False:
        return 10
    (w_dim, x_dim, y_dim) = (2, 3, 2)
    w_init = pyro.param('w_init', lambda : torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param('w_trans', lambda : torch.rand((x_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((w_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param('y_probs', lambda : torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)
    w_prev = x_prev = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), dim=-2, history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample('w_{}'.format(i), dist.Categorical(w_init if isinstance(i, int) and i < 1 else w_trans[x_prev]))
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_init if isinstance(i, int) and i < 1 else x_trans[w_prev]))
        with pyro.plate('tones', data.shape[-1], dim=-1):
            pyro.sample('y_{}'.format(i), dist.Categorical(Vindex(y_probs)[w_curr, x_curr]), obs=data[i])
        (x_prev, w_prev) = (x_curr, w_curr)

def _guide_from_model(model):
    if False:
        for i in range(10):
            print('nop')
    try:
        with pyro_backend('contrib.funsor'):
            return handlers.block(infer.config_enumerate(model, default='parallel'), lambda msg: msg.get('is_observed', False))
    except KeyError:
        return model

@pytest.mark.parametrize('use_replay', [True, False])
@pytest.mark.parametrize('model,data,var,history', [(model_0, torch.rand(3, 5, 4), 'xy', 1), (model_1, torch.rand(5, 4), 'xy', 1), (model_2, torch.ones((5, 4), dtype=torch.long), 'xy', 1), (model_3, torch.ones((5, 4), dtype=torch.long), 'wxy', 1), (model_4, torch.ones((5, 4), dtype=torch.long), 'wxy', 1), (model_5, torch.ones((5, 4), dtype=torch.long), 'xy', 2), (model_6, torch.rand(5, 4), 'xy', 1), (model_6, torch.rand(100, 4), 'xy', 1), (model_7, torch.ones((5, 4), dtype=torch.long), 'wxy', 1), (model_7, torch.ones((50, 4), dtype=torch.long), 'wxy', 1)])
def test_enumeration(model, data, var, history, use_replay):
    if False:
        while True:
            i = 10
    pyro.clear_param_store()
    with pyro_backend('contrib.funsor'):
        with handlers.enum():
            enum_model = infer.config_enumerate(model, default='parallel')
            trace = handlers.trace(enum_model).get_trace(data, history, False)
            if use_replay:
                guide_trace = handlers.trace(_guide_from_model(model)).get_trace(data, history, True)
                vectorized_trace = handlers.trace(handlers.replay(model, trace=guide_trace)).get_trace(data, history, True)
            else:
                vectorized_trace = handlers.trace(enum_model).get_trace(data, history, True)
        factors = list()
        for i in range(data.shape[-2]):
            for v in var:
                factors.append(trace.nodes['{}_{}'.format(v, i)]['funsor']['log_prob'])
        vectorized_factors = list()
        for i in range(history):
            for v in var:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, i)]['funsor']['log_prob'])
        for i in range(history, data.shape[-2]):
            for v in var:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, slice(history, data.shape[-2]))]['funsor']['log_prob'](**{'time': i - history}, **{'{}_{}'.format(k, slice(history - j, data.shape[-2] - j)): '{}_{}'.format(k, i - j) for j in range(history + 1) for k in var}))
        for (f1, f2) in zip(factors, vectorized_factors):
            assert_close(f2, f1.align(tuple(f2.inputs)))
        actual_step = vectorized_trace.nodes['time']['value']
        expected_step = frozenset()
        expected_measure_vars = frozenset()
        for v in var[:-1]:
            v_step = tuple(('{}_{}'.format(v, i) for i in range(history))) + tuple(('{}_{}'.format(v, slice(j, data.shape[-2] - history + j)) for j in range(history + 1)))
            expected_step |= frozenset({v_step})
            if not use_replay:
                expected_measure_vars |= frozenset(v_step)
        assert actual_step == expected_step
        actual_measure_vars = terms_from_trace(vectorized_trace)['measure_vars']
        assert actual_measure_vars == expected_measure_vars

def model_8(weeks_data, days_data, history, vectorized):
    if False:
        while True:
            i = 10
    (x_dim, y_dim, w_dim, z_dim) = (3, 2, 2, 3)
    x_init = pyro.param('x_init', lambda : torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param('x_trans', lambda : torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param('y_probs', lambda : torch.rand(x_dim, y_dim), constraint=constraints.simplex)
    w_init = pyro.param('w_init', lambda : torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param('w_trans', lambda : torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    z_probs = pyro.param('z_probs', lambda : torch.rand(w_dim, z_dim), constraint=constraints.simplex)
    x_prev = None
    weeks_loop = pyro.vectorized_markov(name='weeks', size=len(weeks_data), dim=-1, history=history) if vectorized else pyro.markov(range(len(weeks_data)), history=history)
    for i in weeks_loop:
        if isinstance(i, int) and i == 0:
            x_probs = x_init
        else:
            x_probs = Vindex(x_trans)[x_prev]
        x_curr = pyro.sample('x_{}'.format(i), dist.Categorical(x_probs))
        pyro.sample('y_{}'.format(i), dist.Categorical(Vindex(y_probs)[x_curr]), obs=weeks_data[i])
        x_prev = x_curr
    w_prev = None
    days_loop = pyro.vectorized_markov(name='days', size=len(days_data), dim=-1, history=history) if vectorized else pyro.markov(range(len(days_data)), history=history)
    for j in days_loop:
        if isinstance(j, int) and j == 0:
            w_probs = w_init
        else:
            w_probs = Vindex(w_trans)[w_prev]
        w_curr = pyro.sample('w_{}'.format(j), dist.Categorical(w_probs))
        pyro.sample('z_{}'.format(j), dist.Categorical(Vindex(z_probs)[w_curr]), obs=days_data[j])
        w_prev = w_curr

@pytest.mark.parametrize('use_replay', [True, False])
@pytest.mark.parametrize('model,weeks_data,days_data,vars1,vars2,history', [(model_8, torch.ones(3), torch.zeros(9), 'xy', 'wz', 1), (model_8, torch.ones(30), torch.zeros(50), 'xy', 'wz', 1)])
def test_enumeration_multi(model, weeks_data, days_data, vars1, vars2, history, use_replay):
    if False:
        i = 10
        return i + 15
    pyro.clear_param_store()
    with pyro_backend('contrib.funsor'):
        with handlers.enum():
            enum_model = infer.config_enumerate(model, default='parallel')
            trace = handlers.trace(enum_model).get_trace(weeks_data, days_data, history, False)
            if use_replay:
                guide_trace = handlers.trace(_guide_from_model(model)).get_trace(weeks_data, days_data, history, True)
                vectorized_trace = handlers.trace(handlers.replay(model, trace=guide_trace)).get_trace(weeks_data, days_data, history, True)
            else:
                vectorized_trace = handlers.trace(enum_model).get_trace(weeks_data, days_data, history, True)
        factors = list()
        for i in range(len(weeks_data)):
            for v in vars1:
                factors.append(trace.nodes['{}_{}'.format(v, i)]['funsor']['log_prob'])
        for j in range(len(days_data)):
            for v in vars2:
                factors.append(trace.nodes['{}_{}'.format(v, j)]['funsor']['log_prob'])
        vectorized_factors = list()
        for i in range(history):
            for v in vars1:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, i)]['funsor']['log_prob'])
        for i in range(history, len(weeks_data)):
            for v in vars1:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, slice(history, len(weeks_data)))]['funsor']['log_prob'](**{'weeks': i - history}, **{'{}_{}'.format(k, slice(history - j, len(weeks_data) - j)): '{}_{}'.format(k, i - j) for j in range(history + 1) for k in vars1}))
        for i in range(history):
            for v in vars2:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, i)]['funsor']['log_prob'])
        for i in range(history, len(days_data)):
            for v in vars2:
                vectorized_factors.append(vectorized_trace.nodes['{}_{}'.format(v, slice(history, len(days_data)))]['funsor']['log_prob'](**{'days': i - history}, **{'{}_{}'.format(k, slice(history - j, len(days_data) - j)): '{}_{}'.format(k, i - j) for j in range(history + 1) for k in vars2}))
        for (f1, f2) in zip(factors, vectorized_factors):
            assert_close(f2, f1.align(tuple(f2.inputs)))
        expected_measure_vars = frozenset()
        actual_weeks_step = vectorized_trace.nodes['weeks']['value']
        expected_weeks_step = frozenset()
        for v in vars1[:-1]:
            v_step = tuple(('{}_{}'.format(v, i) for i in range(history))) + tuple(('{}_{}'.format(v, slice(j, len(weeks_data) - history + j)) for j in range(history + 1)))
            expected_weeks_step |= frozenset({v_step})
            if not use_replay:
                expected_measure_vars |= frozenset(v_step)
        actual_days_step = vectorized_trace.nodes['days']['value']
        expected_days_step = frozenset()
        for v in vars2[:-1]:
            v_step = tuple(('{}_{}'.format(v, i) for i in range(history))) + tuple(('{}_{}'.format(v, slice(j, len(days_data) - history + j)) for j in range(history + 1)))
            expected_days_step |= frozenset({v_step})
            if not use_replay:
                expected_measure_vars |= frozenset(v_step)
        assert actual_weeks_step == expected_weeks_step
        assert actual_days_step == expected_days_step
        actual_measure_vars = terms_from_trace(vectorized_trace)['measure_vars']
        assert actual_measure_vars == expected_measure_vars

def guide_empty(data, history, vectorized):
    if False:
        return 10
    pass

@pytest.mark.xfail(reason='funsor version drift')
@pytest.mark.parametrize('model,guide,data,history', [(model_0, guide_empty, torch.rand(3, 5, 4), 1), (model_1, guide_empty, torch.rand(5, 4), 1), (model_2, guide_empty, torch.ones((5, 4), dtype=torch.long), 1), (model_3, guide_empty, torch.ones((5, 4), dtype=torch.long), 1), (model_4, guide_empty, torch.ones((5, 4), dtype=torch.long), 1), (model_5, guide_empty, torch.ones((5, 4), dtype=torch.long), 2), (model_6, guide_empty, torch.rand(5, 4), 1), (model_6, guide_empty, torch.rand(100, 4), 1), (model_7, guide_empty, torch.ones((5, 4), dtype=torch.long), 1), (model_7, guide_empty, torch.ones((50, 4), dtype=torch.long), 1)])
def test_model_enumerated_elbo(model, guide, data, history):
    if False:
        print('Hello World!')
    pyro.clear_param_store()
    with pyro_backend('contrib.funsor'):
        model = infer.config_enumerate(model, default='parallel')
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=4)
        expected_loss = elbo.loss_and_grads(model, guide, data, history, False)
        expected_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        vectorized_elbo = infer.TraceMarkovEnum_ELBO(max_plate_nesting=4)
        actual_loss = vectorized_elbo.loss_and_grads(model, guide, data, history, True)
        actual_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        assert_close(actual_loss, expected_loss)
        for (actual_grad, expected_grad) in zip(actual_grads, expected_grads):
            assert_close(actual_grad, expected_grad)

def guide_empty_multi(weeks_data, days_data, history, vectorized):
    if False:
        for i in range(10):
            print('nop')
    pass

@pytest.mark.xfail(reason='funsor version drift')
@pytest.mark.parametrize('model,guide,weeks_data,days_data,history', [(model_8, guide_empty_multi, torch.ones(3), torch.zeros(9), 1), (model_8, guide_empty_multi, torch.ones(30), torch.zeros(50), 1)])
def test_model_enumerated_elbo_multi(model, guide, weeks_data, days_data, history):
    if False:
        for i in range(10):
            print('nop')
    pyro.clear_param_store()
    with pyro_backend('contrib.funsor'):
        model = infer.config_enumerate(model, default='parallel')
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=4)
        expected_loss = elbo.loss_and_grads(model, guide, weeks_data, days_data, history, False)
        expected_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        vectorized_elbo = infer.TraceMarkovEnum_ELBO(max_plate_nesting=4)
        actual_loss = vectorized_elbo.loss_and_grads(model, guide, weeks_data, days_data, history, True)
        actual_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        assert_close(actual_loss, expected_loss)
        for (actual_grad, expected_grad) in zip(actual_grads, expected_grads):
            assert_close(actual_grad, expected_grad)

def model_10(data, history, vectorized):
    if False:
        i = 10
        return i + 15
    init_probs = torch.tensor([0.5, 0.5])
    transition_probs = pyro.param('transition_probs', torch.tensor([[0.75, 0.25], [0.25, 0.75]]), constraint=constraints.simplex)
    emission_probs = pyro.param('emission_probs', torch.tensor([[0.75, 0.25], [0.25, 0.75]]), constraint=constraints.simplex)
    x = None
    markov_loop = pyro.vectorized_markov(name='time', size=len(data), history=history) if vectorized else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        probs = init_probs if x is None else transition_probs[x]
        x = pyro.sample('x_{}'.format(i), dist.Categorical(probs))
        pyro.sample('y_{}'.format(i), dist.Categorical(emission_probs[x]), obs=data[i])

@pytest.mark.parametrize('model,guide,data,history', [(model_0, _guide_from_model(model_0), torch.rand(3, 5, 4), 1), (model_1, _guide_from_model(model_1), torch.rand(5, 4), 1), (model_2, _guide_from_model(model_2), torch.ones((5, 4), dtype=torch.long), 1), (model_3, _guide_from_model(model_3), torch.ones((5, 4), dtype=torch.long), 1), (model_4, _guide_from_model(model_4), torch.ones((5, 4), dtype=torch.long), 1), (model_5, _guide_from_model(model_5), torch.ones((5, 4), dtype=torch.long), 2), (model_6, _guide_from_model(model_6), torch.rand(5, 4), 1), (model_7, _guide_from_model(model_7), torch.ones((5, 4), dtype=torch.long), 1), (model_10, _guide_from_model(model_10), torch.ones(5), 1)])
def test_guide_enumerated_elbo(model, guide, data, history):
    if False:
        i = 10
        return i + 15
    pyro.clear_param_store()
    with pyro_backend('contrib.funsor'), pytest.raises(NotImplementedError, match='TraceMarkovEnum_ELBO does not yet support guide side Markov enumeration'):
        if history > 1:
            pytest.xfail(reason='TraceMarkovEnum_ELBO does not yet support history > 1')
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=4)
        expected_loss = elbo.loss_and_grads(model, guide, data, history, False)
        expected_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        vectorized_elbo = infer.TraceMarkovEnum_ELBO(max_plate_nesting=4)
        actual_loss = vectorized_elbo.loss_and_grads(model, guide, data, history, True)
        actual_grads = (value.grad for (name, value) in pyro.get_param_store().named_parameters())
        assert_close(actual_loss, expected_loss)
        for (actual_grad, expected_grad) in zip(actual_grads, expected_grads):
            assert_close(actual_grad, expected_grad)