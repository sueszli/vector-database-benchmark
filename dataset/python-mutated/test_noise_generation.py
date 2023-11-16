import numpy as np
from cleanlab.benchmarking import noise_generation
import pytest
seed = 0
np.random.seed(0)

def test_main_pipeline(verbose=False, n=10, valid_noise_matrix=True, frac_zero_noise_rates=0):
    if False:
        for i in range(10):
            print('nop')
    trace = 1.5
    py = [0.1, 0.1, 0.2, 0.6]
    K = len(py)
    y = [z for (i, p) in enumerate(py) for z in [i] * int(p * n)]
    nm = noise_generation.generate_noise_matrix_from_trace(K=K, trace=trace, py=py, seed=0, valid_noise_matrix=valid_noise_matrix, frac_zero_noise_rates=frac_zero_noise_rates)
    assert abs(trace - np.trace(nm) < 0.01)
    assert abs(nm.sum() - K) < 0.0001
    assert all(abs(nm.sum(axis=0) - 1) < 0.0001)
    assert abs(np.sum(nm * py) - 1 < 0.0001)
    s = noise_generation.generate_noisy_labels(y, nm)
    assert noise_generation.noise_matrix_is_valid(nm, py)

def test_main_pipeline_fraczero_high():
    if False:
        print('Hello World!')
    test_main_pipeline(n=1000, frac_zero_noise_rates=0.75)

def test_main_pipeline_verbose(verbose=True, n=10):
    if False:
        print('Hello World!')
    test_main_pipeline(verbose=verbose, n=n)

def test_main_pipeline_many(verbose=False, n=1000):
    if False:
        while True:
            i = 10
    test_main_pipeline(verbose=verbose, n=n)

def test_main_pipeline_many_verbose_valid(verbose=True, n=100):
    if False:
        i = 10
        return i + 15
    test_main_pipeline(verbose, n, valid_noise_matrix=True)

def test_main_pipeline_many_valid(verbose=False, n=100):
    if False:
        i = 10
        return i + 15
    test_main_pipeline(verbose, n, valid_noise_matrix=True)

def test_main_pipeline_many_verbose(verbose=True, n=1000):
    if False:
        i = 10
        return i + 15
    test_main_pipeline(verbose=verbose, n=n)

@pytest.mark.parametrize('verbose', [True, False])
def test_invalid_inputs_verify(verbose):
    if False:
        return 10
    nm = np.array([[0.2, 0.5], [0.8, 0.5]])
    py = [0.1, 0.8]
    assert not noise_generation.noise_matrix_is_valid(nm, py, verbose=verbose)
    nm = np.array([[0.2, 0.5], [0.8, 0.4]])
    py = [0.1, 0.9]
    assert not noise_generation.noise_matrix_is_valid(nm, py)
    py = [0.1, 0.8]
    assert not noise_generation.noise_matrix_is_valid(nm, py)

def test_invalid_matrix():
    if False:
        for i in range(10):
            print('nop')
    nm = np.array([[0.1, 0.9], [0.9, 0.1]])
    py = [0.1, 0.9]
    assert not noise_generation.noise_matrix_is_valid(nm, py)

def test_trace_less_than_1_error(trace=0.5):
    if False:
        while True:
            i = 10
    try:
        noise_generation.generate_noise_matrix_from_trace(3, trace)
    except ValueError as e:
        assert 'trace > 1' in str(e)
        with pytest.raises(ValueError) as e:
            noise_generation.generate_noise_matrix_from_trace(3, trace)

def test_trace_equals_1_error(trace=1):
    if False:
        for i in range(10):
            print('nop')
    test_trace_less_than_1_error(trace)

def test_valid_no_py_error():
    if False:
        print('Hello World!')
    try:
        noise_generation.generate_noise_matrix_from_trace(K=3, trace=2, valid_noise_matrix=True)
    except ValueError as e:
        assert 'py must be' in str(e)
        with pytest.raises(ValueError) as e:
            noise_generation.generate_noise_matrix_from_trace(K=3, trace=2, valid_noise_matrix=True)

def test_one_class_error():
    if False:
        i = 10
        return i + 15
    try:
        noise_generation.generate_noise_matrix_from_trace(K=1, trace=2)
    except ValueError as e:
        assert 'must be >= 2' in str(e)
        with pytest.raises(ValueError) as e:
            noise_generation.generate_noise_matrix_from_trace(K=1, trace=1)

def test_two_class_nofraczero():
    if False:
        while True:
            i = 10
    trace = 1.1
    nm = noise_generation.generate_noise_matrix_from_trace(K=2, trace=trace, valid_noise_matrix=True)
    assert not np.any(nm == 0)
    assert abs(trace - np.trace(nm) < 0.01)

def test_two_class_fraczero_high(valid=False):
    if False:
        i = 10
        return i + 15
    trace = 1.8
    frac_zero_noise_rates = 0.75
    nm = noise_generation.generate_noise_matrix_from_trace(K=2, trace=trace, valid_noise_matrix=valid, frac_zero_noise_rates=frac_zero_noise_rates)
    assert np.any(nm == 0)
    assert abs(trace - np.trace(nm) < 0.01)

def test_two_class_fraczero_high_valid():
    if False:
        while True:
            i = 10
    test_two_class_fraczero_high(True)

def test_gen_probs_sum_empty():
    if False:
        return 10
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    assert len(f(n=0, m=1)) == 0

def test_gen_probs_max_error():
    if False:
        while True:
            i = 10
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    try:
        f(n=5, m=1, max_prob=0.1)
    except ValueError as e:
        assert 'max_prob must be greater' in str(e)
        with pytest.raises(ValueError) as e:
            f(n=5, m=1, max_prob=0.1)

def test_gen_probs_min_error():
    if False:
        return 10
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    try:
        f(n=5, m=1, min_prob=0.9)
    except ValueError as e:
        assert 'min_prob must be less' in str(e)
        with pytest.raises(ValueError) as e:
            f(n=5, m=1, min_prob=0.9)

def test_probs_min_max_error():
    if False:
        return 10
    f = noise_generation.generate_n_rand_probabilities_that_sum_to_m
    min_prob = 0.5
    max_prob = 0.5
    try:
        f(n=2, m=1, min_prob=min_prob, max_prob=max_prob)
    except ValueError as e:
        assert 'min_prob must be less than max_prob' in str(e)
        with pytest.raises(ValueError) as e:
            f(n=5, m=1, min_prob=min_prob, max_prob=max_prob)

def test_balls_zero():
    if False:
        print('Hello World!')
    f = noise_generation.randomly_distribute_N_balls_into_K_bins
    K = 3
    result = f(N=0, K=K)
    assert len(result) == K
    assert sum(result) == 0

def test_balls_params():
    if False:
        i = 10
        return i + 15
    f = noise_generation.randomly_distribute_N_balls_into_K_bins
    N = 10
    K = 10
    for mx in [None, 1, 2, 3]:
        for mn in [None, 1, 2, 3]:
            r = f(N=N, K=K, max_balls_per_bin=mx, min_balls_per_bin=mn)
            assert sum(r) == K
            assert min(r) <= (K if mn is None else mn)
            assert len(r) == K

def test_max_iter():
    if False:
        for i in range(10):
            print('nop')
    trace = 2
    K = 3
    py = [1 / float(K)] * K
    nm = noise_generation.generate_noise_matrix_from_trace(K=K, trace=trace, valid_noise_matrix=True, max_iter=1, py=py, seed=1)
    assert abs(np.trace(nm) - trace) < 1e-06
    assert abs(sum(np.dot(nm, py)) - 1) < 1e-06
    nm2 = noise_generation.generate_noise_matrix_from_trace(K=3, trace=trace, valid_noise_matrix=True, py=[0.1, 0.1, 0.8], max_iter=0)
    assert nm2 is None