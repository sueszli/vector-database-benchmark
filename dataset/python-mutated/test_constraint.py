import numpy as np
from bayes_opt import BayesianOptimization
from pytest import approx, raises
from scipy.optimize import NonlinearConstraint
np.random.seed(42)

def test_single_constraint_upper():
    if False:
        return 10

    def target_function(x, y):
        if False:
            i = 10
            return i + 15
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        if False:
            return 10
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)
    constraint_limit_upper = 0.5
    constraint = NonlinearConstraint(constraint_function, -np.inf, constraint_limit_upper)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    optimizer = BayesianOptimization(f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1)
    optimizer.maximize(init_points=2, n_iter=10)
    assert constraint_function(**optimizer.max['params']) <= constraint_limit_upper

def test_single_constraint_lower():
    if False:
        i = 10
        return i + 15

    def target_function(x, y):
        if False:
            while True:
                i = 10
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        if False:
            return 10
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)
    constraint_limit_lower = -0.5
    constraint = NonlinearConstraint(constraint_function, constraint_limit_lower, np.inf)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    optimizer = BayesianOptimization(f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1)
    optimizer.maximize(init_points=2, n_iter=10)
    assert constraint_function(**optimizer.max['params']) >= constraint_limit_lower

def test_single_constraint_lower_upper():
    if False:
        print('Hello World!')

    def target_function(x, y):
        if False:
            for i in range(10):
                print('nop')
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(x, y):
        if False:
            while True:
                i = 10
        return np.cos(x) * np.cos(y) - np.sin(x) * np.sin(y)
    constraint_limit_lower = -0.5
    constraint_limit_upper = 0.5
    constraint = NonlinearConstraint(constraint_function, constraint_limit_lower, constraint_limit_upper)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    optimizer = BayesianOptimization(f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1)
    assert optimizer.constraint.lb == constraint.lb
    assert optimizer.constraint.ub == constraint.ub
    optimizer.maximize(init_points=2, n_iter=10)
    assert constraint_function(**optimizer.max['params']) <= constraint_limit_upper
    assert constraint_function(**optimizer.max['params']) >= constraint_limit_lower
    res = np.array([[r['target'], r['constraint'], r['params']['x'], r['params']['y']] for r in optimizer.res[:-1]])
    xy = res[:, [2, 3]]
    x = res[:, 2]
    y = res[:, 3]
    assert constraint_function(x, y) == approx(optimizer.constraint.approx(xy), rel=1e-05, abs=1e-05)
    assert constraint_function(x, y) == approx(optimizer.space.constraint_values[:-1], rel=1e-05, abs=1e-05)

def test_multiple_constraints():
    if False:
        while True:
            i = 10

    def target_function(x, y):
        if False:
            i = 10
            return i + 15
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        if False:
            while True:
                i = 10
        return np.array([-np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)])
    constraint_limit_lower = np.array([-np.inf, -np.inf])
    constraint_limit_upper = np.array([0.6, 0.6])
    conmod = NonlinearConstraint(constraint_function_2_dim, constraint_limit_lower, constraint_limit_upper)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    optimizer = BayesianOptimization(f=target_function, constraint=conmod, pbounds=pbounds, verbose=0, random_state=1)
    optimizer.maximize(init_points=2, n_iter=10)
    constraint_at_max = constraint_function_2_dim(**optimizer.max['params'])
    assert np.all((constraint_at_max <= constraint_limit_upper) & (constraint_at_max >= constraint_limit_lower))
    params = optimizer.res[0]['params']
    (x, y) = (params['x'], params['y'])
    assert constraint_function_2_dim(x, y) == approx(optimizer.constraint.approx(np.array([x, y])), rel=0.001, abs=0.001)

def test_kwargs_not_the_same():
    if False:
        print('Hello World!')

    def target_function(x, y):
        if False:
            print('Hello World!')
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function(a, b):
        if False:
            i = 10
            return i + 15
        return np.cos(a) * np.cos(b) - np.sin(a) * np.sin(b)
    constraint_limit_upper = 0.5
    constraint = NonlinearConstraint(constraint_function, -np.inf, constraint_limit_upper)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    optimizer = BayesianOptimization(f=target_function, constraint=constraint, pbounds=pbounds, verbose=0, random_state=1)
    with raises(TypeError, match='Encountered TypeError when evaluating'):
        optimizer.maximize(init_points=2, n_iter=10)

def test_lower_less_than_upper():
    if False:
        print('Hello World!')

    def target_function(x, y):
        if False:
            while True:
                i = 10
        return np.cos(2 * x) * np.cos(y) + np.sin(x)

    def constraint_function_2_dim(x, y):
        if False:
            return 10
        return np.array([-np.cos(x) * np.cos(y) + np.sin(x) * np.sin(y), -np.cos(x) * np.cos(-y) + np.sin(x) * np.sin(-y)])
    constraint_limit_lower = np.array([0.6, -np.inf])
    constraint_limit_upper = np.array([0.3, 0.6])
    conmod = NonlinearConstraint(constraint_function_2_dim, constraint_limit_lower, constraint_limit_upper)
    pbounds = {'x': (0, 6), 'y': (0, 6)}
    with raises(ValueError):
        optimizer = BayesianOptimization(f=target_function, constraint=conmod, pbounds=pbounds, verbose=0, random_state=1)