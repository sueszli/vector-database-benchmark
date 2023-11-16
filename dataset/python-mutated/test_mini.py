import ray
test_values = [1, 1.0, 'test', b'test', (0, 1), [0, 1], {0: 1}]

def test_basic_task_api(ray_start_regular):
    if False:
        i = 10
        return i + 15

    @ray.remote
    def f_simple():
        if False:
            print('Hello World!')
        return 1
    assert ray.get(f_simple.remote()) == 1

    @ray.remote(num_returns=3)
    def f_multiple_returns():
        if False:
            i = 10
            return i + 15
        return (1, 2, 3)
    (x_id1, x_id2, x_id3) = f_multiple_returns.remote()
    assert ray.get([x_id1, x_id2, x_id3]) == [1, 2, 3]

    @ray.remote
    def f_args_by_value(x):
        if False:
            for i in range(10):
                print('nop')
        return x
    for arg in test_values:
        assert ray.get(f_args_by_value.remote(arg)) == arg

def test_put_api(ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    for obj in test_values:
        assert ray.get(ray.put(obj)) == obj
    x_id = ray.put(0)
    for obj in [[x_id], (x_id,), {x_id: x_id}]:
        assert ray.get(ray.put(obj)) == obj

def test_actor_api(ray_start_regular):
    if False:
        print('Hello World!')

    @ray.remote
    class Foo:

        def __init__(self, val):
            if False:
                i = 10
                return i + 15
            self.x = val

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.x
    x = 1
    f = Foo.remote(x)
    assert ray.get(f.get.remote()) == x
if __name__ == '__main__':
    import pytest
    import os
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))