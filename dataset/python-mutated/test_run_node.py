import pytest
from kedro.framework.hooks.manager import _NullPluginManager
from kedro.pipeline import node
from kedro.runner import run_node

def generate_one():
    if False:
        for i in range(10):
            print('nop')
    yield from range(10)

def generate_tuple():
    if False:
        print('Hello World!')
    for i in range(10):
        yield (i, i * i)

def generate_list():
    if False:
        i = 10
        return i + 15
    for i in range(10):
        yield [i, i * i]

def generate_dict():
    if False:
        return 10
    for i in range(10):
        yield {'idx': i, 'square': i * i}

class TestRunGeneratorNode:

    def test_generator_fail_async(self, mocker, catalog):
        if False:
            i = 10
            return i + 15
        fake_dataset = mocker.Mock()
        catalog.add('result', fake_dataset)
        n = node(generate_one, inputs=None, outputs='result')
        with pytest.raises(Exception, match='nodes wrapping generator functions'):
            run_node(n, catalog, _NullPluginManager(), is_async=True)

    def test_generator_node_one(self, mocker, catalog):
        if False:
            while True:
                i = 10
        fake_dataset = mocker.Mock()
        catalog.add('result', fake_dataset)
        n = node(generate_one, inputs=None, outputs='result')
        run_node(n, catalog, _NullPluginManager())
        expected = [((i,),) for i in range(10)]
        assert 10 == fake_dataset.save.call_count
        assert fake_dataset.save.call_args_list == expected

    def test_generator_node_tuple(self, mocker, catalog):
        if False:
            for i in range(10):
                print('nop')
        left = mocker.Mock()
        right = mocker.Mock()
        catalog.add('left', left)
        catalog.add('right', right)
        n = node(generate_tuple, inputs=None, outputs=['left', 'right'])
        run_node(n, catalog, _NullPluginManager())
        expected_left = [((i,),) for i in range(10)]
        expected_right = [((i * i,),) for i in range(10)]
        assert 10 == left.save.call_count
        assert left.save.call_args_list == expected_left
        assert 10 == right.save.call_count
        assert right.save.call_args_list == expected_right

    def test_generator_node_list(self, mocker, catalog):
        if False:
            print('Hello World!')
        left = mocker.Mock()
        right = mocker.Mock()
        catalog.add('left', left)
        catalog.add('right', right)
        n = node(generate_list, inputs=None, outputs=['left', 'right'])
        run_node(n, catalog, _NullPluginManager())
        expected_left = [((i,),) for i in range(10)]
        expected_right = [((i * i,),) for i in range(10)]
        assert 10 == left.save.call_count
        assert left.save.call_args_list == expected_left
        assert 10 == right.save.call_count
        assert right.save.call_args_list == expected_right

    def test_generator_node_dict(self, mocker, catalog):
        if False:
            print('Hello World!')
        left = mocker.Mock()
        right = mocker.Mock()
        catalog.add('left', left)
        catalog.add('right', right)
        n = node(generate_dict, inputs=None, outputs={'idx': 'left', 'square': 'right'})
        run_node(n, catalog, _NullPluginManager())
        expected_left = [((i,),) for i in range(10)]
        expected_right = [((i * i,),) for i in range(10)]
        assert 10 == left.save.call_count
        assert left.save.call_args_list == expected_left
        assert 10 == right.save.call_count
        assert right.save.call_args_list == expected_right