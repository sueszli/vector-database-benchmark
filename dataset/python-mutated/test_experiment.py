from sacred import Ingredient
'Global Docstring'
from mock import patch
import pytest
import sys
from sacred import cli_option
from sacred import host_info_gatherer
from sacred.experiment import Experiment
from sacred.utils import apply_backspaces_and_linefeeds, ConfigAddedError, SacredError

@pytest.fixture
def ex():
    if False:
        while True:
            i = 10
    return Experiment('ator3000')

def test_main(ex):
    if False:
        return 10

    @ex.main
    def foo():
        if False:
            print('Hello World!')
        pass
    assert 'foo' in ex.commands
    assert ex.commands['foo'] == foo
    assert ex.default_command == 'foo'

def test_automain_imported(ex):
    if False:
        for i in range(10):
            print('nop')
    main_called = [False]
    with patch.object(sys, 'argv', ['test.py']):

        @ex.automain
        def foo():
            if False:
                i = 10
                return i + 15
            main_called[0] = True
        assert 'foo' in ex.commands
        assert ex.commands['foo'] == foo
        assert ex.default_command == 'foo'
        assert main_called[0] is False

def test_automain_script_runs_main(ex):
    if False:
        print('Hello World!')
    global __name__
    oldname = __name__
    main_called = [False]
    try:
        __name__ = '__main__'
        with patch.object(sys, 'argv', ['test.py']):

            @ex.automain
            def foo():
                if False:
                    return 10
                main_called[0] = True
            assert 'foo' in ex.commands
            assert ex.commands['foo'] == foo
            assert ex.default_command == 'foo'
            assert main_called[0] is True
    finally:
        __name__ = oldname

def test_fails_on_unused_config_updates(ex):
    if False:
        for i in range(10):
            print('nop')

    @ex.config
    def cfg():
        if False:
            i = 10
            return i + 15
        a = 1
        c = 3

    @ex.main
    def foo(a, b=2):
        if False:
            i = 10
            return i + 15
        return a + b
    assert ex.run(config_updates={'a': 3}).result == 5
    assert ex.run(config_updates={'b': 8}).result == 9
    assert ex.run(config_updates={'c': 9}).result == 3
    with pytest.raises(ConfigAddedError):
        ex.run(config_updates={'d': 3})

def test_fails_on_nested_unused_config_updates(ex):
    if False:
        for i in range(10):
            print('nop')

    @ex.config
    def cfg():
        if False:
            while True:
                i = 10
        a = {'b': 1}
        d = {'e': 3}

    @ex.main
    def foo(a):
        if False:
            i = 10
            return i + 15
        return a['b']
    assert ex.run(config_updates={'a': {'b': 2}}).result == 2
    assert ex.run(config_updates={'a': {'c': 5}}).result == 1
    assert ex.run(config_updates={'d': {'e': 7}}).result == 1
    with pytest.raises(ConfigAddedError):
        ex.run(config_updates={'d': {'f': 3}})

def test_considers_captured_functions_for_fail_on_unused_config(ex):
    if False:
        print('Hello World!')

    @ex.config
    def cfg():
        if False:
            while True:
                i = 10
        a = 1

    @ex.capture
    def transmogrify(a, b=0):
        if False:
            while True:
                i = 10
        return a + b

    @ex.main
    def foo():
        if False:
            i = 10
            return i + 15
        return transmogrify()
    assert ex.run(config_updates={'a': 7}).result == 7
    assert ex.run(config_updates={'b': 3}).result == 4
    with pytest.raises(ConfigAddedError):
        ex.run(config_updates={'c': 3})

def test_considers_prefix_for_fail_on_unused_config(ex):
    if False:
        while True:
            i = 10

    @ex.config
    def cfg():
        if False:
            while True:
                i = 10
        a = {'b': 1}

    @ex.capture(prefix='a')
    def transmogrify(b):
        if False:
            for i in range(10):
                print('nop')
        return b

    @ex.main
    def foo():
        if False:
            return 10
        return transmogrify()
    assert ex.run(config_updates={'a': {'b': 3}}).result == 3
    with pytest.raises(ConfigAddedError):
        ex.run(config_updates={'b': 5})
    with pytest.raises(ConfigAddedError):
        ex.run(config_updates={'a': {'c': 5}})

def test_non_existing_prefix_is_treated_as_empty_dict(ex):
    if False:
        print('Hello World!')

    @ex.capture(prefix='nonexisting')
    def transmogrify(b=10):
        if False:
            i = 10
            return i + 15
        return b

    @ex.main
    def foo():
        if False:
            for i in range(10):
                print('nop')
        return transmogrify()
    assert ex.run().result == 10

def test_using_a_named_config(ex):
    if False:
        while True:
            i = 10

    @ex.config
    def cfg():
        if False:
            i = 10
            return i + 15
        a = 1

    @ex.named_config
    def ncfg_first():
        if False:
            for i in range(10):
                print('nop')
        a = 10

    @ex.named_config
    def ncfg_second(a):
        if False:
            while True:
                i = 10
        a = a * 2

    @ex.main
    def run(a):
        if False:
            return 10
        return a
    assert ex.run().result == 1
    assert ex.run(named_configs=['ncfg_first']).result == 10
    assert ex.run(named_configs=['ncfg_first', 'ncfg_second']).result == 20
    with pytest.raises(KeyError, match='.*not in preset for ConfigScope'):
        ex.run(named_configs=['ncfg_second', 'ncfg_first'])

def test_empty_dict_named_config(ex):
    if False:
        while True:
            i = 10

    @ex.named_config
    def ncfg():
        if False:
            for i in range(10):
                print('nop')
        empty_dict = {}
        nested_empty_dict = {'k1': {'k2': {}}}

    @ex.automain
    def main(empty_dict=1, nested_empty_dict=2):
        if False:
            print('Hello World!')
        return (empty_dict, nested_empty_dict)
    assert ex.run().result == (1, 2)
    assert ex.run(named_configs=['ncfg']).result == ({}, {'k1': {'k2': {}}})

def test_empty_dict_config_updates(ex):
    if False:
        for i in range(10):
            print('nop')

    @ex.config
    def cfg():
        if False:
            for i in range(10):
                print('nop')
        a = 1

    @ex.config
    def default():
        if False:
            print('Hello World!')
        a = {'b': 1}

    @ex.main
    def main():
        if False:
            i = 10
            return i + 15
        pass
    r = ex.run()
    assert r.config['a']['b'] == 1

def test_named_config_and_ingredient():
    if False:
        return 10
    ing = Ingredient('foo')

    @ing.config
    def cfg():
        if False:
            for i in range(10):
                print('nop')
        a = 10
    ex = Experiment(ingredients=[ing])

    @ex.config
    def default():
        if False:
            i = 10
            return i + 15
        b = 20

    @ex.named_config
    def named():
        if False:
            i = 10
            return i + 15
        b = 30

    @ex.main
    def main():
        if False:
            return 10
        pass
    r = ex.run(named_configs=['named'])
    assert r.config['b'] == 30
    assert r.config['foo'] == {'a': 10}

def test_captured_out_filter(ex, capsys):
    if False:
        return 10

    @ex.main
    def run_print_mock_progress():
        if False:
            i = 10
            return i + 15
        sys.stdout.write('progress 0')
        sys.stdout.flush()
        for i in range(10):
            sys.stdout.write('\x08')
            sys.stdout.write('{}'.format(i))
            sys.stdout.flush()
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    options = {'--loglevel': 'CRITICAL', '--capture': 'sys'}
    with capsys.disabled():
        assert ex.run(options=options).captured_out == 'progress 9'

def test_adding_option_hooks(ex):
    if False:
        return 10

    @ex.option_hook
    def hook(options):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ex.option_hook
    def hook2(options):
        if False:
            for i in range(10):
                print('nop')
        pass
    assert hook in ex.option_hooks
    assert hook2 in ex.option_hooks

def test_option_hooks_without_options_arg_raises(ex):
    if False:
        print('Hello World!')
    with pytest.raises(KeyError):

        @ex.option_hook
        def invalid_hook(wrong_arg_name):
            if False:
                while True:
                    i = 10
            pass

def test_config_hook_updates_config(ex):
    if False:
        while True:
            i = 10

    @ex.config
    def cfg():
        if False:
            i = 10
            return i + 15
        a = 'hello'

    @ex.config_hook
    def hook(config, command_name, logger):
        if False:
            i = 10
            return i + 15
        config.update({'a': 'me'})
        return config

    @ex.main
    def foo():
        if False:
            i = 10
            return i + 15
        pass
    r = ex.run()
    assert r.config['a'] == 'me'

def test_info_kwarg_updates_info(ex):
    if False:
        print('Hello World!')
    'Tests that the info kwarg of Experiment.create_run is used to update Run.info'

    @ex.automain
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    run = ex.run(info={'bar': 'baz'})
    assert 'bar' in run.info

def test_info_kwargs_default_behavior(ex):
    if False:
        return 10
    'Tests the default behavior of Experiment.create_run when the info kwarg is not specified.'

    @ex.automain
    def foo(_run):
        if False:
            print('Hello World!')
        _run.info['bar'] = 'baz'
    run = ex.run()
    assert 'bar' in run.info

def test_fails_on_config_write(ex):
    if False:
        print('Hello World!')

    @ex.config
    def cfg():
        if False:
            while True:
                i = 10
        a = 'hello'
        nested_dict = {'dict': {'dict': 1234, 'list': [1, 2, 3, 4]}}
        nested_list = [{'a': 42}, (1, 2, 3, 4), [1, 2, 3, 4]]
        nested_tuple = ({'a': 42}, (1, 2, 3, 4), [1, 2, 3, 4])

    @ex.main
    def main(_config, nested_dict, nested_list, nested_tuple):
        if False:
            for i in range(10):
                print('nop')
        raises_list = pytest.raises(SacredError, match='The configuration is read-only in a captured function!')
        raises_dict = pytest.raises(SacredError, match='The configuration is read-only in a captured function!')
        print('in main')
        with raises_dict:
            _config['a'] = 'world!'
        with raises_dict:
            nested_dict['dict'] = 'world!'
        with raises_dict:
            nested_dict['list'] = 'world!'
        with raises_dict:
            nested_dict.clear()
        with raises_dict:
            nested_dict.update({'a': 'world'})
        with raises_list:
            nested_dict['dict']['list'][0] = 1
        with raises_list:
            nested_list[0] = 'world!'
        with raises_list:
            nested_dict.clear()
        with raises_dict:
            nested_tuple[0]['a'] = 'world!'
        with raises_list:
            nested_tuple[2][0] = 123
    ex.run()

def test_add_config_dict_chain(ex):
    if False:
        return 10

    @ex.config
    def config1():
        if False:
            i = 10
            return i + 15
        'This is my demo configuration'
        dictnest_cap = {'key_1': 'value_1', 'key_2': 'value_2'}

    @ex.config
    def config2():
        if False:
            while True:
                i = 10
        'This is my demo configuration'
        dictnest_cap = {'key_2': 'update_value_2', 'key_3': 'value3', 'key_4': 'value4'}
    adict = {'dictnest_dict': {'key_1': 'value_1', 'key_2': 'value_2'}}
    ex.add_config(adict)
    bdict = {'dictnest_dict': {'key_2': 'update_value_2', 'key_3': 'value3', 'key_4': 'value4'}}
    ex.add_config(bdict)

    @ex.automain
    def run():
        if False:
            while True:
                i = 10
        pass
    final_config = ex.run().config
    assert final_config['dictnest_cap'] == {'key_1': 'value_1', 'key_2': 'update_value_2', 'key_3': 'value3', 'key_4': 'value4'}
    assert final_config['dictnest_cap'] == final_config['dictnest_dict']

def test_additional_gatherers():
    if False:
        while True:
            i = 10

    @host_info_gatherer('hello')
    def get_hello():
        if False:
            i = 10
            return i + 15
        return 'hello world'
    experiment = Experiment('ator3000', additional_host_info=[get_hello])

    @experiment.main
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    experiment.run()
    assert experiment.current_run.host_info['hello'] == 'hello world'

@pytest.mark.parametrize('command_line_option', ['-w', '--warning'])
def test_additional_cli_options_flag(command_line_option):
    if False:
        print('Hello World!')
    executed = [False]

    @cli_option('-w', '--warning', is_flag=True)
    def dummy_option(args, run):
        if False:
            i = 10
            return i + 15
        executed[0] = True
    experiment = Experiment('ator3000', additional_cli_options=[dummy_option])

    @experiment.main
    def foo():
        if False:
            while True:
                i = 10
        pass
    experiment.run_commandline([__file__, command_line_option])
    assert executed[0]

@pytest.mark.parametrize('command_line_option', ['-w', '--warning'])
def test_additional_cli_options(command_line_option):
    if False:
        while True:
            i = 10
    executed = [False]

    @cli_option('-w', '--warning')
    def dummy_option(args, run):
        if False:
            while True:
                i = 10
        executed[0] = args
    experiment = Experiment('ator3000', additional_cli_options=[dummy_option])

    @experiment.main
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    experiment.run_commandline([__file__, command_line_option, '10'])
    assert executed[0] == '10'