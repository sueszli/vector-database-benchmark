from __future__ import annotations
import pytest
from ansible.errors import AnsibleParserError
from ansible.parsing.mod_args import ModuleArgsParser
from ansible.plugins.loader import init_plugin_loader
from ansible.utils.sentinel import Sentinel

class TestModArgsDwim:
    INVALID_MULTIPLE_ACTIONS = (({'action': 'shell echo hi', 'local_action': 'shell echo hi'}, 'action and local_action are mutually exclusive'), ({'action': 'shell echo hi', 'shell': 'echo hi'}, 'conflicting action statements: shell, shell'), ({'local_action': 'shell echo hi', 'shell': 'echo hi'}, 'conflicting action statements: shell, shell'))

    def _debug(self, mod, args, to):
        if False:
            for i in range(10):
                print('nop')
        print('RETURNED module = {0}'.format(mod))
        print('           args = {0}'.format(args))
        print('             to = {0}'.format(to))

    def test_basic_shell(self):
        if False:
            return 10
        m = ModuleArgsParser(dict(shell='echo hi'))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod == 'shell'
        assert args == dict(_raw_params='echo hi')
        assert to is Sentinel

    def test_basic_command(self):
        if False:
            i = 10
            return i + 15
        m = ModuleArgsParser(dict(command='echo hi'))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod == 'command'
        assert args == dict(_raw_params='echo hi')
        assert to is Sentinel

    def test_shell_with_modifiers(self):
        if False:
            for i in range(10):
                print('nop')
        m = ModuleArgsParser(dict(shell='/bin/foo creates=/tmp/baz removes=/tmp/bleep'))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod == 'shell'
        assert args == dict(creates='/tmp/baz', removes='/tmp/bleep', _raw_params='/bin/foo')
        assert to is Sentinel

    def test_normal_usage(self):
        if False:
            i = 10
            return i + 15
        m = ModuleArgsParser(dict(copy='src=a dest=b'))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod, 'copy'
        assert args, dict(src='a', dest='b')
        assert to is Sentinel

    def test_complex_args(self):
        if False:
            print('Hello World!')
        m = ModuleArgsParser(dict(copy=dict(src='a', dest='b')))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod, 'copy'
        assert args, dict(src='a', dest='b')
        assert to is Sentinel

    def test_action_with_complex(self):
        if False:
            return 10
        m = ModuleArgsParser(dict(action=dict(module='copy', src='a', dest='b')))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod == 'copy'
        assert args == dict(src='a', dest='b')
        assert to is Sentinel

    def test_action_with_complex_and_complex_args(self):
        if False:
            i = 10
            return i + 15
        m = ModuleArgsParser(dict(action=dict(module='copy', args=dict(src='a', dest='b'))))
        (mod, args, to) = m.parse()
        self._debug(mod, args, to)
        assert mod == 'copy'
        assert args == dict(src='a', dest='b')
        assert to is Sentinel

    def test_local_action_string(self):
        if False:
            for i in range(10):
                print('nop')
        m = ModuleArgsParser(dict(local_action='copy src=a dest=b'))
        (mod, args, delegate_to) = m.parse()
        self._debug(mod, args, delegate_to)
        assert mod == 'copy'
        assert args == dict(src='a', dest='b')
        assert delegate_to == 'localhost'

    @pytest.mark.parametrize('args_dict, msg', INVALID_MULTIPLE_ACTIONS)
    def test_multiple_actions(self, args_dict, msg):
        if False:
            return 10
        m = ModuleArgsParser(args_dict)
        with pytest.raises(AnsibleParserError) as err:
            m.parse()
        assert err.value.args[0] == msg

    def test_multiple_actions_ping_shell(self):
        if False:
            while True:
                i = 10
        init_plugin_loader()
        args_dict = {'ping': 'data=hi', 'shell': 'echo hi'}
        m = ModuleArgsParser(args_dict)
        with pytest.raises(AnsibleParserError) as err:
            m.parse()
        assert err.value.args[0] == f"conflicting action statements: {', '.join(args_dict)}"

    def test_bogus_action(self):
        if False:
            i = 10
            return i + 15
        init_plugin_loader()
        args_dict = {'bogusaction': {}}
        m = ModuleArgsParser(args_dict)
        with pytest.raises(AnsibleParserError) as err:
            m.parse()
        assert err.value.args[0].startswith(f"couldn't resolve module/action '{next(iter(args_dict))}'")