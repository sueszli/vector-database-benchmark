"""
:codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import salt.modules.sysmod as sysmod
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import patch
from tests.support.unit import TestCase

class MockDocstringable:

    def __init__(self, docstr):
        if False:
            while True:
                i = 10
        self.__doc__ = docstr

    def set_module_docstring(self, docstr):
        if False:
            return 10
        self.__globals__ = {'__doc__': docstr}

class Mockstate:
    """
    Mock of State
    """

    class State:
        """
        Mock state functions
        """
        states = {}

        def __init__(self, opts):
            if False:
                while True:
                    i = 10
            pass

class Mockrunner:
    """
    Mock of runner
    """

    class Runner:
        """
        Mock runner functions
        """

        def __init__(self, opts):
            if False:
                i = 10
                return i + 15
            pass

        @property
        def functions(self):
            if False:
                for i in range(10):
                    print('nop')
            return sysmod.__salt__

class Mockloader:
    """
    Mock of loader
    """
    functions = []

    def __init__(self):
        if False:
            return 10
        pass

    def returners(self, opts, lst):
        if False:
            while True:
                i = 10
        '\n        Mock returner functions\n        '
        return sysmod.__salt__

    def render(self, opts, lst):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock renderers\n        '
        return sysmod.__salt__

class SysmodTestCase(TestCase, LoaderModuleMockMixin):
    """
    Test cases for salt.modules.sysmod
    """

    def setup_loader_modules(self):
        if False:
            i = 10
            return i + 15
        return {sysmod: {'__salt__': self.salt_dunder}}

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls._modules = set()
        cls._functions = ['exist.exist', 'sys.doc', 'sys.list_functions', 'sys.list_modules', 'sysctl.get', 'sysctl.show', 'system.halt', 'system.reboot', 'udev.name', 'udev.path', 'user.add', 'user.info', 'user.rename']
        cls._docstrings = {}
        cls._statedocstrings = {}
        cls.salt_dunder = {}
        for func in cls._functions:
            docstring = 'docstring for {}'.format(func)
            cls.salt_dunder[func] = MockDocstringable(docstring)
            cls._docstrings[func] = docstring
            module = func.split('.')[0]
            cls._statedocstrings[func] = docstring
            cls._statedocstrings[module] = 'docstring for {}'.format(module)
            cls._modules.add(func.split('.')[0])
            docstring = 'docstring for {}'.format(func)
            mock = MockDocstringable(docstring)
            mock.set_module_docstring('docstring for {}'.format(func.split('.')[0]))
            Mockstate.State.states[func] = mock
        cls._modules = sorted(list(cls._modules))
        cls.state_patcher = patch('salt.state', Mockstate())
        cls.state_patcher.start()
        cls.runner_patcher = patch('salt.runner', Mockrunner())
        cls.runner_patcher.start()
        cls.loader_patcher = patch('salt.loader', Mockloader())
        cls.loader_patcher.start()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.runner_patcher.stop()
        cls.state_patcher.stop()
        cls.loader_patcher.stop()
        for attrname in ('_modules', '_functions', '_docstrings', '_statedocstrings', 'salt_dunder', 'runner_patcher', 'state_patcher', 'loader_patcher'):
            try:
                delattr(cls, attrname)
            except AttributeError:
                continue

    def test_doc(self):
        if False:
            return 10
        '\n        Test if it returns the docstrings for all modules.\n        '
        self.assertDictEqual(sysmod.doc(), self._docstrings)
        self.assertDictEqual(sysmod.doc('sys.doc'), {'sys.doc': 'docstring for sys.doc'})

    def test_state_doc(self):
        if False:
            print('Hello World!')
        '\n        Test if it returns the docstrings for all states.\n        '
        self.assertDictEqual(sysmod.state_doc(), self._statedocstrings)
        self.assertDictEqual(sysmod.state_doc('sys.doc'), {'sys': 'docstring for sys', 'sys.doc': 'docstring for sys.doc'})

    def test_runner_doc(self):
        if False:
            return 10
        '\n        Test if it returns the docstrings for all runners.\n        '
        self.assertDictEqual(sysmod.runner_doc(), self._docstrings)
        self.assertDictEqual(sysmod.runner_doc('sys.doc'), {'sys.doc': 'docstring for sys.doc'})

    def test_returner_doc(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it returns the docstrings for all returners.\n        '
        self.assertDictEqual(sysmod.returner_doc(), self._docstrings)
        self.assertDictEqual(sysmod.returner_doc('sys.doc'), {'sys.doc': 'docstring for sys.doc'})

    def test_renderer_doc(self):
        if False:
            return 10
        '\n        Test if it returns the docstrings for all renderers.\n        '
        self.assertDictEqual(sysmod.renderer_doc(), self._docstrings)
        self.assertDictEqual(sysmod.renderer_doc('sys.doc'), {'sys.doc': 'docstring for sys.doc'})

    def test_list_functions(self):
        if False:
            while True:
                i = 10
        '\n        Test if it lists the functions for all modules.\n        '
        self.assertListEqual(sysmod.list_functions(), self._functions)
        self.assertListEqual(sysmod.list_functions('nonexist'), [])
        self.assertListEqual(sysmod.list_functions('sys'), ['sys.doc', 'sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_functions('sys*'), ['sys.doc', 'sys.list_functions', 'sys.list_modules', 'sysctl.get', 'sysctl.show', 'system.halt', 'system.reboot'])
        self.assertListEqual(sysmod.list_functions('sys.list*'), ['sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_functions('sys.list'), [])
        self.assertListEqual(sysmod.list_functions('exist.exist'), ['exist.exist'])

    def test_list_modules(self):
        if False:
            print('Hello World!')
        '\n        Test if it lists the modules loaded on the minion\n        '
        self.assertListEqual(sysmod.list_modules(), self._modules)
        self.assertListEqual(sysmod.list_modules('nonexist'), [])
        self.assertListEqual(sysmod.list_modules('user'), ['user'])
        self.assertListEqual(sysmod.list_modules('s*'), ['sys', 'sysctl', 'system'])

    def test_reload_modules(self):
        if False:
            print('Hello World!')
        '\n        Test if it tell the minion to reload the execution modules\n        '
        self.assertTrue(sysmod.reload_modules())

    def test_argspec(self):
        if False:
            return 10
        '\n        Test if it return the argument specification\n        of functions in Salt execution modules.\n        '
        self.assertDictEqual(sysmod.argspec(), {})

    def test_state_argspec(self):
        if False:
            return 10
        '\n        Test if it return the argument specification\n        of functions in Salt state modules.\n        '
        self.assertDictEqual(sysmod.state_argspec(), {})

    def test_returner_argspec(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if it return the argument specification\n        of functions in Salt returner modules.\n        '
        self.assertDictEqual(sysmod.returner_argspec(), {})

    def test_runner_argspec(self):
        if False:
            print('Hello World!')
        '\n        Test if it return the argument specification of functions in Salt runner\n        modules.\n        '
        self.assertDictEqual(sysmod.runner_argspec(), {})

    def test_list_state_functions(self):
        if False:
            print('Hello World!')
        '\n        Test if it lists the functions for all state modules.\n        '
        self.assertListEqual(sysmod.list_state_functions(), self._functions)
        self.assertListEqual(sysmod.list_state_functions('nonexist'), [])
        self.assertListEqual(sysmod.list_state_functions('sys'), ['sys.doc', 'sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_state_functions('sys*'), ['sys.doc', 'sys.list_functions', 'sys.list_modules', 'sysctl.get', 'sysctl.show', 'system.halt', 'system.reboot'])
        self.assertListEqual(sysmod.list_state_functions('sys.list*'), ['sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_state_functions('sys.list'), [])
        self.assertListEqual(sysmod.list_state_functions('exist.exist'), ['exist.exist'])

    def test_list_state_modules(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it lists the modules loaded on the minion.\n        '
        self.assertListEqual(sysmod.list_state_modules(), self._modules)
        self.assertListEqual(sysmod.list_state_modules('nonexist'), [])
        self.assertListEqual(sysmod.list_state_modules('user'), ['user'])
        self.assertListEqual(sysmod.list_state_modules('s*'), ['sys', 'sysctl', 'system'])

    def test_list_runners(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it list the runners loaded on the minion.\n        '
        self.assertListEqual(sysmod.list_runners(), self._modules)
        self.assertListEqual(sysmod.list_runners('nonexist'), [])
        self.assertListEqual(sysmod.list_runners('user'), ['user'])
        self.assertListEqual(sysmod.list_runners('s*'), ['sys', 'sysctl', 'system'])

    def test_list_runner_functions(self):
        if False:
            return 10
        '\n        Test if it lists the functions for all runner modules.\n        '
        self.assertListEqual(sysmod.list_runner_functions(), self._functions)
        self.assertListEqual(sysmod.list_runner_functions('nonexist'), [])
        self.assertListEqual(sysmod.list_runner_functions('sys'), ['sys.doc', 'sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_runner_functions('sys*'), ['sys.doc', 'sys.list_functions', 'sys.list_modules', 'sysctl.get', 'sysctl.show', 'system.halt', 'system.reboot'])
        self.assertListEqual(sysmod.list_runner_functions('sys.list*'), ['sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_runner_functions('sys.list'), [])
        self.assertListEqual(sysmod.list_runner_functions('exist.exist'), ['exist.exist'])

    def test_list_returners(self):
        if False:
            return 10
        '\n        Test if it lists the returners loaded on the minion\n        '
        self.assertListEqual(sysmod.list_returners(), self._modules)
        self.assertListEqual(sysmod.list_returners('nonexist'), [])
        self.assertListEqual(sysmod.list_returners('user'), ['user'])
        self.assertListEqual(sysmod.list_returners('s*'), ['sys', 'sysctl', 'system'])

    def test_list_returner_functions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it lists the functions for all returner modules.\n        '
        self.assertListEqual(sysmod.list_returner_functions(), self._functions)
        self.assertListEqual(sysmod.list_returner_functions('nonexist'), [])
        self.assertListEqual(sysmod.list_returner_functions('sys'), ['sys.doc', 'sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_returner_functions('sys*'), ['sys.doc', 'sys.list_functions', 'sys.list_modules', 'sysctl.get', 'sysctl.show', 'system.halt', 'system.reboot'])
        self.assertListEqual(sysmod.list_returner_functions('sys.list*'), ['sys.list_functions', 'sys.list_modules'])
        self.assertListEqual(sysmod.list_returner_functions('sys.list'), [])
        self.assertListEqual(sysmod.list_returner_functions('exist.exist'), ['exist.exist'])

    def test_list_renderers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it list the renderers loaded on the minion.\n        '
        self.assertListEqual(sysmod.list_renderers(), self._functions)
        self.assertListEqual(sysmod.list_renderers('nonexist'), [])
        self.assertListEqual(sysmod.list_renderers('user.info'), ['user.info'])
        self.assertListEqual(sysmod.list_renderers('syst*'), ['system.halt', 'system.reboot'])