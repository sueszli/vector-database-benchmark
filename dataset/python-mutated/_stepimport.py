"""
This module provides low-level helper functionality during step imports.

.. warn:: Do not use directly

    It should not be used directly except in behave Runner classes
    that need to provide the correct context (step_registry, matchers, etc.)
    instead of using the global module specific variables.
"""
from __future__ import absolute_import
from contextlib import contextmanager
from threading import Lock
from types import ModuleType
import os.path
import sys
import six
from behave import step_registry as _step_registry
from behave.matchers import StepMatcherFactory
from behave.step_registry import StepRegistry

def setup_api_with_step_decorators(module, step_registry):
    if False:
        for i in range(10):
            print('nop')
    _step_registry.setup_step_decorators(module, step_registry)

def setup_api_with_matcher_functions(module, step_matcher_factory):
    if False:
        while True:
            i = 10
    module.use_default_step_matcher = step_matcher_factory.use_default_step_matcher
    module.use_step_matcher = step_matcher_factory.use_step_matcher
    module.step_matcher = step_matcher_factory.use_step_matcher
    module.register_type = step_matcher_factory.register_type

class SimpleStepContainer(object):

    def __init__(self, step_registry=None):
        if False:
            print('Hello World!')
        if step_registry is None:
            step_registry = StepRegistry()
        self.step_matcher_factory = StepMatcherFactory()
        self.step_registry = step_registry
        self.step_registry.step_matcher_factory = self.step_matcher_factory

class FakeModule(ModuleType):
    ensure_fake = True

    def __setitem__(self, name, value):
        if False:
            return 10
        assert '.' not in name
        setattr(self, name, value)

class StepRegistryModule(FakeModule):
    """Provides a fake :mod:`behave.step_registry` module
    that can be used during step imports.
    """
    __all__ = ['given', 'when', 'then', 'step', 'Given', 'When', 'Then', 'Step']

    def __init__(self, step_registry):
        if False:
            i = 10
            return i + 15
        super(StepRegistryModule, self).__init__('behave.step_registry')
        self.registry = step_registry
        setup_api_with_step_decorators(self, step_registry)

class StepMatchersModule(FakeModule):
    __all__ = ['use_default_step_matcher', 'use_step_matcher', 'step_matcher', 'register_type']

    def __init__(self, step_matcher_factory):
        if False:
            print('Hello World!')
        super(StepMatchersModule, self).__init__('behave.matchers')
        self.step_matcher_factory = step_matcher_factory
        setup_api_with_matcher_functions(self, step_matcher_factory)
        self.make_matcher = step_matcher_factory.make_matcher
        here = os.path.dirname(__file__)
        self.__file__ = os.path.abspath(os.path.join(here, 'matchers.py'))
        self.__name__ = 'behave.matchers'

class BehaveModule(FakeModule):
    __all__ = StepRegistryModule.__all__ + StepMatchersModule.__all__

    def __init__(self, step_registry, step_matcher_factory=None):
        if False:
            i = 10
            return i + 15
        if step_matcher_factory is None:
            step_matcher_factory = step_registry.step_step_matcher_factory
        assert step_matcher_factory is not None
        super(BehaveModule, self).__init__('behave')
        setup_api_with_step_decorators(self, step_registry)
        setup_api_with_matcher_functions(self, step_matcher_factory)
        self.use_default_step_matcher = step_matcher_factory.use_default_step_matcher
        assert step_registry.step_matcher_factory == step_matcher_factory
        here = os.path.dirname(__file__)
        self.__file__ = os.path.abspath(os.path.join(here, '__init__.py'))
        self.__name__ = 'behave'
        self.__path__ = [os.path.abspath(here)]
        self.__package__ = None

class StepImportModuleContext(object):

    def __init__(self, step_container):
        if False:
            i = 10
            return i + 15
        self.step_registry = step_container.step_registry
        self.step_matcher_factory = step_container.step_matcher_factory
        assert self.step_registry.step_matcher_factory == self.step_matcher_factory
        self.step_registry.step_matcher_factory = self.step_matcher_factory
        step_registry_module = StepRegistryModule(self.step_registry)
        step_matchers_module = StepMatchersModule(self.step_matcher_factory)
        behave_module = BehaveModule(self.step_registry, self.step_matcher_factory)
        self.modules = {'behave': behave_module, 'behave.matchers': step_matchers_module, 'behave.step_registry': step_registry_module}

    def reset_current_matcher(self):
        if False:
            return 10
        self.step_matcher_factory.use_default_step_matcher()
_step_import_lock = Lock()
unknown = object()

@contextmanager
def use_step_import_modules(step_container):
    if False:
        while True:
            i = 10
    "\n    Redirect any step/type registration to the runner's step-context object\n    during step imports by using fake modules (instead of using module-globals).\n\n    This allows that multiple runners can be used without polluting the\n    global variables in problematic modules\n    (:mod:`behave.step_registry`, mod:`behave.matchers`).\n\n    .. sourcecode:: python\n\n        # -- RUNNER-IMPLEMENTATION:\n        def load_step_definitions(self, ...):\n            step_container = self.step_container\n            with use_step_import_modules(step_container) as import_context:\n                # -- USE: Fake modules during step imports\n                ...\n                import_context.reset_current_matcher()\n\n    :param step_container:\n        Step context object with step_registry, step_matcher_factory.\n    "
    orig_modules = {}
    import_context = StepImportModuleContext(step_container)
    with _step_import_lock:
        try:
            for (module_name, fake_module) in six.iteritems(import_context.modules):
                orig_module = sys.modules.get(module_name, unknown)
                orig_modules[module_name] = orig_module
                sys.modules[module_name] = fake_module
            yield import_context
        finally:
            for (module_name, orig_module) in six.iteritems(orig_modules):
                if orig_module is unknown:
                    del sys.modules[module_name]
                else:
                    sys.modules[module_name] = orig_module