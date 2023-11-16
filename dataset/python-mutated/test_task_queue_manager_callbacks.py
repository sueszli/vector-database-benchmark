from __future__ import annotations
import unittest
from unittest.mock import MagicMock
from ansible.executor.task_queue_manager import TaskQueueManager
from ansible.playbook import Playbook
from ansible.plugins.callback import CallbackBase
from ansible.utils import context_objects as co

class TestTaskQueueManagerCallbacks(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        inventory = MagicMock()
        variable_manager = MagicMock()
        loader = MagicMock()
        passwords = []
        co.GlobalCLIArgs._Singleton__instance = None
        self._tqm = TaskQueueManager(inventory, variable_manager, loader, passwords)
        self._playbook = Playbook(loader)
        self._register = MagicMock()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        co.GlobalCLIArgs._Singleton__instance = None

    def test_task_queue_manager_callbacks_v2_playbook_on_start(self):
        if False:
            while True:
                i = 10
        '\n        Assert that no exceptions are raised when sending a Playbook\n        start callback to a current callback module plugin.\n        '
        register = self._register

        class CallbackModule(CallbackBase):
            """
            This is a callback module with the current
            method signature for `v2_playbook_on_start`.
            """
            CALLBACK_VERSION = 2.0
            CALLBACK_TYPE = 'notification'
            CALLBACK_NAME = 'current_module'

            def v2_playbook_on_start(self, playbook):
                if False:
                    print('Hello World!')
                register(self, playbook)
        callback_module = CallbackModule()
        self._tqm._callback_plugins.append(callback_module)
        self._tqm.send_callback('v2_playbook_on_start', self._playbook)
        register.assert_called_once_with(callback_module, self._playbook)

    def test_task_queue_manager_callbacks_v2_playbook_on_start_wrapped(self):
        if False:
            print('Hello World!')
        '\n        Assert that no exceptions are raised when sending a Playbook\n        start callback to a wrapped current callback module plugin.\n        '
        register = self._register

        def wrap_callback(func):
            if False:
                i = 10
                return i + 15
            '\n            This wrapper changes the exposed argument\n            names for a method from the original names\n            to (*args, **kwargs). This is used in order\n            to validate that wrappers which change par-\n            ameter names do not break the TQM callback\n            system.\n\n            :param func: function to decorate\n            :return: decorated function\n            '

            def wrapper(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return func(*args, **kwargs)
            return wrapper

        class WrappedCallbackModule(CallbackBase):
            """
            This is a callback module with the current
            method signature for `v2_playbook_on_start`
            wrapped in order to change the signature.
            """
            CALLBACK_VERSION = 2.0
            CALLBACK_TYPE = 'notification'
            CALLBACK_NAME = 'current_module'

            @wrap_callback
            def v2_playbook_on_start(self, playbook):
                if False:
                    print('Hello World!')
                register(self, playbook)
        callback_module = WrappedCallbackModule()
        self._tqm._callback_plugins.append(callback_module)
        self._tqm.send_callback('v2_playbook_on_start', self._playbook)
        register.assert_called_once_with(callback_module, self._playbook)