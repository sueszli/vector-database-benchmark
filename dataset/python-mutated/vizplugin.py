import sys
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union
from . import __version__
from .util import color_print, compare_version
if TYPE_CHECKING:
    from .viztracer import VizTracer

class VizPluginError(Exception):
    pass

class VizPluginBase:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def support_version(self) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError('Plugin of viztracer has to implement support_version method')

    def message(self, m_type: str, payload: Dict) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        This is the only logical interface with VizTracer. To make it simple and flexible,\n        we use m_type for message type, and the payload could be any json compatible\n        data. This is more extensible in the future\n        :param m_type str: the message type VizPlugin is receiving\n        :param payload dict: payload of the message\n\n        :return dict: always return a dict. Return None if nothing needs to be done\n                      by VizTracer. Otherwise refer to the docs\n        '
        if m_type == 'command':
            if payload['cmd_type'] == 'terminate':
                return {'success': True}
        return {}

class VizPluginManager:

    def __init__(self, tracer: 'VizTracer', plugins: Sequence[Union[VizPluginBase, str]]):
        if False:
            while True:
                i = 10
        self._tracer = tracer
        self._plugins = []
        for plugin in plugins:
            if isinstance(plugin, VizPluginBase):
                plugin_instance = plugin
            elif isinstance(plugin, str):
                plugin_instance = self._get_plugin_from_string(plugin)
            else:
                raise TypeError('Invalid plugin!')
            self._plugins.append(plugin_instance)
            support_version = plugin_instance.support_version()
            if compare_version(support_version, __version__) > 0:
                color_print('WARNING', 'The plugin support version is higher than viztracer version. Consider update your viztracer')
            self._send_message(plugin_instance, 'event', {'when': 'initialize'})

    def _get_plugin_from_string(self, plugin: str) -> VizPluginBase:
        if False:
            return 10
        args = plugin.split()
        module = args[0]
        try:
            package = __import__(module)
        except ImportError:
            print(f"There's no module named {module}, maybe you need to install it")
            sys.exit(1)
        m = package
        if '.' in module:
            names = module.split('.')
            try:
                for mod in names[1:]:
                    m = m.__getattribute__(mod)
            except AttributeError:
                raise ImportError(f'Unable to import {module}, wrong path')
        try:
            m = m.__getattribute__('get_vizplugin')
        except AttributeError:
            print(f'Unable to find get_vizplugin in {module}. Incorrect plugin.')
            sys.exit(1)
        if callable(m):
            return m(plugin)
        else:
            print(f'Unable to find get_vizplugin as a callable in {module}. Incorrect plugin.')
            sys.exit(1)

    def _send_message(self, plugin: VizPluginBase, m_type: str, payload: Dict) -> None:
        if False:
            while True:
                i = 10
        support_version = plugin.support_version()
        ret = plugin.message(m_type, payload)
        if m_type == 'command':
            self.assert_success(plugin, payload, ret)
        else:
            self.resolve(support_version, ret)

    @property
    def has_plugin(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return len(self._plugins) > 0

    def event(self, when: str) -> None:
        if False:
            return 10
        for plugin in self._plugins:
            self._send_message(plugin, 'event', {'when': when})

    def command(self, cmd: Dict) -> None:
        if False:
            i = 10
            return i + 15
        for plugin in self._plugins:
            self._send_message(plugin, 'command', cmd)

    def terminate(self) -> None:
        if False:
            print('Hello World!')
        self.command({'cmd_type': 'terminate'})
        for plugin in self._plugins:
            del plugin
        self._plugins = []

    def assert_success(self, plugin: VizPluginBase, cmd: Dict, ret: Optional[Dict]) -> None:
        if False:
            while True:
                i = 10
        if not ret or 'success' not in ret or (not ret['success']):
            raise VizPluginError(f'{plugin} failed to process {cmd}')

    def resolve(self, version: str, ret: Dict) -> None:
        if False:
            return 10
        if not ret or 'action' not in ret:
            return
        if ret['action'] == 'handle_data':
            ret['handler'](self._tracer.data)