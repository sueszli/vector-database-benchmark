"""Spyder global registries for actions, toolbuttons, toolbars and menus."""
import inspect
import logging
from typing import Any, Optional, Dict
import warnings
import weakref
logger = logging.getLogger(__name__)

def get_caller(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get file and line where the methods that create actions, toolbuttons,\n    toolbars and menus are called.\n    '
    frames = []
    for frame in inspect.stack():
        if frame.code_context:
            code_context = frame.code_context[0]
        else:
            code_context = ''
        if func in code_context:
            frames.append(f'{frame.filename}:{frame.lineno}')
    frames = ', '.join(frames)
    return frames

class SpyderRegistry:
    """General registry for global references (per plugin) in Spyder."""

    def __init__(self, creation_func: str, obj_type: str=''):
        if False:
            print('Hello World!')
        self.registry_map = {}
        self.obj_type = obj_type
        self.creation_func = creation_func

    def register_reference(self, obj: Any, id_: str, plugin: Optional[str]=None, context: Optional[str]=None, overwrite: Optional[bool]=False):
        if False:
            return 10
        '\n        Register a reference `obj` for a given plugin name on a given context.\n\n        Parameters\n        ----------\n        obj: Any\n            Object to register as a reference.\n        id_: str\n            String identifier used to store the object reference.\n        plugin: Optional[str]\n            Plugin name used to store the reference. Should belong to\n            :class:`spyder.api.plugins.Plugins`. If None, then the object will\n            be stored under the global `main` key.\n        context: Optional[str]\n            Additional key used to store and identify the object reference.\n            In any Spyder plugin implementation, this context may refer to an\n            identifier of a widget. This context enables plugins to define\n            multiple actions with the same key that live on different widgets.\n            If None, this context will default to the special `__global`\n            identifier.\n        '
        plugin = plugin if plugin is not None else 'main'
        context = context if context is not None else '__global'
        plugin_contexts = self.registry_map.get(plugin, {})
        context_references = plugin_contexts.get(context, weakref.WeakValueDictionary())
        if id_ in context_references:
            try:
                frames = get_caller(self.creation_func)
                if not overwrite:
                    warnings.warn(f'There already exists a reference {context_references[id_]} with id {id_} under the context {context} of plugin {plugin}. The new reference {obj} will overwrite the previous reference. Hint: {obj} should have a different id_. See {frames}')
            except (RuntimeError, KeyError):
                pass
        logger.debug(f'Registering {obj} ({id_}) under context {context} for plugin {plugin}')
        context_references[id_] = obj
        plugin_contexts[context] = context_references
        self.registry_map[plugin] = plugin_contexts

    def get_reference(self, id_: str, plugin: Optional[str]=None, context: Optional[str]=None) -> Any:
        if False:
            return 10
        '\n        Retrieve a stored object reference under a given id of a specific\n        context of a given plugin name.\n\n        Parameters\n        ----------\n        id_: str\n            String identifier used to retrieve the object.\n        plugin: Optional[str]\n            Plugin name used to store the reference. Should belong to\n            :class:`spyder.api.plugins.Plugins`. If None, then the object will\n            be retrieved from the global `main` key.\n        context: Optional[str]\n            Additional key that was used to store the object reference.\n            In any Spyder plugin implementation, this context may refer to an\n            identifier of a widget. This context enables plugins to define\n            multiple actions with the same key that live on different widgets.\n            If None, this context will default to the special `__global`\n            identifier.\n\n        Returns\n        -------\n        obj: Any\n            The object that was stored under the given identifier.\n\n        Raises\n        ------\n        KeyError\n            If neither of `id_`, `plugin` or `context` were found in the\n            registry.\n        '
        plugin = plugin if plugin is not None else 'main'
        context = context if context is not None else '__global'
        plugin_contexts = self.registry_map[plugin]
        context_references = plugin_contexts[context]
        return context_references[id_]

    def get_references(self, plugin: Optional[str]=None, context: Optional[str]=None) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve all stored object references under the context of a\n        given plugin name.\n\n        Parameters\n        ----------\n        plugin: Optional[str]\n            Plugin name used to store the reference. Should belong to\n            :class:`spyder.api.plugins.Plugins`. If None, then the object will\n            be retrieved from the global `main` key.\n        context: Optional[str]\n            Additional key that was used to store the object reference.\n            In any Spyder plugin implementation, this context may refer to an\n            identifier of a widget. This context enables plugins to define\n            multiple actions with the same key that live on different widgets.\n            If None, this context will default to the special `__global`\n            identifier.\n\n        Returns\n        -------\n        objs: Dict[str, Any]\n            A dict that contains the actions mapped by their corresponding\n            identifiers.\n        '
        plugin = plugin if plugin is not None else 'main'
        context = context if context is not None else '__global'
        plugin_contexts = self.registry_map.get(plugin, {})
        context_references = plugin_contexts.get(context, weakref.WeakValueDictionary())
        return context_references

    def reset_registry(self):
        if False:
            for i in range(10):
                print('nop')
        self.registry_map = {}

    def __str__(self) -> str:
        if False:
            return 10
        return f'SpyderRegistry[{self.obj_type}, {self.registry_map}]'
ACTION_REGISTRY = SpyderRegistry('create_action', 'SpyderAction')
TOOLBUTTON_REGISTRY = SpyderRegistry('create_toolbutton', 'QToolButton')
TOOLBAR_REGISTRY = SpyderRegistry('create_toolbar', 'QToolBar')
MENU_REGISTRY = SpyderRegistry('create_menu', 'SpyderMenu')