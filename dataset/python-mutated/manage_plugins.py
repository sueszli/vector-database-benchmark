"""Handle image reading, writing and plotting plugins.

To improve performance, plugins are only loaded as needed. As a result, there
can be multiple states for a given plugin:

    available: Defined in an *ini file located in ``skimage.io._plugins``.
        See also :func:`skimage.io.available_plugins`.
    partial definition: Specified in an *ini file, but not defined in the
        corresponding plugin module. This will raise an error when loaded.
    available but not on this system: Defined in ``skimage.io._plugins``, but
        a dependent library (e.g. Qt, PIL) is not available on your system.
        This will raise an error when loaded.
    loaded: The real availability is determined when it's explicitly loaded,
        either because it's one of the default plugins, or because it's
        loaded explicitly by the user.

"""
import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
__all__ = ['use_plugin', 'call_plugin', 'plugin_info', 'plugin_order', 'reset_plugins', 'find_available_plugins', 'available_plugins']
plugin_store = None
plugin_provides = {}
plugin_module_name = {}
plugin_meta_data = {}
preferred_plugins = {'all': ['imageio', 'pil', 'matplotlib'], 'imshow': ['matplotlib'], 'imshow_collection': ['matplotlib']}

def _clear_plugins():
    if False:
        print('Hello World!')
    'Clear the plugin state to the default, i.e., where no plugins are loaded'
    global plugin_store
    plugin_store = {'imread': [], 'imsave': [], 'imshow': [], 'imread_collection': [], 'imshow_collection': [], '_app_show': []}
_clear_plugins()

def _load_preferred_plugins():
    if False:
        return 10
    io_types = ['imsave', 'imshow', 'imread_collection', 'imshow_collection', 'imread']
    for p_type in io_types:
        _set_plugin(p_type, preferred_plugins['all'])
    plugin_types = (p for p in preferred_plugins.keys() if p != 'all')
    for p_type in plugin_types:
        _set_plugin(p_type, preferred_plugins[p_type])

def _set_plugin(plugin_type, plugin_list):
    if False:
        return 10
    for plugin in plugin_list:
        if plugin not in available_plugins:
            continue
        try:
            use_plugin(plugin, kind=plugin_type)
            break
        except (ImportError, RuntimeError, OSError):
            pass

def reset_plugins():
    if False:
        return 10
    _clear_plugins()
    _load_preferred_plugins()

def _parse_config_file(filename):
    if False:
        while True:
            i = 10
    'Return plugin name and meta-data dict from plugin config file.'
    parser = ConfigParser()
    parser.read(filename)
    name = parser.sections()[0]
    meta_data = {}
    for opt in parser.options(name):
        meta_data[opt] = parser.get(name, opt)
    return (name, meta_data)

def _scan_plugins():
    if False:
        for i in range(10):
            print('nop')
    'Scan the plugins directory for .ini files and parse them\n    to gather plugin meta-data.\n    '
    pd = os.path.dirname(__file__)
    config_files = glob(os.path.join(pd, '_plugins', '*.ini'))
    for filename in config_files:
        (name, meta_data) = _parse_config_file(filename)
        if 'provides' not in meta_data:
            warnings.warn(f'file {filename} not recognized as a scikit-image io plugin, skipping.')
            continue
        plugin_meta_data[name] = meta_data
        provides = [s.strip() for s in meta_data['provides'].split(',')]
        valid_provides = [p for p in provides if p in plugin_store]
        for p in provides:
            if p not in plugin_store:
                print(f'Plugin `{name}` wants to provide non-existent `{p}`. Ignoring.')
        need_to_add_collection = 'imread_collection' not in valid_provides and 'imread' in valid_provides
        if need_to_add_collection:
            valid_provides.append('imread_collection')
        plugin_provides[name] = valid_provides
        plugin_module_name[name] = os.path.basename(filename)[:-4]
_scan_plugins()

def find_available_plugins(loaded=False):
    if False:
        print('Hello World!')
    'List available plugins.\n\n    Parameters\n    ----------\n    loaded : bool\n        If True, show only those plugins currently loaded.  By default,\n        all plugins are shown.\n\n    Returns\n    -------\n    p : dict\n        Dictionary with plugin names as keys and exposed functions as\n        values.\n\n    '
    active_plugins = set()
    for plugin_func in plugin_store.values():
        for (plugin, func) in plugin_func:
            active_plugins.add(plugin)
    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin] if not f.startswith('_')]
    return d
available_plugins = find_available_plugins()

def call_plugin(kind, *args, **kwargs):
    if False:
        while True:
            i = 10
    "Find the appropriate plugin of 'kind' and execute it.\n\n    Parameters\n    ----------\n    kind : {'imshow', 'imsave', 'imread', 'imread_collection'}\n        Function to look up.\n    plugin : str, optional\n        Plugin to load.  Defaults to None, in which case the first\n        matching plugin is used.\n    *args, **kwargs : arguments and keyword arguments\n        Passed to the plugin function.\n\n    "
    if kind not in plugin_store:
        raise ValueError(f'Invalid function ({kind}) requested.')
    plugin_funcs = plugin_store[kind]
    if len(plugin_funcs) == 0:
        msg = f'No suitable plugin registered for {kind}.\n\nYou may load I/O plugins with the `skimage.io.use_plugin` command.  A list of all available plugins are shown in the `skimage.io` docstring.'
        raise RuntimeError(msg)
    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        (_, func) = plugin_funcs[0]
    else:
        _load(plugin)
        try:
            func = [f for (p, f) in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    return func(*args, **kwargs)

def use_plugin(name, kind=None):
    if False:
        return 10
    "Set the default plugin for a specified operation.  The plugin\n    will be loaded if it hasn't been already.\n\n    Parameters\n    ----------\n    name : str\n        Name of plugin. See ``skimage.io.available_plugins`` for a list of available\n        plugins.\n    kind : {'imsave', 'imread', 'imshow', 'imread_collection', 'imshow_collection'}, optional\n        Set the plugin for this function.  By default,\n        the plugin is set for all functions.\n\n    Examples\n    --------\n    To use Matplotlib as the default image reader, you would write:\n\n    >>> from skimage import io\n    >>> io.use_plugin('matplotlib', 'imread')\n\n    To see a list of available plugins run ``skimage.io.available_plugins``. Note\n    that this lists plugins that are defined, but the full list may not be usable\n    if your system does not have the required libraries installed.\n\n    "
    if kind is None:
        kind = plugin_store.keys()
    else:
        if kind not in plugin_provides[name]:
            raise RuntimeError(f'Plugin {name} does not support `{kind}`.')
        if kind == 'imshow':
            kind = [kind, '_app_show']
        else:
            kind = [kind]
    _load(name)
    for k in kind:
        if k not in plugin_store:
            raise RuntimeError(f"'{k}' is not a known plugin function.")
        funcs = plugin_store[k]
        funcs = [(n, f) for (n, f) in funcs if n == name] + [(n, f) for (n, f) in funcs if n != name]
        plugin_store[k] = funcs

def _inject_imread_collection_if_needed(module):
    if False:
        print('Hello World!')
    'Add `imread_collection` to module if not already present.'
    if not hasattr(module, 'imread_collection') and hasattr(module, 'imread'):
        imread = getattr(module, 'imread')
        func = imread_collection_wrapper(imread)
        setattr(module, 'imread_collection', func)

def _load(plugin):
    if False:
        for i in range(10):
            print('nop')
    'Load the given plugin.\n\n    Parameters\n    ----------\n    plugin : str\n        Name of plugin to load.\n\n    See Also\n    --------\n    plugins : List of available plugins\n\n    '
    if plugin in find_available_plugins(loaded=True):
        return
    if plugin not in plugin_module_name:
        raise ValueError(f'Plugin {plugin} not found.')
    else:
        modname = plugin_module_name[plugin]
        plugin_module = __import__('skimage.io._plugins.' + modname, fromlist=[modname])
    provides = plugin_provides[plugin]
    for p in provides:
        if p == 'imread_collection':
            _inject_imread_collection_if_needed(plugin_module)
        elif not hasattr(plugin_module, p):
            print(f'Plugin {plugin} does not provide {p} as advertised.  Ignoring.')
            continue
        store = plugin_store[p]
        func = getattr(plugin_module, p)
        if (plugin, func) not in store:
            store.append((plugin, func))

def plugin_info(plugin):
    if False:
        while True:
            i = 10
    'Return plugin meta-data.\n\n    Parameters\n    ----------\n    plugin : str\n        Name of plugin.\n\n    Returns\n    -------\n    m : dict\n        Meta data as specified in plugin ``.ini``.\n\n    '
    try:
        return plugin_meta_data[plugin]
    except KeyError:
        raise ValueError(f'No information on plugin "{plugin}"')

def plugin_order():
    if False:
        i = 10
        return i + 15
    'Return the currently preferred plugin order.\n\n    Returns\n    -------\n    p : dict\n        Dictionary of preferred plugin order, with function name as key and\n        plugins (in order of preference) as value.\n\n    '
    p = {}
    for func in plugin_store:
        p[func] = [plugin_name for (plugin_name, f) in plugin_store[func]]
    return p