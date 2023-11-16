"""
Plugin dependency solver.
"""
import importlib
import logging
import traceback
import pkg_resources
from spyder.api.exceptions import SpyderAPIError
from spyder.api.plugins import Plugins
from spyder.api.utils import get_class_values
from spyder.config.base import STDERR
logger = logging.getLogger(__name__)

def find_internal_plugins():
    if False:
        print('Hello World!')
    '\n    Find internal plugins based on setuptools entry points.\n    '
    internal_plugins = {}
    entry_points = list(pkg_resources.iter_entry_points('spyder.plugins'))
    internal_names = get_class_values(Plugins)
    for entry_point in entry_points:
        name = entry_point.name
        if name not in internal_names:
            continue
        class_name = entry_point.attrs[0]
        mod = importlib.import_module(entry_point.module_name)
        plugin_class = getattr(mod, class_name, None)
        internal_plugins[name] = plugin_class
    internal_plugins = {key: value for (key, value) in sorted(internal_plugins.items())}
    return internal_plugins

def find_external_plugins():
    if False:
        while True:
            i = 10
    '\n    Find available external plugins based on setuptools entry points.\n    '
    internal_names = get_class_values(Plugins)
    plugins = list(pkg_resources.iter_entry_points('spyder.plugins'))
    external_plugins = {}
    for entry_point in plugins:
        name = entry_point.name
        if name not in internal_names:
            try:
                class_name = entry_point.attrs[0]
                mod = importlib.import_module(entry_point.module_name)
                plugin_class = getattr(mod, class_name, None)
                plugin_class._spyder_module_name = entry_point.module_name
                plugin_class._spyder_package_name = entry_point.dist.project_name
                plugin_class._spyder_version = entry_point.dist.version
                external_plugins[name] = plugin_class
                if name != plugin_class.NAME:
                    raise SpyderAPIError("Entry point name '{0}' and plugin.NAME '{1}' do not match!".format(name, plugin_class.NAME))
            except (ModuleNotFoundError, ImportError) as error:
                print('%s: %s' % (name, str(error)), file=STDERR)
                traceback.print_exc(file=STDERR)
    return external_plugins