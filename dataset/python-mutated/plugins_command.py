from __future__ import annotations
import inspect
from typing import Any
from airflow import plugins_manager
from airflow.cli.simple_table import AirflowConsole
from airflow.plugins_manager import PluginsDirectorySource, get_plugin_info
from airflow.utils.cli import suppress_logs_and_warning
from airflow.utils.providers_configuration_loader import providers_configuration_loaded

def _get_name(class_like_object) -> str:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(class_like_object, (str, PluginsDirectorySource)):
        return str(class_like_object)
    if inspect.isclass(class_like_object):
        return class_like_object.__name__
    return class_like_object.__class__.__name__

def _join_plugins_names(value: list[Any] | Any) -> str:
    if False:
        for i in range(10):
            print('nop')
    value = value if isinstance(value, list) else [value]
    return ','.join((_get_name(v) for v in value))

@suppress_logs_and_warning
@providers_configuration_loaded
def dump_plugins(args):
    if False:
        for i in range(10):
            print('nop')
    'Dump plugins information.'
    plugins_info: list[dict[str, str]] = get_plugin_info()
    if not plugins_manager.plugins:
        print('No plugins loaded')
        return
    if args.output == 'table':
        for col in list(plugins_info[0]):
            if all((not bool(p[col]) for p in plugins_info)):
                for plugin in plugins_info:
                    del plugin[col]
    AirflowConsole().print_as(plugins_info, output=args.output)