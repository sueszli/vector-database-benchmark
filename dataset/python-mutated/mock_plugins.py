from __future__ import annotations
from contextlib import ExitStack, contextmanager
from unittest import mock
PLUGINS_MANAGER_NULLABLE_ATTRIBUTES = ['plugins', 'registered_hooks', 'macros_modules', 'executors_modules', 'admin_views', 'flask_blueprints', 'menu_links', 'flask_appbuilder_views', 'flask_appbuilder_menu_links', 'global_operator_extra_links', 'operator_extra_links', 'registered_operator_link_classes', 'timetable_classes']

@contextmanager
def mock_plugin_manager(plugins=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Protects the initial state and sets the default state for the airflow.plugins module.\n\n    You can also overwrite variables by passing a keyword argument.\n\n    airflow.plugins_manager uses many global variables. To avoid side effects, this decorator performs\n    the following operations:\n\n    1. saves variables state,\n    2. set variables to default value,\n    3. executes context code,\n    4. restores the state of variables to the state from point 1.\n\n    Use this context if you want your test to not have side effects in airflow.plugins_manager, and\n    other tests do not affect the results of this test.\n    '
    illegal_arguments = set(kwargs.keys()) - set(PLUGINS_MANAGER_NULLABLE_ATTRIBUTES) - {'import_errors'}
    if illegal_arguments:
        raise TypeError(f'TypeError: mock_plugin_manager got an unexpected keyword arguments: {illegal_arguments}')
    with ExitStack() as exit_stack:

        def mock_loaded_plugins():
            if False:
                print('Hello World!')
            exit_stack.enter_context(mock.patch('airflow.plugins_manager.plugins', plugins or []))
        exit_stack.enter_context(mock.patch('airflow.plugins_manager.load_plugins_from_plugin_directory', side_effect=mock_loaded_plugins))
        for attr in PLUGINS_MANAGER_NULLABLE_ATTRIBUTES:
            exit_stack.enter_context(mock.patch(f'airflow.plugins_manager.{attr}', kwargs.get(attr)))
        exit_stack.enter_context(mock.patch('airflow.plugins_manager.plugins', None))
        exit_stack.enter_context(mock.patch('airflow.plugins_manager.import_errors', kwargs.get('import_errors', {})))
        yield