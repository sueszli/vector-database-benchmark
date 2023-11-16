from __future__ import annotations
from argparse import Namespace
import pluggy
import pytest
from airflow import policies

@pytest.fixture
def plugin_manager():
    if False:
        print('Hello World!')
    pm = pluggy.PluginManager(policies.local_settings_hookspec.project_name)
    pm.add_hookspecs(policies)
    return pm

def test_local_settings_plain_function(plugin_manager: pluggy.PluginManager):
    if False:
        i = 10
        return i + 15
    'Test that a "plain" function from airflow_local_settings is registered via a plugin'
    called = False

    def dag_policy(dag):
        if False:
            print('Hello World!')
        nonlocal called
        called = True
    mod = Namespace(dag_policy=dag_policy)
    policies.make_plugin_from_local_settings(plugin_manager, mod, {'dag_policy'})
    plugin_manager.hook.dag_policy(dag='a')
    assert called

def test_local_settings_misnamed_argument(plugin_manager: pluggy.PluginManager):
    if False:
        while True:
            i = 10
    '\n    If an function in local_settings doesn\'t have the "correct" name we can\'t naively turn it in to a\n    plugin.\n\n    This tests the sig-mismatch detection and shimming code path\n    '
    called_with = None

    def dag_policy(wrong_arg_name):
        if False:
            for i in range(10):
                print('nop')
        nonlocal called_with
        called_with = wrong_arg_name
    mod = Namespace(dag_policy=dag_policy)
    policies.make_plugin_from_local_settings(plugin_manager, mod, {'dag_policy'})
    plugin_manager.hook.dag_policy(dag='passed_dag_value')
    assert called_with == 'passed_dag_value'