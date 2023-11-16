"""Utilities for unit testing DAGs."""
from airflow import models
from airflow.utils.dag_cycle_tester import test_cycle

def assert_has_valid_dag(module):
    if False:
        while True:
            i = 10
    'Assert that a module contains a valid DAG.'
    no_dag_found = True
    for dag in vars(module).values():
        if isinstance(dag, models.DAG):
            no_dag_found = False
            test_cycle(dag)
    if no_dag_found:
        raise AssertionError('module does not contain a valid DAG')