from __future__ import annotations
import re
import pytest
from airflow.providers.cncf.kubernetes.kubernetes_helper_functions import create_pod_id
from airflow.providers.cncf.kubernetes.operators.pod import _create_pod_id
pod_name_regex = '^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$'

@pytest.mark.parametrize('create_pod_id', [pytest.param(_create_pod_id, id='provider'), pytest.param(create_pod_id, id='core')])
class TestCreatePodId:

    @pytest.mark.parametrize('val, expected', [('task-id', 'task-id'), ('task_id', 'task-id'), ('---task.id---', 'task-id'), ('.task.id', 'task-id'), ('**task.id', 'task-id'), ('-90Abc*&', '90abc'), ('90AçLbˆˆç˙ßß˜˜˙c*a', '90aclb-c-ssss-c-a')])
    def test_create_pod_id_task_only(self, val, expected, create_pod_id):
        if False:
            for i in range(10):
                print('nop')
        actual = create_pod_id(task_id=val, unique=False)
        assert actual == expected
        assert re.match(pod_name_regex, actual)

    @pytest.mark.parametrize('val, expected', [('dag-id', 'dag-id'), ('dag_id', 'dag-id'), ('---dag.id---', 'dag-id'), ('.dag.id', 'dag-id'), ('**dag.id', 'dag-id'), ('-90Abc*&', '90abc'), ('90AçLbˆˆç˙ßß˜˜˙c*a', '90aclb-c-ssss-c-a')])
    def test_create_pod_id_dag_only(self, val, expected, create_pod_id):
        if False:
            i = 10
            return i + 15
        actual = create_pod_id(dag_id=val, unique=False)
        assert actual == expected
        assert re.match(pod_name_regex, actual)

    @pytest.mark.parametrize('dag_id, task_id, expected', [('dag-id', 'task-id', 'dag-id-task-id'), ('dag_id', 'task_id', 'dag-id-task-id'), ('dag.id', 'task.id', 'dag-id-task-id'), ('.dag.id', '.---task.id', 'dag-id-task-id'), ('**dag.id', '**task.id', 'dag-id-task-id'), ('-90Abc*&', '-90Abc*&', '90abc-90abc'), ('90AçLbˆˆç˙ßß˜˜˙c*a', '90AçLbˆˆç˙ßß˜˜˙c*a', '90aclb-c-ssss-c-a-90aclb-c-ssss-c-a')])
    def test_create_pod_id_dag_and_task(self, dag_id, task_id, expected, create_pod_id):
        if False:
            print('Hello World!')
        actual = create_pod_id(dag_id=dag_id, task_id=task_id, unique=False)
        assert actual == expected
        assert re.match(pod_name_regex, actual)

    def test_create_pod_id_dag_too_long_with_suffix(self, create_pod_id):
        if False:
            while True:
                i = 10
        actual = create_pod_id('0' * 254)
        assert len(actual) == 80
        assert re.match('0{71}-[a-z0-9]{8}', actual)
        assert re.match(pod_name_regex, actual)

    def test_create_pod_id_dag_too_long_non_unique(self, create_pod_id):
        if False:
            i = 10
            return i + 15
        actual = create_pod_id('0' * 254, unique=False)
        assert len(actual) == 80
        assert re.match('0{80}', actual)
        assert re.match(pod_name_regex, actual)

    @pytest.mark.parametrize('unique', [True, False])
    @pytest.mark.parametrize('length', [25, 100, 200, 300])
    def test_create_pod_id(self, create_pod_id, length, unique):
        if False:
            while True:
                i = 10
        'Test behavior of max_length and unique.'
        dag_id = 'dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-dag-'
        task_id = 'task-task-task-task-task-task-task-task-task-task-task-task-task-task-task-task-task-'
        actual = create_pod_id(dag_id=dag_id, task_id=task_id, max_length=length, unique=unique)
        base = f'{dag_id}{task_id}'.strip('-')
        if unique:
            assert actual[:-9] == base[:length - 9].strip('-')
            assert re.match('-[a-z0-9]{8}', actual[-9:])
        else:
            assert actual == base[:length]