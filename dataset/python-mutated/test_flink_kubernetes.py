from __future__ import annotations
import json
from unittest.mock import patch
import pytest
from airflow import DAG
from airflow.models import Connection
from airflow.providers.apache.flink.operators.flink_kubernetes import FlinkKubernetesOperator
from airflow.utils import db, timezone
pytestmark = pytest.mark.db_test
TEST_VALID_APPLICATION_YAML = '\napiVersion: flink.apache.org/v1beta1\nkind: FlinkDeployment\nmetadata:\n  name: flink-sm-ex\n  namespace: default\nspec:\n  image: flink:1.15\n  flinkVersion: v1_15\n  flinkConfiguration:\n    taskmanager.numberOfTaskSlots: "2"\n    state.savepoints.dir: file:///flink-data/savepoints\n    state.checkpoints.dir: file:///flink-data/checkpoints\n    high-availability: org.apache.flink.kubernetes.highavailability.KubernetesHaServicesFactory\n    high-availability.storageDir: file:///flink-data/ha\n  ingress:\n    template: "{{name}}.{{namespace}}.flink.k8s.io"\n  serviceAccount: flink\n  jobManager:\n    resource:\n      memory: "2048m"\n      cpu: 1\n  taskManager:\n    resource:\n      memory: "2048m"\n      cpu: 1\n    podTemplate:\n      apiVersion: v1\n      kind: Pod\n      metadata:\n        name: task-manager-pod-template\n      spec:\n        initContainers:\n          # Sample sidecar container\n          - name: busybox\n            image: busybox:latest\n            command: [ \'sh\',\'-c\',\'echo hello from task manager\' ]\n  job:\n    jarURI: local:///opt/flink/examples/streaming/StateMachineExample.jar\n    parallelism: 2\n    upgradeMode: stateless\n    state: running\n    savepointTriggerNonce: 0\n'
TEST_VALID_APPLICATION_JSON = '\n{\n  "apiVersion": "flink.apache.org/v1beta1",\n  "kind": "FlinkDeployment",\n  "metadata": {\n    "name": "flink-sm-ex",\n    "namespace": "default"\n  },\n  "spec": {\n    "image": "flink:1.15",\n    "flinkVersion": "v1_15",\n    "flinkConfiguration": {\n      "taskmanager.numberOfTaskSlots": "2",\n      "state.savepoints.dir": "file:///flink-data/savepoints",\n      "state.checkpoints.dir": "file:///flink-data/checkpoints",\n      "high-availability": "org.apache.flink.kubernetes.highavailability.KubernetesHaServicesFactory",\n      "high-availability.storageDir": "file:///flink-data/ha"\n    },\n    "ingress": {\n      "template": "{{name}}.{{namespace}}.flink.k8s.io"\n    },\n    "serviceAccount": "flink",\n    "jobManager": {\n      "resource": {\n        "memory": "2048m",\n        "cpu": 1\n      }\n    },\n    "taskManager": {\n      "resource": {\n        "memory": "2048m",\n        "cpu": 1\n      },\n      "podTemplate": {\n        "apiVersion": "v1",\n        "kind": "Pod",\n        "metadata": {\n          "name": "task-manager-pod-template"\n        },\n        "spec": {\n          "initContainers": [\n            {\n              "name": "busybox",\n              "image": "busybox:latest",\n              "command": [\n                "sh",\n                "-c",\n                "echo hello from task manager"\n              ]\n            }\n          ]\n        }\n      }\n    },\n    "job": {\n      "jarURI": "local:///opt/flink/examples/streaming/StateMachineExample.jar",\n      "parallelism": 2,\n      "upgradeMode": "stateless",\n      "state": "running",\n      "savepointTriggerNonce": 0\n    }\n  }\n}\n'
TEST_APPLICATION_DICT = {'apiVersion': 'flink.apache.org/v1beta1', 'kind': 'FlinkDeployment', 'metadata': {'name': 'flink-sm-ex', 'namespace': 'default'}, 'spec': {'image': 'flink:1.15', 'flinkVersion': 'v1_15', 'flinkConfiguration': {'taskmanager.numberOfTaskSlots': '2', 'state.savepoints.dir': 'file:///flink-data/savepoints', 'state.checkpoints.dir': 'file:///flink-data/checkpoints', 'high-availability': 'org.apache.flink.kubernetes.highavailability.KubernetesHaServicesFactory', 'high-availability.storageDir': 'file:///flink-data/ha'}, 'ingress': {'template': '{{name}}.{{namespace}}.flink.k8s.io'}, 'serviceAccount': 'flink', 'jobManager': {'resource': {'memory': '2048m', 'cpu': 1}}, 'taskManager': {'resource': {'memory': '2048m', 'cpu': 1}, 'podTemplate': {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': 'task-manager-pod-template'}, 'spec': {'initContainers': [{'name': 'busybox', 'image': 'busybox:latest', 'command': ['sh', '-c', 'echo hello from task manager']}]}}}, 'job': {'jarURI': 'local:///opt/flink/examples/streaming/StateMachineExample.jar', 'parallelism': 2, 'upgradeMode': 'stateless', 'state': 'running', 'savepointTriggerNonce': 0}}}

@patch('airflow.providers.cncf.kubernetes.hooks.kubernetes.KubernetesHook.get_conn')
class TestFlinkKubernetesOperator:

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='kubernetes_default_kube_config', conn_type='kubernetes', extra=json.dumps({})))
        db.merge_conn(Connection(conn_id='kubernetes_with_namespace', conn_type='kubernetes', extra=json.dumps({'extra__kubernetes__namespace': 'mock_namespace'})))
        args = {'owner': 'airflow', 'start_date': timezone.datetime(2020, 2, 1)}
        self.dag = DAG('test_dag_id', default_args=args)

    @patch('kubernetes.client.api.custom_objects_api.CustomObjectsApi.create_namespaced_custom_object')
    def test_create_application_from_yaml(self, mock_create_namespaced_crd, mock_kubernetes_hook):
        if False:
            while True:
                i = 10
        op = FlinkKubernetesOperator(application_file=TEST_VALID_APPLICATION_YAML, dag=self.dag, kubernetes_conn_id='kubernetes_default_kube_config', task_id='test_task_id')
        op.execute(None)
        mock_kubernetes_hook.assert_called_once_with()
        mock_create_namespaced_crd.assert_called_with(body=TEST_APPLICATION_DICT, group='flink.apache.org', namespace='default', plural='flinkdeployments', version='v1beta1')

    @patch('kubernetes.client.api.custom_objects_api.CustomObjectsApi.create_namespaced_custom_object')
    def test_create_application_from_json(self, mock_create_namespaced_crd, mock_kubernetes_hook):
        if False:
            return 10
        op = FlinkKubernetesOperator(application_file=TEST_VALID_APPLICATION_JSON, dag=self.dag, kubernetes_conn_id='kubernetes_default_kube_config', task_id='test_task_id')
        op.execute(None)
        mock_kubernetes_hook.assert_called_once_with()
        mock_create_namespaced_crd.assert_called_with(body=TEST_APPLICATION_DICT, group='flink.apache.org', namespace='default', plural='flinkdeployments', version='v1beta1')

    @patch('kubernetes.client.api.custom_objects_api.CustomObjectsApi.create_namespaced_custom_object')
    def test_create_application_from_json_with_api_group_and_version(self, mock_create_namespaced_crd, mock_kubernetes_hook):
        if False:
            while True:
                i = 10
        api_group = 'flink.apache.org'
        api_version = 'v1beta1'
        op = FlinkKubernetesOperator(application_file=TEST_VALID_APPLICATION_JSON, dag=self.dag, kubernetes_conn_id='kubernetes_default_kube_config', task_id='test_task_id', api_group=api_group, api_version=api_version)
        op.execute(None)
        mock_kubernetes_hook.assert_called_once_with()
        mock_create_namespaced_crd.assert_called_with(body=TEST_APPLICATION_DICT, group=api_group, namespace='default', plural='flinkdeployments', version=api_version)

    @patch('kubernetes.client.api.custom_objects_api.CustomObjectsApi.create_namespaced_custom_object')
    def test_namespace_from_operator(self, mock_create_namespaced_crd, mock_kubernetes_hook):
        if False:
            return 10
        op = FlinkKubernetesOperator(application_file=TEST_VALID_APPLICATION_JSON, dag=self.dag, namespace='operator_namespace', kubernetes_conn_id='kubernetes_with_namespace', task_id='test_task_id')
        op.execute(None)
        mock_kubernetes_hook.assert_called_once_with()
        mock_create_namespaced_crd.assert_called_with(body=TEST_APPLICATION_DICT, group='flink.apache.org', namespace='operator_namespace', plural='flinkdeployments', version='v1beta1')

    @patch('kubernetes.client.api.custom_objects_api.CustomObjectsApi.create_namespaced_custom_object')
    def test_namespace_from_connection(self, mock_create_namespaced_crd, mock_kubernetes_hook):
        if False:
            while True:
                i = 10
        op = FlinkKubernetesOperator(application_file=TEST_VALID_APPLICATION_JSON, dag=self.dag, kubernetes_conn_id='kubernetes_with_namespace', task_id='test_task_id')
        op.execute(None)
        mock_kubernetes_hook.assert_called_once_with()
        mock_create_namespaced_crd.assert_called_with(body=TEST_APPLICATION_DICT, group='flink.apache.org', namespace='mock_namespace', plural='flinkdeployments', version='v1beta1')