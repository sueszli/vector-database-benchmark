from __future__ import annotations
import os
import re
import sys
from unittest import mock
from unittest.mock import MagicMock
import pendulum
import pytest
from dateutil import parser
from kubernetes.client import ApiClient, models as k8s
from airflow import __version__
from airflow.exceptions import AirflowConfigException
from airflow.providers.cncf.kubernetes.executors.kubernetes_executor import PodReconciliationError
from airflow.providers.cncf.kubernetes.pod_generator import PodDefaults, PodGenerator, datetime_to_label_safe_datestring, extend_object_field, merge_objects
from airflow.providers.cncf.kubernetes.secret import Secret
now = pendulum.now('UTC')

class TestPodGenerator:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.rand_str = 'abcd1234'
        self.deserialize_result = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': 'memory-demo', 'namespace': 'mem-example'}, 'spec': {'containers': [{'args': ['--vm', '1', '--vm-bytes', '150M', '--vm-hang', '1'], 'command': ['stress'], 'image': 'ghcr.io/apache/airflow-stress:1.0.4-2021.07.04', 'name': 'memory-demo-ctr', 'resources': {'limits': {'memory': '200Mi'}, 'requests': {'memory': '100Mi'}}}]}}
        self.envs = {'ENVIRONMENT': 'prod', 'LOG_LEVEL': 'warning'}
        self.secrets = [Secret('env', None, 'secret_a'), Secret('volume', '/etc/foo', 'secret_b'), Secret('env', 'TARGET', 'secret_b', 'source_b')]
        self.execution_date = parser.parse('2020-08-24 00:00:00.000000')
        self.execution_date_label = datetime_to_label_safe_datestring(self.execution_date)
        self.dag_id = 'dag_id'
        self.task_id = 'task_id'
        self.try_number = 3
        self.labels = {'airflow-worker': 'uuid', 'dag_id': self.dag_id, 'execution_date': self.execution_date_label, 'task_id': self.task_id, 'try_number': str(self.try_number), 'airflow_version': __version__.replace('+', '-'), 'kubernetes_executor': 'True'}
        self.annotations = {'dag_id': self.dag_id, 'task_id': self.task_id, 'execution_date': self.execution_date.isoformat(), 'try_number': str(self.try_number)}
        self.metadata = {'labels': self.labels, 'name': 'pod_id-' + self.rand_str, 'namespace': 'namespace', 'annotations': self.annotations}
        self.resources = k8s.V1ResourceRequirements(requests={'cpu': 1, 'memory': '1Gi', 'ephemeral-storage': '2Gi'}, limits={'cpu': 2, 'memory': '2Gi', 'ephemeral-storage': '4Gi', 'nvidia.com/gpu': 1})
        self.k8s_client = ApiClient()
        self.expected = k8s.V1Pod(api_version='v1', kind='Pod', metadata=k8s.V1ObjectMeta(namespace='default', name='myapp-pod-' + self.rand_str, labels={'app': 'myapp'}), spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', image='busybox', command=['sh', '-c', 'echo Hello Kubernetes!'], env=[k8s.V1EnvVar(name='ENVIRONMENT', value='prod'), k8s.V1EnvVar(name='LOG_LEVEL', value='warning'), k8s.V1EnvVar(name='TARGET', value_from=k8s.V1EnvVarSource(secret_key_ref=k8s.V1SecretKeySelector(name='secret_b', key='source_b')))], env_from=[k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name='configmap_a')), k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name='configmap_b')), k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name='secret_a'))], ports=[k8s.V1ContainerPort(name='foo', container_port=1234)], resources=k8s.V1ResourceRequirements(requests={'memory': '100Mi'}, limits={'memory': '200Mi'}))], security_context=k8s.V1PodSecurityContext(fs_group=2000, run_as_user=1000), host_network=True, image_pull_secrets=[k8s.V1LocalObjectReference(name='pull_secret_a'), k8s.V1LocalObjectReference(name='pull_secret_b')]))

    @mock.patch('airflow.providers.cncf.kubernetes.kubernetes_helper_functions.rand_str')
    def test_gen_pod_extract_xcom(self, mock_rand_str):
        if False:
            return 10
        '\n        Method gen_pod is used nowhere in codebase and is deprecated.\n        This test is only retained for backcompat.\n        '
        mock_rand_str.return_value = self.rand_str
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        pod_generator = PodGenerator(pod_template_file=path, extract_xcom=True)
        result = pod_generator.gen_pod()
        container_two = {'name': 'airflow-xcom-sidecar', 'image': 'alpine', 'command': ['sh', '-c', PodDefaults.XCOM_CMD], 'volumeMounts': [{'name': 'xcom', 'mountPath': '/airflow/xcom'}], 'resources': {'requests': {'cpu': '1m'}}}
        self.expected.spec.containers.append(container_two)
        base_container: k8s.V1Container = self.expected.spec.containers[0]
        base_container.volume_mounts = base_container.volume_mounts or []
        base_container.volume_mounts.append(k8s.V1VolumeMount(name='xcom', mount_path='/airflow/xcom'))
        self.expected.spec.containers[0] = base_container
        self.expected.spec.volumes = self.expected.spec.volumes or []
        self.expected.spec.volumes.append(k8s.V1Volume(name='xcom', empty_dir={}))
        result_dict = self.k8s_client.sanitize_for_serialization(result)
        expected_dict = self.k8s_client.sanitize_for_serialization(self.expected)
        assert result_dict == expected_dict

    def test_from_obj(self):
        if False:
            i = 10
            return i + 15
        result = PodGenerator.from_obj({'pod_override': k8s.V1Pod(api_version='v1', kind='Pod', metadata=k8s.V1ObjectMeta(name='foo', annotations={'test': 'annotation'}), spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', volume_mounts=[k8s.V1VolumeMount(mount_path='/foo/', name='example-kubernetes-test-volume')])], volumes=[k8s.V1Volume(name='example-kubernetes-test-volume', host_path=k8s.V1HostPathVolumeSource(path='/tmp/'))]))})
        result = self.k8s_client.sanitize_for_serialization(result)
        assert {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': 'foo', 'annotations': {'test': 'annotation'}}, 'spec': {'containers': [{'name': 'base', 'volumeMounts': [{'mountPath': '/foo/', 'name': 'example-kubernetes-test-volume'}]}], 'volumes': [{'hostPath': {'path': '/tmp/'}, 'name': 'example-kubernetes-test-volume'}]}} == result
        result = PodGenerator.from_obj({'KubernetesExecutor': {'annotations': {'test': 'annotation'}, 'volumes': [{'name': 'example-kubernetes-test-volume', 'hostPath': {'path': '/tmp/'}}], 'volume_mounts': [{'mountPath': '/foo/', 'name': 'example-kubernetes-test-volume'}]}})
        result_from_pod = PodGenerator.from_obj({'pod_override': k8s.V1Pod(metadata=k8s.V1ObjectMeta(annotations={'test': 'annotation'}), spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', volume_mounts=[k8s.V1VolumeMount(name='example-kubernetes-test-volume', mount_path='/foo/')])], volumes=[k8s.V1Volume(name='example-kubernetes-test-volume', host_path='/tmp/')]))})
        result = self.k8s_client.sanitize_for_serialization(result)
        result_from_pod = self.k8s_client.sanitize_for_serialization(result_from_pod)
        expected_from_pod = {'metadata': {'annotations': {'test': 'annotation'}}, 'spec': {'containers': [{'name': 'base', 'volumeMounts': [{'mountPath': '/foo/', 'name': 'example-kubernetes-test-volume'}]}], 'volumes': [{'hostPath': '/tmp/', 'name': 'example-kubernetes-test-volume'}]}}
        assert result_from_pod == expected_from_pod, 'There was a discrepancy between KubernetesExecutor and pod_override'
        assert {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'annotations': {'test': 'annotation'}}, 'spec': {'containers': [{'args': [], 'command': [], 'env': [], 'envFrom': [], 'name': 'base', 'ports': [], 'volumeMounts': [{'mountPath': '/foo/', 'name': 'example-kubernetes-test-volume'}]}], 'hostNetwork': False, 'imagePullSecrets': [], 'volumes': [{'hostPath': {'path': '/tmp/'}, 'name': 'example-kubernetes-test-volume'}]}} == result

    def test_reconcile_pods_empty_mutator_pod(self):
        if False:
            for i in range(10):
                print('nop')
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        pod_generator = PodGenerator(pod_template_file=path, extract_xcom=True)
        base_pod = pod_generator.ud_pod
        mutator_pod = None
        result = PodGenerator.reconcile_pods(base_pod, mutator_pod)
        assert base_pod == result
        mutator_pod = k8s.V1Pod()
        result = PodGenerator.reconcile_pods(base_pod, mutator_pod)
        assert base_pod == result

    @mock.patch('airflow.providers.cncf.kubernetes.kubernetes_helper_functions.rand_str')
    def test_reconcile_pods(self, mock_rand_str):
        if False:
            i = 10
            return i + 15
        mock_rand_str.return_value = self.rand_str
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        base_pod = PodGenerator(pod_template_file=path, extract_xcom=False).ud_pod
        mutator_pod = k8s.V1Pod(metadata=k8s.V1ObjectMeta(name='name2', labels={'bar': 'baz'}), spec=k8s.V1PodSpec(containers=[k8s.V1Container(image='', name='name', command=['/bin/command2.sh', 'arg2'], volume_mounts=[k8s.V1VolumeMount(mount_path='/foo/', name='example-kubernetes-test-volume2')])], volumes=[k8s.V1Volume(host_path=k8s.V1HostPathVolumeSource(path='/tmp/'), name='example-kubernetes-test-volume2')]))
        result = PodGenerator.reconcile_pods(base_pod, mutator_pod)
        expected: k8s.V1Pod = self.expected
        expected.metadata.name = 'name2'
        expected.metadata.labels['bar'] = 'baz'
        expected.spec.volumes = expected.spec.volumes or []
        expected.spec.volumes.append(k8s.V1Volume(host_path=k8s.V1HostPathVolumeSource(path='/tmp/'), name='example-kubernetes-test-volume2'))
        base_container: k8s.V1Container = expected.spec.containers[0]
        base_container.command = ['/bin/command2.sh', 'arg2']
        base_container.volume_mounts = [k8s.V1VolumeMount(mount_path='/foo/', name='example-kubernetes-test-volume2')]
        base_container.name = 'name'
        expected.spec.containers[0] = base_container
        result_dict = self.k8s_client.sanitize_for_serialization(result)
        expected_dict = self.k8s_client.sanitize_for_serialization(expected)
        assert result_dict == expected_dict

    @pytest.mark.parametrize('config_image, expected_image', [pytest.param('my_image:my_tag', 'my_image:my_tag', id='image_in_cfg'), pytest.param(None, 'busybox', id='no_image_in_cfg')])
    def test_construct_pod(self, config_image, expected_image):
        if False:
            print('Hello World!')
        template_file = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        worker_config = PodGenerator.deserialize_model_file(template_file)
        executor_config = k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='', resources=k8s.V1ResourceRequirements(limits={'cpu': '1m', 'memory': '1G'}))]))
        result = PodGenerator.construct_pod(dag_id=self.dag_id, task_id=self.task_id, pod_id='pod_id', kube_image=config_image, try_number=self.try_number, date=self.execution_date, args=['command'], pod_override_object=executor_config, base_worker_pod=worker_config, namespace='test_namespace', scheduler_job_id='uuid')
        expected = self.expected
        expected.metadata.labels = self.labels
        expected.metadata.labels['app'] = 'myapp'
        expected.metadata.annotations = self.annotations
        expected.metadata.name = 'pod_id'
        expected.metadata.namespace = 'test_namespace'
        expected.spec.containers[0].args = ['command']
        expected.spec.containers[0].image = expected_image
        expected.spec.containers[0].resources = {'limits': {'cpu': '1m', 'memory': '1G'}}
        expected.spec.containers[0].env.append(k8s.V1EnvVar(name='AIRFLOW_IS_K8S_EXECUTOR_POD', value='True'))
        result_dict = self.k8s_client.sanitize_for_serialization(result)
        expected_dict = self.k8s_client.sanitize_for_serialization(self.expected)
        assert expected_dict == result_dict

    def test_construct_pod_mapped_task(self):
        if False:
            i = 10
            return i + 15
        template_file = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base.yaml'
        worker_config = PodGenerator.deserialize_model_file(template_file)
        result = PodGenerator.construct_pod(dag_id=self.dag_id, task_id=self.task_id, pod_id='pod_id', try_number=self.try_number, kube_image='', map_index=0, date=self.execution_date, args=['command'], pod_override_object=None, base_worker_pod=worker_config, namespace='test_namespace', scheduler_job_id='uuid')
        expected = self.expected
        expected.metadata.labels = self.labels
        expected.metadata.labels['app'] = 'myapp'
        expected.metadata.labels['map_index'] = '0'
        expected.metadata.annotations = self.annotations
        expected.metadata.annotations['map_index'] = '0'
        expected.metadata.name = 'pod_id'
        expected.metadata.namespace = 'test_namespace'
        expected.spec.containers[0].args = ['command']
        del expected.spec.containers[0].env_from[1:]
        del expected.spec.containers[0].env[-1:]
        expected.spec.containers[0].env.append(k8s.V1EnvVar(name='AIRFLOW_IS_K8S_EXECUTOR_POD', value='True'))
        result_dict = self.k8s_client.sanitize_for_serialization(result)
        expected_dict = self.k8s_client.sanitize_for_serialization(expected)
        assert result_dict == expected_dict

    def test_construct_pod_empty_executor_config(self):
        if False:
            i = 10
            return i + 15
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        worker_config = PodGenerator.deserialize_model_file(path)
        executor_config = None
        result = PodGenerator.construct_pod(dag_id='dag_id', task_id='task_id', pod_id='pod_id', kube_image='test-image', try_number=3, date=self.execution_date, args=['command'], pod_override_object=executor_config, base_worker_pod=worker_config, namespace='namespace', scheduler_job_id='uuid')
        sanitized_result = self.k8s_client.sanitize_for_serialization(result)
        worker_config.spec.containers[0].image = 'test-image'
        worker_config.spec.containers[0].args = ['command']
        worker_config.metadata.annotations = self.annotations
        worker_config.metadata.labels = self.labels
        worker_config.metadata.labels['app'] = 'myapp'
        worker_config.metadata.name = 'pod_id'
        worker_config.metadata.namespace = 'namespace'
        worker_config.spec.containers[0].env.append(k8s.V1EnvVar(name='AIRFLOW_IS_K8S_EXECUTOR_POD', value='True'))
        worker_config_result = self.k8s_client.sanitize_for_serialization(worker_config)
        assert sanitized_result == worker_config_result

    @mock.patch('airflow.providers.cncf.kubernetes.kubernetes_helper_functions.rand_str')
    def test_construct_pod_attribute_error(self, mock_rand_str):
        if False:
            print('Hello World!')
        '\n        After upgrading k8s library we might get attribute error.\n        In this case it should raise PodReconciliationError\n        '
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod_generator_base_with_secrets.yaml'
        worker_config = PodGenerator.deserialize_model_file(path)
        mock_rand_str.return_value = self.rand_str
        executor_config = MagicMock()
        executor_config.side_effect = AttributeError('error')
        with pytest.raises(PodReconciliationError):
            PodGenerator.construct_pod(dag_id='dag_id', task_id='task_id', pod_id='pod_id', kube_image='test-image', try_number=3, date=self.execution_date, args=['command'], pod_override_object=executor_config, base_worker_pod=worker_config, namespace='namespace', scheduler_job_id='uuid')

    @mock.patch('airflow.providers.cncf.kubernetes.kubernetes_helper_functions.rand_str')
    def test_ensure_max_identifier_length(self, mock_rand_str):
        if False:
            print('Hello World!')
        mock_rand_str.return_value = self.rand_str
        path = os.path.join(os.path.dirname(__file__), 'pod_generator_base_with_secrets.yaml')
        worker_config = PodGenerator.deserialize_model_file(path)
        result = PodGenerator.construct_pod(dag_id='a' * 512, task_id='a' * 512, pod_id='a' * 512, kube_image='a' * 512, try_number=3, date=self.execution_date, args=['command'], namespace='namespace', scheduler_job_id='a' * 512, pod_override_object=None, base_worker_pod=worker_config)
        assert result.metadata.name == 'a' * 244 + '-' + self.rand_str
        for v in result.metadata.labels.values():
            assert len(v) <= 63
        assert 'a' * 512 == result.metadata.annotations['dag_id']
        assert 'a' * 512 == result.metadata.annotations['task_id']

    def test_merge_objects_empty(self):
        if False:
            for i in range(10):
                print('nop')
        annotations = {'foo1': 'bar1'}
        base_obj = k8s.V1ObjectMeta(annotations=annotations)
        client_obj = None
        res = merge_objects(base_obj, client_obj)
        assert base_obj == res
        client_obj = k8s.V1ObjectMeta()
        res = merge_objects(base_obj, client_obj)
        assert base_obj == res
        client_obj = k8s.V1ObjectMeta(annotations=annotations)
        base_obj = None
        res = merge_objects(base_obj, client_obj)
        assert client_obj == res
        base_obj = k8s.V1ObjectMeta()
        res = merge_objects(base_obj, client_obj)
        assert client_obj == res

    def test_merge_objects(self):
        if False:
            return 10
        base_annotations = {'foo1': 'bar1'}
        base_labels = {'foo1': 'bar1'}
        client_annotations = {'foo2': 'bar2'}
        base_obj = k8s.V1ObjectMeta(annotations=base_annotations, labels=base_labels)
        client_obj = k8s.V1ObjectMeta(annotations=client_annotations)
        res = merge_objects(base_obj, client_obj)
        client_obj.labels = base_labels
        assert client_obj == res

    def test_extend_object_field_empty(self):
        if False:
            for i in range(10):
                print('nop')
        ports = [k8s.V1ContainerPort(container_port=1, name='port')]
        base_obj = k8s.V1Container(name='base_container', ports=ports)
        client_obj = k8s.V1Container(name='client_container')
        res = extend_object_field(base_obj, client_obj, 'ports')
        client_obj.ports = ports
        assert client_obj == res
        base_obj = k8s.V1Container(name='base_container')
        client_obj = k8s.V1Container(name='base_container', ports=ports)
        res = extend_object_field(base_obj, client_obj, 'ports')
        assert client_obj == res

    def test_extend_object_field_not_list(self):
        if False:
            return 10
        base_obj = k8s.V1Container(name='base_container', image='image')
        client_obj = k8s.V1Container(name='client_container')
        with pytest.raises(ValueError):
            extend_object_field(base_obj, client_obj, 'image')
        base_obj = k8s.V1Container(name='base_container')
        client_obj = k8s.V1Container(name='client_container', image='image')
        with pytest.raises(ValueError):
            extend_object_field(base_obj, client_obj, 'image')

    def test_extend_object_field(self):
        if False:
            while True:
                i = 10
        base_ports = [k8s.V1ContainerPort(container_port=1, name='base_port')]
        base_obj = k8s.V1Container(name='base_container', ports=base_ports)
        client_ports = [k8s.V1ContainerPort(container_port=1, name='client_port')]
        client_obj = k8s.V1Container(name='client_container', ports=client_ports)
        res = extend_object_field(base_obj, client_obj, 'ports')
        client_obj.ports = base_ports + client_ports
        assert client_obj == res

    def test_reconcile_containers_empty(self):
        if False:
            for i in range(10):
                print('nop')
        base_objs = [k8s.V1Container(name='base_container')]
        client_objs = []
        res = PodGenerator.reconcile_containers(base_objs, client_objs)
        assert base_objs == res
        client_objs = [k8s.V1Container(name='client_container')]
        base_objs = []
        res = PodGenerator.reconcile_containers(base_objs, client_objs)
        assert client_objs == res
        res = PodGenerator.reconcile_containers([], [])
        assert res == []

    def test_reconcile_containers(self):
        if False:
            for i in range(10):
                print('nop')
        base_ports = [k8s.V1ContainerPort(container_port=1, name='base_port')]
        base_objs = [k8s.V1Container(name='base_container1', ports=base_ports), k8s.V1Container(name='base_container2', image='base_image')]
        client_ports = [k8s.V1ContainerPort(container_port=2, name='client_port')]
        client_objs = [k8s.V1Container(name='client_container1', ports=client_ports), k8s.V1Container(name='client_container2', image='client_image')]
        res = PodGenerator.reconcile_containers(base_objs, client_objs)
        client_objs[0].ports = base_ports + client_ports
        assert client_objs == res
        base_ports = [k8s.V1ContainerPort(container_port=1, name='base_port')]
        base_objs = [k8s.V1Container(name='base_container1', ports=base_ports), k8s.V1Container(name='base_container2', image='base_image')]
        client_ports = [k8s.V1ContainerPort(container_port=2, name='client_port')]
        client_objs = [k8s.V1Container(name='client_container1', ports=client_ports), k8s.V1Container(name='client_container2', stdin=True)]
        res = PodGenerator.reconcile_containers(base_objs, client_objs)
        client_objs[0].ports = base_ports + client_ports
        client_objs[1].image = 'base_image'
        assert client_objs == res

    def test_reconcile_specs_empty(self):
        if False:
            i = 10
            return i + 15
        base_spec = k8s.V1PodSpec(containers=[])
        client_spec = None
        res = PodGenerator.reconcile_specs(base_spec, client_spec)
        assert base_spec == res
        base_spec = None
        client_spec = k8s.V1PodSpec(containers=[])
        res = PodGenerator.reconcile_specs(base_spec, client_spec)
        assert client_spec == res

    def test_reconcile_specs(self):
        if False:
            i = 10
            return i + 15
        base_objs = [k8s.V1Container(name='base_container1', image='base_image')]
        client_objs = [k8s.V1Container(name='client_container1')]
        base_spec = k8s.V1PodSpec(priority=1, active_deadline_seconds=100, containers=base_objs)
        client_spec = k8s.V1PodSpec(priority=2, hostname='local', containers=client_objs)
        res = PodGenerator.reconcile_specs(base_spec, client_spec)
        client_spec.containers = [k8s.V1Container(name='client_container1', image='base_image')]
        client_spec.active_deadline_seconds = 100
        assert client_spec == res

    def test_reconcile_specs_init_containers(self):
        if False:
            return 10
        base_spec = k8s.V1PodSpec(containers=[], init_containers=[k8s.V1Container(name='base_container1')])
        client_spec = k8s.V1PodSpec(containers=[], init_containers=[k8s.V1Container(name='client_container1')])
        res = PodGenerator.reconcile_specs(base_spec, client_spec)
        assert res.init_containers == base_spec.init_containers + client_spec.init_containers

    def test_deserialize_model_file(self, caplog):
        if False:
            print('Hello World!')
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/pod.yaml'
        result = PodGenerator.deserialize_model_file(path)
        sanitized_res = self.k8s_client.sanitize_for_serialization(result)
        assert sanitized_res == self.deserialize_result
        assert len(caplog.records) == 0

    def test_deserialize_non_existent_model_file(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        path = sys.path[0] + '/tests/providers/cncf/kubernetes/non_existent.yaml'
        result = PodGenerator.deserialize_model_file(path)
        sanitized_res = self.k8s_client.sanitize_for_serialization(result)
        assert sanitized_res == {}
        assert len(caplog.records) == 1
        assert 'does not exist' in caplog.text

    @pytest.mark.parametrize('input', (pytest.param('a' * 70, id='max_label_length'), pytest.param('a' * 253, id='max_subdomain_length'), pytest.param('a' * 95, id='close to max'), pytest.param('aaa', id='tiny')))
    def test_pod_name_confirm_to_max_length(self, input):
        if False:
            return 10
        actual = PodGenerator.make_unique_pod_id(input)
        assert len(actual) <= 100
        (actual_base, actual_suffix) = actual.rsplit('-', maxsplit=1)
        assert actual_base == input[:91]
        assert re.match('^[a-z0-9]{8}$', actual_suffix)

    @pytest.mark.parametrize('pod_id, expected_starts_with', (('somewhat-long-pod-name-maybe-longer-than-previously-supported-with-hyphen-', 'somewhat-long-pod-name-maybe-longer-than-previously-supported-with-hyphen'), ('pod-name-with-hyphen-', 'pod-name-with-hyphen'), ('pod-name-with-double-hyphen--', 'pod-name-with-double-hyphen'), ('pod0-name', 'pod0-name'), ('simple', 'simple'), ('pod-name-with-dot.', 'pod-name-with-dot'), ('pod-name-with-double-dot..', 'pod-name-with-double-dot'), ('pod-name-with-hyphen-dot-.', 'pod-name-with-hyphen-dot')))
    def test_pod_name_is_valid(self, pod_id, expected_starts_with):
        if False:
            print('Hello World!')
        "\n        `make_unique_pod_id` doesn't actually guarantee that the regex passes for any input.\n        But I guess this test verifies that an otherwise valid pod_id doesn't get _screwed up_.\n        "
        actual = PodGenerator.make_unique_pod_id(pod_id)
        assert len(actual) <= 253
        assert actual == actual.lower(), 'not lowercase'
        regex = '^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$'
        assert re.match(regex, actual), 'pod_id is invalid - fails allowed regex check'
        assert actual.rsplit('-', 1)[0] == expected_starts_with
        assert re.match(f'^{expected_starts_with}-[a-z0-9]{{8}}$', actual), "doesn't match expected pattern"

    def test_validate_pod_generator(self):
        if False:
            return 10
        with pytest.raises(AirflowConfigException):
            PodGenerator(pod=k8s.V1Pod(), pod_template_file='k')
        with pytest.raises(AirflowConfigException):
            PodGenerator()
        PodGenerator(pod_template_file='tests/kubernetes/pod.yaml')
        PodGenerator(pod=k8s.V1Pod())

    @pytest.mark.parametrize('extra, extra_expected', [pytest.param(dict(), {}, id='base'), pytest.param(dict(airflow_worker=2), {'airflow-worker': '2'}, id='worker'), pytest.param(dict(map_index=2), {'map_index': '2'}, id='map_index'), pytest.param(dict(run_id='2'), {'run_id': '2'}, id='run_id'), pytest.param(dict(execution_date=now), {'execution_date': datetime_to_label_safe_datestring(now)}, id='date'), pytest.param(dict(airflow_worker=2, map_index=2, run_id='2', execution_date=now), {'airflow-worker': '2', 'map_index': '2', 'run_id': '2', 'execution_date': datetime_to_label_safe_datestring(now)}, id='all')])
    def test_build_labels_for_k8s_executor_pod(self, extra, extra_expected):
        if False:
            for i in range(10):
                print('nop')
        from airflow.version import version as airflow_version
        kwargs = dict(dag_id='dag*', task_id='task*', try_number=1)
        expected = dict(dag_id='dag-6b24921d4', task_id='task-b6aca8991', try_number='1', airflow_version=airflow_version, kubernetes_executor='True')
        labels = PodGenerator.build_labels_for_k8s_executor_pod(**kwargs, **extra)
        assert labels == {**expected, **extra_expected}
        items = [f'{k}={v}' for (k, v) in sorted(labels.items())]
        if 'airflow_worker' not in extra:
            items.append('airflow-worker')
        exp_selector = ','.join(items)
        assert PodGenerator.build_selector_for_k8s_executor_pod(**kwargs, **extra) == exp_selector