"""
This is an example dag for using a Kubernetes Executor Configuration.
"""
from __future__ import annotations
import logging
import os
import pendulum
from airflow.configuration import conf
from airflow.decorators import task
from airflow.example_dags.libs.helper import print_stuff
from airflow.models.dag import DAG
log = logging.getLogger(__name__)
try:
    from kubernetes.client import models as k8s
except ImportError:
    log.warning('The example_kubernetes_executor example DAG requires the kubernetes provider. Please install it with: pip install apache-airflow[cncf.kubernetes]')
    k8s = None
if k8s:
    with DAG(dag_id='example_kubernetes_executor', schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example3']) as dag:
        start_task_executor_config = {'pod_override': k8s.V1Pod(metadata=k8s.V1ObjectMeta(annotations={'test': 'annotation'}))}

        @task(executor_config=start_task_executor_config)
        def start_task():
            if False:
                i = 10
                return i + 15
            print_stuff()
        executor_config_volume_mount = {'pod_override': k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', volume_mounts=[k8s.V1VolumeMount(mount_path='/foo/', name='example-kubernetes-test-volume')])], volumes=[k8s.V1Volume(name='example-kubernetes-test-volume', host_path=k8s.V1HostPathVolumeSource(path='/tmp/'))]))}

        @task(executor_config=executor_config_volume_mount)
        def test_volume_mount():
            if False:
                return 10
            '\n            Tests whether the volume has been mounted.\n            '
            with open('/foo/volume_mount_test.txt', 'w') as foo:
                foo.write('Hello')
            return_code = os.system('cat /foo/volume_mount_test.txt')
            if return_code != 0:
                raise ValueError(f'Error when checking volume mount. Return code {return_code}')
        volume_task = test_volume_mount()
        executor_config_sidecar = {'pod_override': k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', volume_mounts=[k8s.V1VolumeMount(mount_path='/shared/', name='shared-empty-dir')]), k8s.V1Container(name='sidecar', image='ubuntu', args=['echo "retrieved from mount" > /shared/test.txt'], command=['bash', '-cx'], volume_mounts=[k8s.V1VolumeMount(mount_path='/shared/', name='shared-empty-dir')])], volumes=[k8s.V1Volume(name='shared-empty-dir', empty_dir=k8s.V1EmptyDirVolumeSource())]))}

        @task(executor_config=executor_config_sidecar)
        def test_sharedvolume_mount():
            if False:
                for i in range(10):
                    print('nop')
            '\n            Tests whether the volume has been mounted.\n            '
            for i in range(5):
                try:
                    return_code = os.system('cat /shared/test.txt')
                    if return_code != 0:
                        raise ValueError(f'Error when checking volume mount. Return code {return_code}')
                except ValueError as e:
                    if i > 4:
                        raise e
        sidecar_task = test_sharedvolume_mount()
        executor_config_non_root = {'pod_override': k8s.V1Pod(metadata=k8s.V1ObjectMeta(labels={'release': 'stable'}))}

        @task(executor_config=executor_config_non_root)
        def non_root_task():
            if False:
                return 10
            print_stuff()
        third_task = non_root_task()
        executor_config_other_ns = {'pod_override': k8s.V1Pod(metadata=k8s.V1ObjectMeta(namespace='test-namespace', labels={'release': 'stable'}))}

        @task(executor_config=executor_config_other_ns)
        def other_namespace_task():
            if False:
                i = 10
                return i + 15
            print_stuff()
        other_ns_task = other_namespace_task()
        worker_container_repository = conf.get('kubernetes_executor', 'worker_container_repository')
        worker_container_tag = conf.get('kubernetes_executor', 'worker_container_tag')
        kube_exec_config_special = {'pod_override': k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', image=f'{worker_container_repository}:{worker_container_tag}')]))}

        @task(executor_config=kube_exec_config_special)
        def base_image_override_task():
            if False:
                for i in range(10):
                    print('nop')
            print_stuff()
        base_image_task = base_image_override_task()
        k8s_affinity = k8s.V1Affinity(pod_anti_affinity=k8s.V1PodAntiAffinity(required_during_scheduling_ignored_during_execution=[k8s.V1PodAffinityTerm(label_selector=k8s.V1LabelSelector(match_expressions=[k8s.V1LabelSelectorRequirement(key='app', operator='In', values=['airflow'])]), topology_key='kubernetes.io/hostname')]))
        k8s_tolerations = [k8s.V1Toleration(key='dedicated', operator='Equal', value='airflow')]
        k8s_resource_requirements = k8s.V1ResourceRequirements(requests={'memory': '512Mi'}, limits={'memory': '512Mi'})
        kube_exec_config_resource_limits = {'pod_override': k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', resources=k8s_resource_requirements)], affinity=k8s_affinity, tolerations=k8s_tolerations))}

        @task(executor_config=kube_exec_config_resource_limits)
        def task_with_resource_limits():
            if False:
                print('Hello World!')
            print_stuff()
        four_task = task_with_resource_limits()
        start_task() >> [volume_task, other_ns_task, sidecar_task] >> third_task >> [base_image_task, four_task]