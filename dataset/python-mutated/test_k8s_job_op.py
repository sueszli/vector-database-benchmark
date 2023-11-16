import uuid
import kubernetes
import pytest
from dagster import RetryRequested, job, op
from dagster._core.test_utils import instance_for_test
from dagster_k8s import execute_k8s_job, k8s_job_op
from dagster_k8s.client import DagsterK8sError, DagsterKubernetesClient
from dagster_k8s.job import get_k8s_job_name

def _get_pods_logs(cluster_provider, job_name, namespace, container_name=None):
    if False:
        return 10
    kubernetes.config.load_kube_config(cluster_provider.kubeconfig_file)
    api_client = DagsterKubernetesClient.production_client()
    pod_names = api_client.get_pod_names_in_job(job_name, namespace=namespace)
    pods_logs = []
    for pod_name in pod_names:
        pod_logs = api_client.retrieve_pod_logs(pod_name, namespace=namespace, container_name=container_name)
        pods_logs.append(pod_logs)
    return pods_logs

def _get_pod_logs(cluster_provider, job_name, namespace, container_name=None):
    if False:
        return 10
    return _get_pods_logs(cluster_provider, job_name, namespace, container_name)[0]

@pytest.mark.default
def test_k8s_job_op(namespace, cluster_provider):
    if False:
        print('Hello World!')
    first_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo HI'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file}, name='first_op')
    second_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo GOODBYE'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file}, name='second_op')

    @job
    def my_full_job():
        if False:
            i = 10
            return i + 15
        second_op(first_op())
    execute_result = my_full_job.execute_in_process()
    assert execute_result.success
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, first_op.name)
    assert 'HI' in _get_pod_logs(cluster_provider, job_name, namespace)
    job_name = get_k8s_job_name(run_id, second_op.name)
    assert 'GOODBYE' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_custom_k8s_op_override_job_name(namespace, cluster_provider):
    if False:
        print('Hello World!')
    custom_k8s_job_name = str(uuid.uuid4())

    @op
    def my_custom_op(context):
        if False:
            return 10
        execute_k8s_job(context, image='busybox', command=['/bin/sh', '-c'], args=['echo HI'], namespace=namespace, load_incluster_config=False, kubeconfig_file=cluster_provider.kubeconfig_file, k8s_job_name=custom_k8s_job_name)

    @job
    def my_job_with_custom_ops():
        if False:
            print('Hello World!')
        my_custom_op()
    execute_result = my_job_with_custom_ops.execute_in_process()
    assert execute_result.success
    assert 'HI' in _get_pod_logs(cluster_provider, custom_k8s_job_name, namespace)

@pytest.mark.default
def test_custom_k8s_op(namespace, cluster_provider):
    if False:
        while True:
            i = 10

    @op
    def my_custom_op(context):
        if False:
            for i in range(10):
                print('nop')
        execute_k8s_job(context, image='busybox', command=['/bin/sh', '-c'], args=['echo HI'], namespace=namespace, load_incluster_config=False, kubeconfig_file=cluster_provider.kubeconfig_file)
        return 'GOODBYE'

    @op
    def my_second_custom_op(context, what_to_echo: str):
        if False:
            print('Hello World!')
        execute_k8s_job(context, image='busybox', command=['/bin/sh', '-c'], args=[f'echo {what_to_echo}'], namespace=namespace, load_incluster_config=False, kubeconfig_file=cluster_provider.kubeconfig_file)

    @job
    def my_job_with_custom_ops():
        if False:
            for i in range(10):
                print('nop')
        my_second_custom_op(my_custom_op())
    execute_result = my_job_with_custom_ops.execute_in_process()
    assert execute_result.success
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, my_custom_op.name)
    assert 'HI' in _get_pod_logs(cluster_provider, job_name, namespace)
    job_name = get_k8s_job_name(run_id, my_second_custom_op.name)
    assert 'GOODBYE' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_k8s_job_op_with_timeout_success(namespace, cluster_provider):
    if False:
        i = 10
        return i + 15
    first_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo HI'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'timeout': 600}, name='first_op')

    @job
    def my_full_job():
        if False:
            return 10
        first_op()
    execute_result = my_full_job.execute_in_process()
    assert execute_result.success
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, first_op.name)
    assert 'HI' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_k8s_job_op_with_timeout_fail(namespace, cluster_provider):
    if False:
        return 10
    timeout_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['sleep 15 && echo HI'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'timeout': 5}, name='timeout_op')

    @job
    def timeout_job():
        if False:
            print('Hello World!')
        timeout_op()
    with pytest.raises(DagsterK8sError, match='Timed out while waiting for pod to become ready'):
        timeout_job.execute_in_process()

@pytest.mark.default
def test_k8s_job_op_with_failure(namespace, cluster_provider):
    if False:
        for i in range(10):
            print('nop')
    failure_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['sleep 10 && exit 1'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'timeout': 5}, name='failure_op')

    @job
    def failure_job():
        if False:
            return 10
        failure_op()
    with pytest.raises(DagsterK8sError):
        failure_job.execute_in_process()

@pytest.mark.default
def test_k8s_job_op_with_container_config(namespace, cluster_provider):
    if False:
        print('Hello World!')
    with_container_config = k8s_job_op.configured({'image': 'busybox', 'container_config': {'command': ['echo', 'SHELL_FROM_CONTAINER_CONFIG']}, 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file}, name='with_container_config')

    @job
    def with_config_job():
        if False:
            i = 10
            return i + 15
        with_container_config()
    execute_result = with_config_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, with_container_config.name)
    assert 'SHELL_FROM_CONTAINER_CONFIG' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_k8s_job_op_with_deep_merge(namespace, cluster_provider):
    if False:
        i = 10
        return i + 15
    with instance_for_test(overrides={'run_launcher': {'module': 'dagster_k8s', 'class': 'K8sRunLauncher', 'config': {'instance_config_map': 'doesnt_matter', 'service_account_name': 'default', 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'run_k8s_config': {'container_config': {'env': [{'name': 'FOO', 'value': '1'}]}}}}}) as instance:

        @job
        def with_config_job():
            if False:
                return 10
            k8s_job_op()
        execute_result = with_config_job.execute_in_process(instance=instance, run_config={'ops': {'k8s_job_op': {'config': {'image': 'busybox', 'container_config': {'command': ['/bin/sh', '-c'], 'args': ['echo "FOO IS $FOO AND BAR IS $BAR"'], 'env': [{'name': 'BAR', 'value': '2'}]}, 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file}}}})
        run_id = execute_result.dagster_run.run_id
        job_name = get_k8s_job_name(run_id, k8s_job_op.name)
        assert 'FOO IS  AND BAR IS 2' in _get_pod_logs(cluster_provider, job_name, namespace)
        execute_result = with_config_job.execute_in_process(instance=instance, run_config={'ops': {'k8s_job_op': {'config': {'image': 'busybox', 'container_config': {'command': ['/bin/sh', '-c'], 'args': ['echo "FOO IS $FOO AND BAR IS $BAR"'], 'env': [{'name': 'BAR', 'value': '2'}]}, 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'merge_behavior': 'DEEP'}}}})
        run_id = execute_result.dagster_run.run_id
        job_name = get_k8s_job_name(run_id, k8s_job_op.name)
        assert 'FOO IS 1 AND BAR IS 2' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_k8s_job_op_with_container_config_and_command(namespace, cluster_provider):
    if False:
        i = 10
        return i + 15
    with_container_config = k8s_job_op.configured({'image': 'busybox', 'container_config': {'command': ['echo', 'SHELL_FROM_CONTAINER_CONFIG']}, 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'command': ['echo', 'OVERRIDES_CONTAINER_CONFIG']}, name='with_container_config')

    @job
    def with_config_job():
        if False:
            return 10
        with_container_config()
    execute_result = with_config_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, with_container_config.name)
    assert 'OVERRIDES_CONTAINER_CONFIG' in _get_pod_logs(cluster_provider, job_name, namespace)

@pytest.mark.default
def test_k8s_job_op_with_multiple_containers(namespace, cluster_provider):
    if False:
        i = 10
        return i + 15
    with_multiple_containers = k8s_job_op.configured({'image': 'busybox', 'container_config': {'name': 'first-container'}, 'command': ['/bin/sh', '-c'], 'args': ['echo MAIN_CONTAINER'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'pod_spec_config': {'containers': [{'name': 'other-container', 'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo OTHER_CONTAINER']}]}}, name='with_multiple_containers')

    @job
    def with_multiple_containers_job():
        if False:
            return 10
        with_multiple_containers()
    execute_result = with_multiple_containers_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, with_multiple_containers.name)
    assert 'MAIN_CONTAINER' in _get_pod_logs(cluster_provider, job_name, namespace, container_name='first-container')
    assert 'OTHER_CONTAINER' in _get_pod_logs(cluster_provider, job_name, namespace, container_name='other-container')

@pytest.mark.default
def test_k8s_job_op_retries(namespace, cluster_provider):
    if False:
        print('Hello World!')

    @op
    def fails_sometimes(context):
        if False:
            print('Hello World!')
        execute_k8s_job(context, image='busybox', command=['/bin/sh', '-c'], args=[f'echo HERE IS RETRY NUMBER {context.retry_number}'], namespace=namespace, load_incluster_config=False, kubeconfig_file=cluster_provider.kubeconfig_file)
        if context.retry_number == 0:
            raise RetryRequested(max_retries=1, seconds_to_wait=1)

    @job
    def fails_sometimes_job():
        if False:
            i = 10
            return i + 15
        fails_sometimes()
    execute_result = fails_sometimes_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, fails_sometimes.name)
    assert 'HERE IS RETRY NUMBER 0' in _get_pod_logs(cluster_provider, job_name, namespace)
    assert 'HERE IS RETRY NUMBER 1' in _get_pod_logs(cluster_provider, job_name + '-1', namespace)

@pytest.mark.default
def test_k8s_job_op_ignore_job_tags(namespace, cluster_provider):
    if False:
        for i in range(10):
            print('nop')

    @op
    def the_op(context):
        if False:
            for i in range(10):
                print('nop')
        execute_k8s_job(context, image='busybox', command=['/bin/sh', '-c'], args=['echo DID I GET CONFIG? $THE_ENV_VAR_FROM_JOB $THE_ENV_VAR_FROM_OP'], namespace=namespace, load_incluster_config=False, kubeconfig_file=cluster_provider.kubeconfig_file, container_config={'env': [{'name': 'THE_ENV_VAR_FROM_OP', 'value': 'FROM_OP_TAGS'}]})

    @job(tags={'dagster-k8s/config': {'container_config': {'env': [{'name': 'THE_ENV_VAR_FROM_JOB', 'value': 'FROM_JOB_TAGS'}]}}})
    def tagged_job():
        if False:
            for i in range(10):
                print('nop')
        the_op()
    execute_result = tagged_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, the_op.name)
    pod_logs = _get_pod_logs(cluster_provider, job_name, namespace)
    assert 'FROM_JOB_TAGS' not in pod_logs
    assert 'FROM_OP_TAGS' in pod_logs

@pytest.mark.default
def test_k8s_job_op_with_paralellism(namespace, cluster_provider):
    if False:
        print('Hello World!')
    with_parallelism = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo HI'], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'job_spec_config': {'parallelism': 2, 'completions': 2}}, name='with_parallelism')

    @job
    def with_parallelism_job():
        if False:
            return 10
        with_parallelism()
    execute_result = with_parallelism_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, with_parallelism.name)
    pods_logs = _get_pods_logs(cluster_provider, job_name, namespace)
    assert 'HI' in pods_logs[0]
    assert 'HI' in pods_logs[1]

@pytest.mark.default
def test_k8s_job_op_with_restart_policy(namespace, cluster_provider):
    if False:
        for i in range(10):
            print('nop')
    'This tests works by creating a file in a volume mount, and then incrementing the number\n    in the file on each retry. If the number is 2, then the pod will succeed. Otherwise, it will\n    fail. This is to test that the pod restart policy is working as expected.\n    '
    with_restart_policy = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['filename=/data/retries; (count=$(cat $filename) && echo $(($count+1)) > $filename) || (touch $filename && echo 0 > $filename); retries=$(cat $filename); if [ "$retries" = "2" ]; then echo HI && exit 0; else exit 1; fi;'], 'volume_mounts': [{'name': 'retry-policy-persistent-storage', 'mount_path': '/data'}], 'namespace': namespace, 'load_incluster_config': False, 'kubeconfig_file': cluster_provider.kubeconfig_file, 'job_spec_config': {'backoffLimit': 5, 'parallelism': 2, 'completions': 2}, 'pod_spec_config': {'restart_policy': 'OnFailure', 'volumes': [{'name': 'retry-policy-persistent-storage', 'empty_dir': {}}]}}, name='with_restart_policy')

    @job
    def with_restart_policy_job():
        if False:
            i = 10
            return i + 15
        with_restart_policy()
    execute_result = with_restart_policy_job.execute_in_process()
    run_id = execute_result.dagster_run.run_id
    job_name = get_k8s_job_name(run_id, with_restart_policy.name)
    assert 'HI' in _get_pod_logs(cluster_provider, job_name, namespace)