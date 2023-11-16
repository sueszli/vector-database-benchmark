import json
import math
import random
import time
from metaflow.tracing import inject_tracing_vars
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import KUBERNETES_SECRETS
CLIENT_REFRESH_INTERVAL_SECONDS = 300

class KubernetesJobException(MetaflowException):
    headline = 'Kubernetes job error'

def k8s_retry(deadline_seconds=60, max_backoff=32):
    if False:
        i = 10
        return i + 15

    def decorator(function):
        if False:
            print('Hello World!')
        from functools import wraps

        @wraps(function)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            from kubernetes import client
            deadline = time.time() + deadline_seconds
            retry_number = 0
            while True:
                try:
                    result = function(*args, **kwargs)
                    return result
                except client.rest.ApiException as e:
                    if e.status == 500:
                        current_t = time.time()
                        backoff_delay = min(math.pow(2, retry_number) + random.random(), max_backoff)
                        if current_t + backoff_delay < deadline:
                            time.sleep(backoff_delay)
                            retry_number += 1
                            continue
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator

class KubernetesJob(object):

    def __init__(self, client, **kwargs):
        if False:
            i = 10
            return i + 15
        self._client = client
        self._kwargs = kwargs

    def create(self):
        if False:
            return 10
        client = self._client.get()
        use_tmpfs = self._kwargs['use_tmpfs']
        tmpfs_size = self._kwargs['tmpfs_size']
        tmpfs_enabled = use_tmpfs or (tmpfs_size and (not use_tmpfs))
        self._job = client.V1Job(api_version='batch/v1', kind='Job', metadata=client.V1ObjectMeta(annotations=self._kwargs.get('annotations', {}), labels=self._kwargs.get('labels', {}), generate_name=self._kwargs['generate_name'], namespace=self._kwargs['namespace']), spec=client.V1JobSpec(backoff_limit=self._kwargs.get('retries', 0), completions=1, ttl_seconds_after_finished=7 * 60 * 60 * 24, template=client.V1PodTemplateSpec(metadata=client.V1ObjectMeta(annotations=self._kwargs.get('annotations', {}), labels=self._kwargs.get('labels', {}), namespace=self._kwargs['namespace']), spec=client.V1PodSpec(active_deadline_seconds=self._kwargs['timeout_in_seconds'], containers=[client.V1Container(command=self._kwargs['command'], env=[client.V1EnvVar(name=k, value=str(v)) for (k, v) in self._kwargs.get('environment_variables', {}).items()] + [client.V1EnvVar(name=k, value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path=str(v)))) for (k, v) in {'METAFLOW_KUBERNETES_POD_NAMESPACE': 'metadata.namespace', 'METAFLOW_KUBERNETES_POD_NAME': 'metadata.name', 'METAFLOW_KUBERNETES_POD_ID': 'metadata.uid', 'METAFLOW_KUBERNETES_SERVICE_ACCOUNT_NAME': 'spec.serviceAccountName', 'METAFLOW_KUBERNETES_NODE_IP': 'status.hostIP'}.items()] + [client.V1EnvVar(name=k, value=str(v)) for (k, v) in inject_tracing_vars({}).items()], env_from=[client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name=str(k))) for k in list(self._kwargs.get('secrets', [])) + KUBERNETES_SECRETS.split(',') if k], image=self._kwargs['image'], image_pull_policy=self._kwargs['image_pull_policy'], name=self._kwargs['step_name'].replace('_', '-'), resources=client.V1ResourceRequirements(requests={'cpu': str(self._kwargs['cpu']), 'memory': '%sM' % str(self._kwargs['memory']), 'ephemeral-storage': '%sM' % str(self._kwargs['disk'])}, limits={'%s.com/gpu'.lower() % self._kwargs['gpu_vendor']: str(self._kwargs['gpu']) for k in [0] if self._kwargs['gpu'] is not None}), volume_mounts=([client.V1VolumeMount(mount_path=self._kwargs.get('tmpfs_path'), name='tmpfs-ephemeral-volume')] if tmpfs_enabled else []) + ([client.V1VolumeMount(mount_path=path, name=claim) for (claim, path) in self._kwargs['persistent_volume_claims'].items()] if self._kwargs['persistent_volume_claims'] is not None else []))], node_selector=self._kwargs.get('node_selector'), restart_policy='Never', service_account_name=self._kwargs['service_account'], termination_grace_period_seconds=0, tolerations=[client.V1Toleration(**toleration) for toleration in self._kwargs.get('tolerations') or []], volumes=([client.V1Volume(name='tmpfs-ephemeral-volume', empty_dir=client.V1EmptyDirVolumeSource(medium='Memory', size_limit='{}Mi'.format(tmpfs_size)))] if tmpfs_enabled else []) + ([client.V1Volume(name=claim, persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=claim)) for claim in self._kwargs['persistent_volume_claims'].keys()] if self._kwargs['persistent_volume_claims'] is not None else [])))))
        return self

    def execute(self):
        if False:
            i = 10
            return i + 15
        client = self._client.get()
        try:
            response = client.BatchV1Api().create_namespaced_job(body=self._job, namespace=self._kwargs['namespace']).to_dict()
            return RunningJob(client=self._client, name=response['metadata']['name'], uid=response['metadata']['uid'], namespace=response['metadata']['namespace'])
        except client.rest.ApiException as e:
            raise KubernetesJobException('Unable to launch Kubernetes job.\n %s' % (json.loads(e.body)['message'] if e.body is not None else e.reason))

    def step_name(self, step_name):
        if False:
            while True:
                i = 10
        self._kwargs['step_name'] = step_name
        return self

    def namespace(self, namespace):
        if False:
            for i in range(10):
                print('nop')
        self._kwargs['namespace'] = namespace
        return self

    def name(self, name):
        if False:
            i = 10
            return i + 15
        self._kwargs['name'] = name
        return self

    def command(self, command):
        if False:
            return 10
        self._kwargs['command'] = command
        return self

    def image(self, image):
        if False:
            i = 10
            return i + 15
        self._kwargs['image'] = image
        return self

    def cpu(self, cpu):
        if False:
            print('Hello World!')
        self._kwargs['cpu'] = cpu
        return self

    def memory(self, mem):
        if False:
            return 10
        self._kwargs['memory'] = mem
        return self

    def environment_variable(self, name, value):
        if False:
            print('Hello World!')
        if value is None:
            return self
        self._kwargs['environment_variables'] = dict(self._kwargs.get('environment_variables', {}), **{name: value})
        return self

    def label(self, name, value):
        if False:
            return 10
        self._kwargs['labels'] = dict(self._kwargs.get('labels', {}), **{name: value})
        return self

    def annotation(self, name, value):
        if False:
            print('Hello World!')
        self._kwargs['annotations'] = dict(self._kwargs.get('annotations', {}), **{name: value})
        return self

class RunningJob(object):

    def __init__(self, client, name, uid, namespace):
        if False:
            for i in range(10):
                print('nop')
        self._client = client
        self._name = name
        self._pod_name = None
        self._id = uid
        self._namespace = namespace
        self._job = self._fetch_job()
        self._pod = self._fetch_pod()
        import atexit

        def best_effort_kill():
            if False:
                i = 10
                return i + 15
            try:
                self.kill()
            except:
                pass
        atexit.register(best_effort_kill)

    def __repr__(self):
        if False:
            print('Hello World!')
        return "{}('{}/{}')".format(self.__class__.__name__, self._namespace, self._name)

    @k8s_retry()
    def _fetch_job(self):
        if False:
            print('Hello World!')
        client = self._client.get()
        try:
            return client.BatchV1Api().read_namespaced_job(name=self._name, namespace=self._namespace).to_dict()
        except client.rest.ApiException as e:
            if e.status == 404:
                raise KubernetesJobException('Unable to locate Kubernetes batch/v1 job %s' % self._name)
            raise

    @k8s_retry()
    def _fetch_pod(self):
        if False:
            while True:
                i = 10
        client = self._client.get()
        pods = client.CoreV1Api().list_namespaced_pod(namespace=self._namespace, label_selector='job-name={}'.format(self._name)).to_dict()['items']
        if pods:
            return pods[0]
        return {}

    def kill(self):
        if False:
            return 10
        client = self._client.get()
        if not self.is_done:
            if self.is_running:
                from kubernetes.stream import stream
                api_instance = client.CoreV1Api
                try:
                    stream(api_instance().connect_get_namespaced_pod_exec, name=self._pod['metadata']['name'], namespace=self._namespace, command=['/bin/sh', '-c', '/sbin/killall5'], stderr=True, stdin=False, stdout=True, tty=False)
                except:
                    try:
                        client.BatchV1Api().patch_namespaced_job(name=self._name, namespace=self._namespace, field_manager='metaflow', body={'spec': {'parallelism': 0}})
                    except:
                        pass
            else:
                try:
                    client.BatchV1Api().patch_namespaced_job(name=self._name, namespace=self._namespace, field_manager='metaflow', body={'spec': {'parallelism': 0}})
                except:
                    pass
        return self

    @property
    def id(self):
        if False:
            print('Hello World!')
        if self._pod_name:
            return 'pod %s' % self._pod_name
        if self._pod:
            self._pod_name = self._pod['metadata']['name']
            return self.id
        return 'job %s' % self._name

    @property
    def is_done(self):
        if False:
            return 10

        def done():
            if False:
                return 10
            return bool(self._job['status'].get('succeeded')) or bool(self._job['status'].get('failed')) or self._are_pod_containers_done or (self._job['spec']['parallelism'] == 0)
        if not done():
            self._job = self._fetch_job()
            self._pod = self._fetch_pod()
        return done()

    @property
    def status(self):
        if False:
            return 10
        if not self.is_done:
            if bool(self._job['status'].get('active')):
                if self._pod:
                    msg = 'Pod is %s' % self._pod.get('status', {}).get('phase', 'uninitialized').lower()
                    container_status = (self._pod['status'].get('container_statuses') or [None])[0]
                    if container_status:
                        status = {'status': 'waiting'}
                        for (k, v) in container_status['state'].items():
                            if v is not None:
                                status['status'] = k
                                status.update(v)
                        msg += ', Container is %s' % status['status'].lower()
                        reason = ''
                        if status.get('reason'):
                            pass
                            reason = status['reason']
                        if status.get('message'):
                            reason += ' - %s' % status['message']
                        if reason:
                            msg += ' - %s' % reason
                    return msg
                return 'Job is active'
            return 'Job status is unknown'
        return 'Job is done'

    @property
    def has_succeeded(self):
        if False:
            print('Hello World!')
        return self.is_done and self._have_containers_succeeded

    @property
    def has_failed(self):
        if False:
            for i in range(10):
                print('nop')
        retval = self.is_done and (bool(self._job['status'].get('failed')) or self._has_any_container_failed or self._job['spec']['parallelism'] == 0)
        return retval

    @property
    def _have_containers_succeeded(self):
        if False:
            while True:
                i = 10
        container_statuses = self._pod.get('status', {}).get('container_statuses', [])
        if not container_statuses:
            return False
        for cstatus in container_statuses:
            terminated = cstatus.get('state', {}).get('terminated', {})
            if not terminated:
                return False
            if not terminated.get('finished_at'):
                return False
            if terminated.get('reason', '').lower() != 'completed':
                return False
        return True

    @property
    def _has_any_container_failed(self):
        if False:
            i = 10
            return i + 15
        container_statuses = self._pod.get('status', {}).get('container_statuses', [])
        if not container_statuses:
            return False
        for cstatus in container_statuses:
            terminated = cstatus.get('state', {}).get('terminated', {})
            if not terminated:
                return False
            if not terminated.get('finished_at'):
                return False
            if terminated.get('reason', '').lower() == 'error':
                return True
        return False

    @property
    def _are_pod_containers_done(self):
        if False:
            while True:
                i = 10
        container_statuses = self._pod.get('status', {}).get('container_statuses', [])
        if not container_statuses:
            return False
        for cstatus in container_statuses:
            terminated = cstatus.get('state', {}).get('terminated', {})
            if not terminated:
                return False
            if not terminated.get('finished_at'):
                return False
        return True

    @property
    def is_running(self):
        if False:
            return 10
        if self.is_done:
            return False
        return not self._are_pod_containers_done

    @property
    def is_waiting(self):
        if False:
            i = 10
            return i + 15
        return not self.is_done and (not self.is_running)

    @property
    def reason(self):
        if False:
            while True:
                i = 10
        if self.is_done:
            if self.has_succeeded:
                return (0, None)
            else:
                if self._pod.get('status', {}).get('phase') not in ('Succeeded', 'Failed'):
                    self._pod = self._fetch_pod()
                if self._pod:
                    if self._pod.get('status', {}).get('container_statuses') is None:
                        return (None, ': '.join(filter(None, [self._pod.get('status', {}).get('reason'), self._pod.get('status', {}).get('message')])))
                    for (k, v) in self._pod.get('status', {}).get('container_statuses', [{}])[0].get('state', {}).items():
                        if v is not None:
                            return (v.get('exit_code'), ': '.join(filter(None, [v.get('reason'), v.get('message')])))
        return (None, None)