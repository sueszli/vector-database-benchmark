"""Executes task in a Kubernetes POD."""
from __future__ import annotations
from kubernetes.client import ApiClient, models as k8s
from airflow.exceptions import AirflowException

def _convert_kube_model_object(obj, new_class):
    if False:
        while True:
            i = 10
    convert_op = getattr(obj, 'to_k8s_client_obj', None)
    if callable(convert_op):
        return obj.to_k8s_client_obj()
    elif isinstance(obj, new_class):
        return obj
    else:
        raise AirflowException(f'Expected {new_class}, got {type(obj)}')

def _convert_from_dict(obj, new_class):
    if False:
        return 10
    if isinstance(obj, new_class):
        return obj
    elif isinstance(obj, dict):
        api_client = ApiClient()
        return api_client._ApiClient__deserialize_model(obj, new_class)
    else:
        raise AirflowException(f'Expected dict or {new_class}, got {type(obj)}')

def convert_volume(volume) -> k8s.V1Volume:
    if False:
        return 10
    '\n    Convert an airflow Volume object into a k8s.V1Volume.\n\n    :param volume:\n    '
    return _convert_kube_model_object(volume, k8s.V1Volume)

def convert_volume_mount(volume_mount) -> k8s.V1VolumeMount:
    if False:
        return 10
    '\n    Convert an airflow VolumeMount object into a k8s.V1VolumeMount.\n\n    :param volume_mount:\n    '
    return _convert_kube_model_object(volume_mount, k8s.V1VolumeMount)

def convert_port(port) -> k8s.V1ContainerPort:
    if False:
        while True:
            i = 10
    '\n    Convert an airflow Port object into a k8s.V1ContainerPort.\n\n    :param port:\n    '
    return _convert_kube_model_object(port, k8s.V1ContainerPort)

def convert_env_vars(env_vars) -> list[k8s.V1EnvVar]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a dictionary into a list of env_vars.\n\n    :param env_vars:\n    '
    if isinstance(env_vars, dict):
        res = []
        for (k, v) in env_vars.items():
            res.append(k8s.V1EnvVar(name=k, value=v))
        return res
    elif isinstance(env_vars, list):
        return env_vars
    else:
        raise AirflowException(f'Expected dict or list, got {type(env_vars)}')

def convert_pod_runtime_info_env(pod_runtime_info_envs) -> k8s.V1EnvVar:
    if False:
        print('Hello World!')
    '\n    Convert a PodRuntimeInfoEnv into an k8s.V1EnvVar.\n\n    :param pod_runtime_info_envs:\n    '
    return _convert_kube_model_object(pod_runtime_info_envs, k8s.V1EnvVar)

def convert_image_pull_secrets(image_pull_secrets) -> list[k8s.V1LocalObjectReference]:
    if False:
        i = 10
        return i + 15
    '\n    Convert a PodRuntimeInfoEnv into an k8s.V1EnvVar.\n\n    :param image_pull_secrets:\n    '
    if isinstance(image_pull_secrets, str):
        secrets = image_pull_secrets.split(',')
        return [k8s.V1LocalObjectReference(name=secret) for secret in secrets]
    else:
        return image_pull_secrets

def convert_configmap(configmaps) -> k8s.V1EnvFromSource:
    if False:
        print('Hello World!')
    '\n    Convert a str into an k8s.V1EnvFromSource.\n\n    :param configmaps:\n    '
    return k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name=configmaps))

def convert_affinity(affinity) -> k8s.V1Affinity:
    if False:
        print('Hello World!')
    'Convert a dict into an k8s.V1Affinity.'
    return _convert_from_dict(affinity, k8s.V1Affinity)

def convert_toleration(toleration) -> k8s.V1Toleration:
    if False:
        while True:
            i = 10
    'Convert a dict into an k8s.V1Toleration.'
    return _convert_from_dict(toleration, k8s.V1Toleration)