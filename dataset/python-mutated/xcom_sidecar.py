"""Attach a sidecar container that blocks the pod from completing until Airflow pulls result data."""
from __future__ import annotations
import copy
from kubernetes.client import models as k8s

class PodDefaults:
    """Static defaults for Pods."""
    XCOM_MOUNT_PATH = '/airflow/xcom'
    SIDECAR_CONTAINER_NAME = 'airflow-xcom-sidecar'
    XCOM_CMD = 'trap "exit 0" INT; while true; do sleep 1; done;'
    VOLUME_MOUNT = k8s.V1VolumeMount(name='xcom', mount_path=XCOM_MOUNT_PATH)
    VOLUME = k8s.V1Volume(name='xcom', empty_dir=k8s.V1EmptyDirVolumeSource())
    SIDECAR_CONTAINER = k8s.V1Container(name=SIDECAR_CONTAINER_NAME, command=['sh', '-c', XCOM_CMD], image='alpine', volume_mounts=[VOLUME_MOUNT], resources=k8s.V1ResourceRequirements(requests={'cpu': '1m', 'memory': '10Mi'}))

def add_xcom_sidecar(pod: k8s.V1Pod, *, sidecar_container_image: str | None=None, sidecar_container_resources: k8s.V1ResourceRequirements | dict | None=None) -> k8s.V1Pod:
    if False:
        print('Hello World!')
    'Add sidecar.'
    pod_cp = copy.deepcopy(pod)
    pod_cp.spec.volumes = pod.spec.volumes or []
    pod_cp.spec.volumes.insert(0, PodDefaults.VOLUME)
    pod_cp.spec.containers[0].volume_mounts = pod_cp.spec.containers[0].volume_mounts or []
    pod_cp.spec.containers[0].volume_mounts.insert(0, PodDefaults.VOLUME_MOUNT)
    sidecar = copy.deepcopy(PodDefaults.SIDECAR_CONTAINER)
    sidecar.image = sidecar_container_image or PodDefaults.SIDECAR_CONTAINER.image
    if sidecar_container_resources:
        sidecar.resources = sidecar_container_resources
    pod_cp.spec.containers.append(sidecar)
    return pod_cp