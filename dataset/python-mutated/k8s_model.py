"""Classes for interacting with Kubernetes API."""
from __future__ import annotations
from abc import ABC, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kubernetes.client import models as k8s

class K8SModel(ABC):
    """
    Airflow Kubernetes models are here for backwards compatibility reasons only.

    Ideally clients should use the kubernetes API
    and the process of

        client input -> Airflow k8s models -> k8s models

    can be avoided. All of these models implement the
    `attach_to_pod` method so that they integrate with the kubernetes client.
    """

    @abstractmethod
    def attach_to_pod(self, pod: k8s.V1Pod) -> k8s.V1Pod:
        if False:
            print('Hello World!')
        '\n        Attaches to pod.\n\n        :param pod: A pod to attach this Kubernetes object to\n        :return: The pod with the object attached\n        '

def append_to_pod(pod: k8s.V1Pod, k8s_objects: list[K8SModel] | None):
    if False:
        print('Hello World!')
    '\n    Attach additional specs to an existing pod object.\n\n    :param pod: A pod to attach a list of Kubernetes objects to\n    :param k8s_objects: a potential None list of K8SModels\n    :return: pod with the objects attached if they exist\n    '
    if not k8s_objects:
        return pod
    return reduce(lambda p, o: o.attach_to_pod(p), k8s_objects, pod)