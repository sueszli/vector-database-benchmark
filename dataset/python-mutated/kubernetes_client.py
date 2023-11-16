import os
import sys
import time
from metaflow.exception import MetaflowException
from .kubernetes_job import KubernetesJob
CLIENT_REFRESH_INTERVAL_SECONDS = 300

class KubernetesClientException(MetaflowException):
    headline = 'Kubernetes client error'

class KubernetesClient(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        try:
            from kubernetes import client, config
        except (NameError, ImportError):
            raise KubernetesClientException("Could not import module 'kubernetes'.\n\nInstall Kubernetes Python package (https://pypi.org/project/kubernetes/) first.\nYou can install the module by executing - %s -m pip install kubernetes\nor equivalent through your favorite Python package manager." % sys.executable)
        self._refresh_client()

    def _refresh_client(self):
        if False:
            i = 10
            return i + 15
        from kubernetes import client, config
        if os.getenv('KUBERNETES_SERVICE_HOST'):
            config.load_incluster_config()
        else:
            config.load_kube_config()
        self._client = client
        self._client_refresh_timestamp = time.time()

    def get(self):
        if False:
            print('Hello World!')
        if time.time() - self._client_refresh_timestamp > CLIENT_REFRESH_INTERVAL_SECONDS:
            self._refresh_client()
        return self._client

    def job(self, **kwargs):
        if False:
            return 10
        return KubernetesJob(self, **kwargs)