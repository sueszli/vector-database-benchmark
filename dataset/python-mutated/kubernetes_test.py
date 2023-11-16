"""
Tests for the Kubernetes Job wrapper.

Requires:

- pykube: ``pip install pykube-ng``
- A local minikube custer up and running: http://kubernetes.io/docs/getting-started-guides/minikube/

**WARNING**: For Python versions < 3.5 the kubeconfig file must point to a Kubernetes API
hostname, and NOT to an IP address.

Written and maintained by Marco Capuccini (@mcapuccini).
"""
import unittest
import luigi
import logging
import mock
from luigi.contrib.kubernetes import KubernetesJobTask
import pytest
logger = logging.getLogger('luigi-interface')
try:
    from pykube.config import KubeConfig
    from pykube.http import HTTPClient
    from pykube.objects import Job
except ImportError:
    raise unittest.SkipTest('pykube is not installed. This test requires pykube.')

class SuccessJob(KubernetesJobTask):
    name = 'success'
    spec_schema = {'containers': [{'name': 'hello', 'image': 'alpine:3.4', 'command': ['echo', 'Hello World!']}]}

class FailJob(KubernetesJobTask):
    name = 'fail'
    max_retrials = 3
    backoff_limit = 3
    spec_schema = {'containers': [{'name': 'fail', 'image': 'alpine:3.4', 'command': ['You', 'Shall', 'Not', 'Pass']}]}

    @property
    def labels(self):
        if False:
            i = 10
            return i + 15
        return {'dummy_label': 'dummy_value'}

@pytest.mark.contrib
class TestK8STask(unittest.TestCase):

    def test_success_job(self):
        if False:
            i = 10
            return i + 15
        success = luigi.run(['SuccessJob', '--local-scheduler'])
        self.assertTrue(success)

    def test_fail_job(self):
        if False:
            while True:
                i = 10
        fail = FailJob()
        self.assertRaises(RuntimeError, fail.run)
        kube_api = HTTPClient(KubeConfig.from_file('~/.kube/config'))
        jobs = Job.objects(kube_api).filter(selector='luigi_task_id=' + fail.job_uuid)
        self.assertEqual(len(jobs.response['items']), 1)
        job = Job(kube_api, jobs.response['items'][0])
        self.assertTrue('failed' in job.obj['status'])
        self.assertTrue(job.obj['status']['failed'] > fail.max_retrials)
        self.assertTrue(job.obj['spec']['template']['metadata']['labels'] == fail.labels())

    @mock.patch.object(KubernetesJobTask, '_KubernetesJobTask__get_job_status')
    @mock.patch.object(KubernetesJobTask, 'signal_complete')
    def test_output(self, mock_signal, mock_job_status):
        if False:
            i = 10
            return i + 15
        mock_job_status.return_value = 'succeeded'
        kubernetes_job = KubernetesJobTask()
        kubernetes_job._KubernetesJobTask__logger = logger
        kubernetes_job.uu_name = 'test'
        kubernetes_job._KubernetesJobTask__track_job()
        self.assertTrue(mock_signal.called)