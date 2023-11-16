"""
Tests for OpenPAI wrapper for Luigi.


Written and maintained by Liu, Dongqing (@liudongqing).
"""
from helpers import unittest
import responses
import time
import luigi
import logging
from luigi.contrib.pai import PaiTask
from luigi.contrib.pai import TaskRole
logging.basicConfig(level=logging.DEBUG)
'\nThe following configurations are required to run the test\n[OpenPai]\npai_url:http://host:port/\nusername:admin\npassword:admin-password\nexpiration:3600\n\n'

class SklearnJob(PaiTask):
    image = 'openpai/pai.example.sklearn'
    name = 'test_job_sk_{0}'.format(time.time())
    command = 'cd scikit-learn/benchmarks && python bench_mnist.py'
    virtual_cluster = 'spark'
    tasks = [TaskRole('test', 'cd scikit-learn/benchmarks && python bench_mnist.py', memoryMB=4096)]

class TestPaiTask(unittest.TestCase):

    @responses.activate
    def test_success(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Here using the responses lib to mock the PAI rest api call, the following specify the response of the call.\n        '
        responses.add(responses.POST, 'http://127.0.0.1:9186/api/v1/token', json={'token': 'test', 'user': 'admin', 'admin': True}, status=200)
        sk_task = SklearnJob()
        responses.add(responses.POST, 'http://127.0.0.1:9186/api/v1/jobs', json={'message': 'update job {0} successfully'.format(sk_task.name)}, status=202)
        responses.add(responses.GET, 'http://127.0.0.1:9186/api/v1/jobs/{0}'.format(sk_task.name), json={}, status=404)
        responses.add(responses.GET, 'http://127.0.0.1:9186/api/v1/jobs/{0}'.format(sk_task.name), body='{"jobStatus": {"state":"SUCCEED"}}', status=200)
        success = luigi.build([sk_task], local_scheduler=True)
        self.assertTrue(success)
        self.assertTrue(sk_task.complete())

    @responses.activate
    def test_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        Here using the responses lib to mock the PAI rest api call, the following specify the response of the call.\n        '
        responses.add(responses.POST, 'http://127.0.0.1:9186/api/v1/token', json={'token': 'test', 'user': 'admin', 'admin': True}, status=200)
        fail_task = SklearnJob()
        responses.add(responses.POST, 'http://127.0.0.1:9186/api/v1/jobs', json={'message': 'update job {0} successfully'.format(fail_task.name)}, status=202)
        responses.add(responses.GET, 'http://127.0.0.1:9186/api/v1/jobs/{0}'.format(fail_task.name), json={}, status=404)
        responses.add(responses.GET, 'http://127.0.0.1:9186/api/v1/jobs/{0}'.format(fail_task.name), body='{"jobStatus": {"state":"FAILED"}}', status=200)
        success = luigi.build([fail_task], local_scheduler=True)
        self.assertFalse(success)
        self.assertFalse(fail_task.complete())