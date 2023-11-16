import os
import unittest
import docker
import paramiko.ssh_exception
import pytest
from .base import TEST_API_VERSION

class SSHConnectionTest(unittest.TestCase):

    @pytest.mark.skipif('UNKNOWN_DOCKER_SSH_HOST' not in os.environ, reason='Unknown Docker SSH host not configured')
    def test_ssh_unknown_host(self):
        if False:
            print('Hello World!')
        with self.assertRaises(paramiko.ssh_exception.SSHException) as cm:
            docker.APIClient(version=TEST_API_VERSION, timeout=60, use_ssh_client=False, base_url=os.environ['UNKNOWN_DOCKER_SSH_HOST'])
        self.assertIn('not found in known_hosts', str(cm.exception))