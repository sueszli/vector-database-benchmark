import unittest
import pytest
from helpers import with_config
from luigi.contrib.hdfs import WebHdfsClient
InsecureClient = pytest.importorskip('hdfs.InsecureClient')
KerberosClient = pytest.importorskip('hdfs.ext.kerberos.KerberosClient')

@pytest.mark.apache
class TestWebHdfsClient(unittest.TestCase):

    @with_config({'webhdfs': {'client_type': 'insecure'}})
    def test_insecure_client_type(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebHdfsClient(host='localhost').client
        self.assertIsInstance(client, InsecureClient)

    @with_config({'webhdfs': {'client_type': 'kerberos'}})
    def test_kerberos_client_type(self):
        if False:
            print('Hello World!')
        client = WebHdfsClient(host='localhost').client
        self.assertIsInstance(client, KerberosClient)