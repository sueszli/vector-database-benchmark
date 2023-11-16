import unittest
import requests
import requests_mock
from streamlit import net_util

class UtilTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        net_util._external_ip = None

    def test_get_external_ip(self):
        if False:
            for i in range(10):
                print('nop')
        with requests_mock.mock() as m:
            m.get(net_util._AWS_CHECK_IP, text='1.2.3.4')
            self.assertEqual('1.2.3.4', net_util.get_external_ip())
        net_util._external_ip = None
        with requests_mock.mock() as m:
            m.get(net_util._AWS_CHECK_IP, exc=requests.exceptions.ConnectTimeout)
            self.assertEqual(None, net_util.get_external_ip())

    def test_get_external_ip_html(self):
        if False:
            print('Hello World!')
        response_text = '\n        <html>\n            ... stuff\n        </html>\n        '
        with requests_mock.mock() as m:
            m.get(net_util._AWS_CHECK_IP, text=response_text)
            self.assertEqual(None, net_util.get_external_ip())
        net_util._external_ip = None