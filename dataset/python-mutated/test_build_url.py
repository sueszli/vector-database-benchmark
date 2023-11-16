import os
import tempfile
import textwrap
import unittest
from pocsuite3.api import get_results, init_pocsuite, start_pocsuite

class CustomNamedTemporaryFile:
    """
    This custom implementation is needed because of the following limitation of tempfile.NamedTemporaryFile:

    > Whether the name can be used to open the file a second time, while the named temporary file is still open,
    > varies across platforms (it can be so used on Unix; it cannot on Windows NT or later).
    """

    def __init__(self, mode='wb', delete=True):
        if False:
            i = 10
            return i + 15
        self._mode = mode
        self._delete = delete

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        file_name = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        open(file_name, 'x').close()
        self._tempFile = open(file_name, self._mode)
        return self._tempFile

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        self._tempFile.close()
        if self._delete:
            os.remove(self._tempFile.name)

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_target_url_format(self):
        if False:
            return 10
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent("                    from pocsuite3.api import POCBase, register_poc\n\n\n                    class TestPoC(POCBase):\n                        def _verify(self):\n                            result = {}\n                            result['VerifyInfo'] = {}\n                            result['VerifyInfo']['url'] = self.url\n                            result['VerifyInfo']['scheme'] = self.scheme\n                            result['VerifyInfo']['rhost'] = self.rhost\n                            result['VerifyInfo']['rport'] = self.rport\n                            result['VerifyInfo']['netloc'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ")
            f.write(poc_content)
            f.seek(0)
            config = {'url': 'http://127.0.0.1:8080', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'http://127.0.0.1:8080')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'http')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8080)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8080')
            f.seek(0)
            config = {'url': 'https://127.0.0.1:8080', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'https://127.0.0.1:8080')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'https')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8080)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8080')
            f.seek(0)
            config = {'url': '127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'http://127.0.0.1:80')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'http')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 80)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:80')
            f.seek(0)
            config = {'url': '127.0.0.1:8443', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'https://127.0.0.1:8443')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'https')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8443)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8443')
            f.seek(0)
            config = {'url': '[fd12:3456:789a:1::2]:8443', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'https://[fd12:3456:789a:1::2]:8443')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'https')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], 'fd12:3456:789a:1::2')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8443)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '[fd12:3456:789a:1::2]:8443')

    def test_url_protocol_correct(self):
        if False:
            while True:
                i = 10
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent("                    from pocsuite3.api import POCBase, register_poc, POC_CATEGORY\n\n\n                    class TestPoC(POCBase):\n                        protocol = POC_CATEGORY.PROTOCOL.FTP\n\n                        def _verify(self):\n                            result = {}\n                            result['VerifyInfo'] = {}\n                            result['VerifyInfo']['url'] = self.url\n                            result['VerifyInfo']['scheme'] = self.scheme\n                            result['VerifyInfo']['rhost'] = self.rhost\n                            result['VerifyInfo']['rport'] = self.rport\n                            result['VerifyInfo']['netloc'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ")
            f.write(poc_content)
            print(f.name)
            f.seek(0)
            config = {'url': 'https://127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:21')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 21)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:21')
            f.seek(0)
            config = {'url': '127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:21')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 21)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:21')
            f.seek(0)
            config = {'url': '127.0.0.1:8821', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:8821')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8821)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8821')
            f.seek(0)
            config = {'url': 'ftp://127.0.0.1:8821', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:8821')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8821)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8821')

    def test_set_protocol_and_default_port(self):
        if False:
            for i in range(10):
                print('nop')
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent("                    from pocsuite3.api import POCBase, register_poc, POC_CATEGORY\n\n\n                    class TestPoC(POCBase):\n                        protocol = POC_CATEGORY.PROTOCOL.FTP\n                        protocol_default_port = 10086\n\n                        def _verify(self):\n                            result = {}\n                            result['VerifyInfo'] = {}\n                            result['VerifyInfo']['url'] = self.url\n                            result['VerifyInfo']['scheme'] = self.scheme\n                            result['VerifyInfo']['rhost'] = self.rhost\n                            result['VerifyInfo']['rport'] = self.rport\n                            result['VerifyInfo']['netloc'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ")
            f.write(poc_content)
            f.seek(0)
            config = {'url': 'https://127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:10086')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 10086)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:10086')
            f.seek(0)
            config = {'url': 'https://127.0.0.1:21', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'ftp://127.0.0.1:21')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'ftp')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 21)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:21')

    def test_custom_protocol_and_default_port(self):
        if False:
            for i in range(10):
                print('nop')
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent('                    from pocsuite3.api import POCBase, register_poc, POC_CATEGORY\n\n\n                    class TestPoC(POCBase):\n                        protocol = "CUSTOM"\n                        protocol_default_port = 10086\n\n                        def _verify(self):\n                            result = {}\n                            result[\'VerifyInfo\'] = {}\n                            result[\'VerifyInfo\'][\'url\'] = self.url\n                            result[\'VerifyInfo\'][\'scheme\'] = self.scheme\n                            result[\'VerifyInfo\'][\'rhost\'] = self.rhost\n                            result[\'VerifyInfo\'][\'rport\'] = self.rport\n                            result[\'VerifyInfo\'][\'netloc\'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ')
            f.write(poc_content)
            f.seek(0)
            config = {'url': 'https://127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'custom://127.0.0.1:10086')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'custom')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 10086)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:10086')
            f.seek(0)
            config = {'url': 'https://127.0.0.1:8080', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'custom://127.0.0.1:8080')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'custom')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 8080)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:8080')

    def test_custom_protocol(self):
        if False:
            for i in range(10):
                print('nop')
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent('                    from pocsuite3.api import POCBase, register_poc, POC_CATEGORY\n\n\n                    class TestPoC(POCBase):\n                        protocol = "CUSTOM"\n\n                        def _verify(self):\n                            result = {}\n                            result[\'VerifyInfo\'] = {}\n                            result[\'VerifyInfo\'][\'url\'] = self.url\n                            result[\'VerifyInfo\'][\'scheme\'] = self.scheme\n                            result[\'VerifyInfo\'][\'rhost\'] = self.rhost\n                            result[\'VerifyInfo\'][\'rport\'] = self.rport\n                            result[\'VerifyInfo\'][\'netloc\'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ')
            f.write(poc_content)
            f.seek(0)
            config = {'url': '127.0.0.1:443', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'https://127.0.0.1:443')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'https')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 443)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:443')

    def test_custom_default_port(self):
        if False:
            print('Hello World!')
        with CustomNamedTemporaryFile('w+t') as f:
            poc_content = textwrap.dedent("                    from pocsuite3.api import POCBase, register_poc, POC_CATEGORY\n\n\n                    class TestPoC(POCBase):\n                        protocol_default_port = 10443\n\n                        def _verify(self):\n                            result = {}\n                            result['VerifyInfo'] = {}\n                            result['VerifyInfo']['url'] = self.url\n                            result['VerifyInfo']['scheme'] = self.scheme\n                            result['VerifyInfo']['rhost'] = self.rhost\n                            result['VerifyInfo']['rport'] = self.rport\n                            result['VerifyInfo']['netloc'] = self.netloc\n                            return self.parse_output(result)\n\n\n                    register_poc(TestPoC)\n            ")
            f.write(poc_content)
            f.seek(0)
            config = {'url': '127.0.0.1', 'poc': f.name}
            init_pocsuite(config)
            start_pocsuite()
            res = get_results()
            self.assertEqual(res[0]['result']['VerifyInfo']['url'], 'https://127.0.0.1:10443')
            self.assertEqual(res[0]['result']['VerifyInfo']['scheme'], 'https')
            self.assertEqual(res[0]['result']['VerifyInfo']['rhost'], '127.0.0.1')
            self.assertEqual(res[0]['result']['VerifyInfo']['rport'], 10443)
            self.assertEqual(res[0]['result']['VerifyInfo']['netloc'], '127.0.0.1:10443')