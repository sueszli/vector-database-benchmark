import unittest
import warnings
from w3lib.http import basic_auth_header
from scrapy import Request
from scrapy.utils.curl import curl_to_request_kwargs

class CurlToRequestKwargsTest(unittest.TestCase):
    maxDiff = 5000

    def _test_command(self, curl_command, expected_result):
        if False:
            i = 10
            return i + 15
        result = curl_to_request_kwargs(curl_command)
        self.assertEqual(result, expected_result)
        try:
            Request(**result)
        except TypeError as e:
            self.fail(f'Request kwargs are not correct {e}')

    def test_get(self):
        if False:
            while True:
                i = 10
        curl_command = 'curl http://example.org/'
        expected_result = {'method': 'GET', 'url': 'http://example.org/'}
        self._test_command(curl_command, expected_result)

    def test_get_without_scheme(self):
        if False:
            i = 10
            return i + 15
        curl_command = 'curl www.example.org'
        expected_result = {'method': 'GET', 'url': 'http://www.example.org'}
        self._test_command(curl_command, expected_result)

    def test_get_basic_auth(self):
        if False:
            return 10
        curl_command = 'curl "https://api.test.com/" -u "some_username:some_password"'
        expected_result = {'method': 'GET', 'url': 'https://api.test.com/', 'headers': [('Authorization', basic_auth_header('some_username', 'some_password'))]}
        self._test_command(curl_command, expected_result)

    def test_get_complex(self):
        if False:
            while True:
                i = 10
        curl_command = "curl 'http://httpbin.org/get' -H 'Accept-Encoding: gzip, deflate' -H 'Accept-Language: en-US,en;q=0.9,ru;q=0.8,es;q=0.7' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/62.0.3202.75 Chrome/62.0.3202.75 Safari/537.36' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' -H 'Referer: http://httpbin.org/' -H 'Cookie: _gauges_unique_year=1; _gauges_unique=1; _gauges_unique_month=1; _gauges_unique_hour=1; _gauges_unique_day=1' -H 'Connection: keep-alive' --compressed"
        expected_result = {'method': 'GET', 'url': 'http://httpbin.org/get', 'headers': [('Accept-Encoding', 'gzip, deflate'), ('Accept-Language', 'en-US,en;q=0.9,ru;q=0.8,es;q=0.7'), ('Upgrade-Insecure-Requests', '1'), ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/62.0.3202.75 Chrome/62.0.3202.75 Safari/537.36'), ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'), ('Referer', 'http://httpbin.org/'), ('Connection', 'keep-alive')], 'cookies': {'_gauges_unique_year': '1', '_gauges_unique_hour': '1', '_gauges_unique_day': '1', '_gauges_unique': '1', '_gauges_unique_month': '1'}}
        self._test_command(curl_command, expected_result)

    def test_post(self):
        if False:
            i = 10
            return i + 15
        curl_command = "curl 'http://httpbin.org/post' -X POST -H 'Cookie: _gauges_unique_year=1; _gauges_unique=1; _gauges_unique_month=1; _gauges_unique_hour=1; _gauges_unique_day=1' -H 'Origin: http://httpbin.org' -H 'Accept-Encoding: gzip, deflate' -H 'Accept-Language: en-US,en;q=0.9,ru;q=0.8,es;q=0.7' -H 'Upgrade-Insecure-Requests: 1' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/62.0.3202.75 Chrome/62.0.3202.75 Safari/537.36' -H 'Content-Type: application/x-www-form-urlencoded' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8' -H 'Cache-Control: max-age=0' -H 'Referer: http://httpbin.org/forms/post' -H 'Connection: keep-alive' --data 'custname=John+Smith&custtel=500&custemail=jsmith%40example.org&size=small&topping=cheese&topping=onion&delivery=12%3A15&comments=' --compressed"
        expected_result = {'method': 'POST', 'url': 'http://httpbin.org/post', 'body': 'custname=John+Smith&custtel=500&custemail=jsmith%40example.org&size=small&topping=cheese&topping=onion&delivery=12%3A15&comments=', 'cookies': {'_gauges_unique_year': '1', '_gauges_unique_hour': '1', '_gauges_unique_day': '1', '_gauges_unique': '1', '_gauges_unique_month': '1'}, 'headers': [('Origin', 'http://httpbin.org'), ('Accept-Encoding', 'gzip, deflate'), ('Accept-Language', 'en-US,en;q=0.9,ru;q=0.8,es;q=0.7'), ('Upgrade-Insecure-Requests', '1'), ('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/62.0.3202.75 Chrome/62.0.3202.75 Safari/537.36'), ('Content-Type', 'application/x-www-form-urlencoded'), ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'), ('Cache-Control', 'max-age=0'), ('Referer', 'http://httpbin.org/forms/post'), ('Connection', 'keep-alive')]}
        self._test_command(curl_command, expected_result)

    def test_post_data_raw(self):
        if False:
            i = 10
            return i + 15
        curl_command = "curl 'https://www.example.org/' --data-raw 'excerptLength=200&enableDidYouMean=true&sortCriteria=ffirstz32xnamez32x201740686%20ascending&queryFunctions=%5B%5D&rankingFunctions=%5B%5D'"
        expected_result = {'method': 'POST', 'url': 'https://www.example.org/', 'body': 'excerptLength=200&enableDidYouMean=true&sortCriteria=ffirstz32xnamez32x201740686%20ascending&queryFunctions=%5B%5D&rankingFunctions=%5B%5D'}
        self._test_command(curl_command, expected_result)

    def test_post_data_raw_with_string_prefix(self):
        if False:
            print('Hello World!')
        curl_command = 'curl \'https://www.example.org/\' --data-raw $\'{"$filters":"Filter!"}\''
        expected_result = {'method': 'POST', 'url': 'https://www.example.org/', 'body': '{"$filters":"Filter!"}'}
        self._test_command(curl_command, expected_result)

    def test_explicit_get_with_data(self):
        if False:
            return 10
        curl_command = 'curl httpbin.org/anything -X GET --data asdf'
        expected_result = {'method': 'GET', 'url': 'http://httpbin.org/anything', 'body': 'asdf'}
        self._test_command(curl_command, expected_result)

    def test_patch(self):
        if False:
            print('Hello World!')
        curl_command = 'curl "https://example.com/api/fake" -u "username:password" -H "Accept: application/vnd.go.cd.v4+json" -H "Content-Type: application/json" -X PATCH -d \'{"hostname": "agent02.example.com",  "agent_config_state": "Enabled", "resources": ["Java","Linux"], "environments": ["Dev"]}\''
        expected_result = {'method': 'PATCH', 'url': 'https://example.com/api/fake', 'headers': [('Accept', 'application/vnd.go.cd.v4+json'), ('Content-Type', 'application/json'), ('Authorization', basic_auth_header('username', 'password'))], 'body': '{"hostname": "agent02.example.com",  "agent_config_state": "Enabled", "resources": ["Java","Linux"], "environments": ["Dev"]}'}
        self._test_command(curl_command, expected_result)

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        curl_command = 'curl -X "DELETE" https://www.url.com/page'
        expected_result = {'method': 'DELETE', 'url': 'https://www.url.com/page'}
        self._test_command(curl_command, expected_result)

    def test_get_silent(self):
        if False:
            return 10
        curl_command = 'curl --silent "www.example.com"'
        expected_result = {'method': 'GET', 'url': 'http://www.example.com'}
        self.assertEqual(curl_to_request_kwargs(curl_command), expected_result)

    def test_too_few_arguments_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(ValueError, 'too few arguments|the following arguments are required:\\s*url', lambda : curl_to_request_kwargs('curl'))

    def test_ignore_unknown_options(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            curl_command = 'curl --bar --baz http://www.example.com'
            expected_result = {'method': 'GET', 'url': 'http://www.example.com'}
            self.assertEqual(curl_to_request_kwargs(curl_command), expected_result)
        self.assertRaisesRegex(ValueError, 'Unrecognized options:.*--bar.*--baz', lambda : curl_to_request_kwargs('curl --bar --baz http://www.example.com', ignore_unknown_options=False))

    def test_must_start_with_curl_error(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, lambda : curl_to_request_kwargs('carl -X POST http://example.org'))