import os, sys, re
import ssl
import public
import json
import socket
import requests
import config
import requests.packages.urllib3.util.connection as urllib3_conn
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class http:
    _ip_type = None

    def __init__(self):
        if False:
            return 10
        self._ip_type = config.config().get_request_iptype()

    def get(self, url, timeout=60, headers={}, verify=False, type='python'):
        if False:
            i = 10
            return i + 15
        url = self.quote(url)
        if type in ['python', 'src', 'php']:
            old_family = urllib3_conn.allowed_gai_family
            try:
                if self._ip_type == 'ipv4':
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET
                elif self._ip_type == 'ipv6':
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET6
                result = requests.get(url, timeout=timeout, headers=get_headers(headers), verify=verify)
            except:
                try:
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET6
                    result = requests.get(url, timeout=timeout, headers=get_headers(headers), verify=verify)
                except:
                    result = self._get_curl(url, timeout, headers, verify)
            urllib3_conn.allowed_gai_family = old_family
        elif type == 'curl':
            result = self._get_curl(url, timeout, headers, verify)
        elif type == 'php':
            result = self._get_php(url, timeout, headers, verify)
        elif type == 'src':
            if sys.version_info[0] == 2:
                result = self._get_py2(url, timeout, headers, verify)
            else:
                result = self._get_py3(url, timeout, headers, verify)
        return result

    def post(self, url, data, timeout=60, headers={}, verify=False, type='python'):
        if False:
            print('Hello World!')
        url = self.quote(url)
        if type in ['python', 'src', 'php']:
            old_family = urllib3_conn.allowed_gai_family
            try:
                if self._ip_type == 'ipv4':
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET
                elif self._ip_type == 'ipv6':
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET6
                result = requests.post(url, data, timeout=timeout, headers=headers, verify=verify)
            except:
                try:
                    urllib3_conn.allowed_gai_family = lambda : socket.AF_INET6
                    result = requests.post(url, data, timeout=timeout, headers=headers, verify=verify)
                except:
                    result = self._post_curl(url, data, timeout, headers, verify)
            urllib3_conn.allowed_gai_family = old_family
        elif type == 'curl':
            result = self._post_curl(url, data, timeout, headers, verify)
        elif type == 'php':
            result = self._post_php(url, data, timeout, headers, verify)
        elif type == 'src':
            if sys.version_info[0] == 2:
                result = self._post_py2(url, data, timeout, headers, verify)
            else:
                result = self._post_py3(url, data, timeout, headers, verify)
        return result

    def download_file(self, url, filename, data=None, timeout=1800, speed_file='/dev/shm/download_speed.pl'):
        if False:
            return 10
        '\n            @name 下载文件\n            @author hwliang<2021-07-08>\n            @param url<string> 下载地址\n            @param filename<string> 保存路径\n            @param data<dict> POST参数，不传则使用GET方法，否则使用POST方法\n            @param timeout<int> 超时时间,默认1800秒\n            @param speed_file<string> \n        '
        import requests
        from requests.packages.urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
        headers = public.get_requests_headers()
        if data is None:
            res = requests.get(url, headers=headers, timeout=timeout, stream=True)
        else:
            res = requests.post(url, data, headers=headers, timeout=timeout, stream=True)
        with open(filename, 'wb') as f:
            for _chunk in res.iter_content(chunk_size=8192):
                f.write(_chunk)

    def _post_py2(self, url, data, timeout, headers, verify):
        if False:
            print('Hello World!')
        import urllib2
        req = urllib2.Request(url, self._str_py_post(data, headers), headers=headers)
        try:
            if not verify:
                context = ssl._create_unverified_context()
                r_response = urllib2.urlopen(req, timeout=timeout, context=context)
            else:
                r_response = urllib2.urlopen(req, timeout=timeout)
        except urllib2.HTTPError as err:
            return response(str(err), err.code, [])
        except urllib2.URLError as err:
            return response(str(err), 0, [])
        return response(r_response.read(), r_response.getcode(), r_response.info().headers)

    def _post_py3(self, url, data, timeout, headers, verify):
        if False:
            for i in range(10):
                print('nop')
        import urllib.request
        req = urllib.request.Request(url, self._str_py_post(data, headers), headers=headers)
        try:
            if not verify:
                context = ssl._create_unverified_context()
                r_response = urllib.request.urlopen(req, timeout=timeout, context=context)
            else:
                r_response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as err:
            return response(str(err), err.code, [])
        except urllib.error.URLError as err:
            return response(str(err), 0, [])
        r_body = r_response.read()
        if type(r_body) == bytes:
            r_body = r_body.decode('utf-8')
        return response(r_body, r_response.getcode(), r_response.getheaders())

    def _post_curl(self, url, data, timeout, headers, verify):
        if False:
            print('Hello World!')
        headers_str = self._str_headers(headers)
        pdata = self._str_post(data, headers_str)
        _ssl_verify = ''
        if not verify:
            _ssl_verify = ' -k'
        result = public.ExecShell("{} -X POST -sS -i --connect-timeout {} {} {} '{}' 2>&1".format(self._curl_bin() + _ssl_verify, timeout, headers_str, pdata, url))[0]
        (r_body, r_headers, r_status_code) = self._curl_format(result)
        return response(r_body, r_status_code, r_headers)

    def _post_php(self, url, data, timeout, headers, verify):
        if False:
            i = 10
            return i + 15
        php_version = self._get_php_version()
        if not php_version:
            raise Exception('没有可用的PHP版本!')
        ip_type = ''
        if self._ip_type == 'ipv6':
            ip_type = 'curl_setopt($ch, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V6);'
        elif self._ip_type == 'ipv4':
            ip_type = 'curl_setopt($ch, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);'
        tmp_file = '/dev/shm/http.php'
        http_php = '<?php\nerror_reporting(E_ERROR);\nif(isset($_POST[\'data\'])){{\n    $data = json_decode($_POST[\'data\'],1);\n}}else{{\n    $s = getopt(\'\',array(\'post:\'));\n    $data = json_decode($s[\'post\'],1);\n}}\n$url  = $data[\'url\'];\n$ch = curl_init();\ncurl_setopt($ch, CURLOPT_URL, $url);\ncurl_setopt($ch, CURLOPT_HTTPHEADER,$data[\'headers\']);\ncurl_setopt($ch, CURLOPT_HEADER, true);\ncurl_setopt($ch, CURLINFO_HEADER_OUT, true);\ncurl_setopt($ch, CURLOPT_POST, true);\ncurl_setopt($ch, CURLOPT_POSTFIELDS, $data[\'data\']);\ncurl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);\ncurl_setopt($ch, CURLOPT_SSL_VERIFYPEER, $data[\'verify\']);\ncurl_setopt($ch, CURLOPT_SSL_VERIFYHOST, $data[\'verify\']);\ncurl_setopt($ch, CURLOPT_CONNECTTIMEOUT, $data[\'timeout\']);\ncurl_setopt($ch, CURLOPT_TIMEOUT, $data[\'timeout\']);\n{ip_type}\n$result = curl_exec($ch);\n$h_size = curl_getinfo($ch, CURLINFO_HEADER_SIZE);\n$header = substr($result, 0, $h_size);\n$body = substr($result,$h_size,strlen($result));\ncurl_close($ch);\nexit($header."\r\n\r\n".json_encode($body));\n?>'.format(ip_type=ip_type)
        public.writeFile(tmp_file, http_php)
        data = json.dumps({'url': url, 'timeout': timeout, 'verify': verify, 'headers': self._php_headers(headers), 'data': data})
        if php_version in ['53']:
            php_version = '/www/server/php/' + php_version + '/bin/php'
        if php_version.find('/www/server/php') != -1:
            result = public.ExecShell(php_version + ' ' + tmp_file + " --post='" + data + "'")[0]
        else:
            result = public.request_php(php_version, '/http.php', '/dev/shm', 'POST', {'data': data})
            if isinstance(result, bytes):
                result = result.decode('utf-8')
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        (r_body, r_headers, r_status_code) = self._curl_format(result)
        return response(json.loads(r_body), r_status_code, r_headers)

    def _get_py2(self, url, timeout, headers, verify):
        if False:
            for i in range(10):
                print('nop')
        import urllib2
        req = urllib2.Request(url, headers=headers)
        try:
            if not verify:
                context = ssl._create_unverified_context()
                r_response = urllib2.urlopen(req, timeout=timeout, context=context)
            else:
                r_response = urllib2.urlopen(req, timeout=timeout)
        except urllib2.HTTPError as err:
            return response(str(err), err.code, [])
        except urllib2.URLError as err:
            return response(str(err), 0, [])
        return response(r_response.read(), r_response.getcode(), r_response.info().headers)

    def quote(self, url):
        if False:
            return 10
        if url.find('[') == -1:
            return url
        url_tmp = url.split('?')
        if len(url_tmp) == 1:
            return url
        url_last = url_tmp[0]
        url_args = '?'.join(url_tmp[1:])
        if sys.version_info[0] == 2:
            import urllib2
            url_args = urllib2.quote(url_args)
        else:
            import urllib.parse
            url_args = urllib.parse.quote(url_args)
        return url_last + '?' + url_args

    def _get_py3(self, url, timeout, headers, verify):
        if False:
            return 10
        import urllib.request
        req = urllib.request.Request(url, headers=headers)
        try:
            if not verify:
                context = ssl._create_unverified_context()
                r_response = urllib.request.urlopen(req, timeout=timeout, context=context)
            else:
                r_response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as err:
            return response(str(err), err.code, [])
        except urllib.error.URLError as err:
            return response(str(err), 0, [])
        r_body = r_response.read()
        if type(r_body) == bytes:
            r_body = r_body.decode('utf-8')
        return response(r_body, r_response.getcode(), r_response.getheaders())

    def _get_curl(self, url, timeout, headers, verify):
        if False:
            i = 10
            return i + 15
        headers_str = self._str_headers(headers)
        _ssl_verify = ''
        if not verify:
            _ssl_verify = ' -k'
        result = public.ExecShell('{} -sS -i --connect-timeout {} {} {} 2>&1'.format(self._curl_bin() + ' ' + str(_ssl_verify), timeout, headers_str, url))[0]
        (r_body, r_headers, r_status_code) = self._curl_format(result)
        return response(r_body, r_status_code, r_headers)

    def _get_php(self, url, timeout, headers, verify):
        if False:
            for i in range(10):
                print('nop')
        php_version = self._get_php_version()
        if not php_version:
            raise Exception('没有可用的PHP版本!')
        ip_type = ''
        if self._ip_type == 'ipv6':
            ip_type = 'curl_setopt($ch, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V6);'
        elif self._ip_type == 'ipv4':
            ip_type = 'curl_setopt($ch, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);'
        tmp_file = '/dev/shm/http.php'
        http_php = '<?php\nerror_reporting(E_ERROR);\nif(isset($_POST[\'data\'])){{\n    $data = json_decode($_POST[\'data\'],1);\n}}else{{\n    $s = getopt(\'\',array(\'post:\'));\n    $data = json_decode($s[\'post\'],1);\n}}\n$url  = $data[\'url\'];\n$ch = curl_init();\n$user_agent = "BT-Panel";\ncurl_setopt($ch, CURLOPT_URL, $url);\ncurl_setopt($ch, CURLOPT_HTTPHEADER,$data[\'headers\']);\ncurl_setopt($ch, CURLOPT_HEADER, true);\ncurl_setopt($ch, CURLINFO_HEADER_OUT, TRUE);\ncurl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);\ncurl_setopt($ch, CURLOPT_SSL_VERIFYPEER, $data[\'verify\']);\ncurl_setopt($ch, CURLOPT_SSL_VERIFYHOST, $data[\'verify\']);\ncurl_setopt($ch, CURLOPT_CONNECTTIMEOUT, $data[\'timeout\']);\ncurl_setopt($ch, CURLOPT_TIMEOUT, $data[\'timeout\']);\n{ip_type}\ncurl_setopt($ch, CURLOPT_POST, false);\n$result = curl_exec($ch);\n$h_size = curl_getinfo($ch, CURLINFO_HEADER_SIZE);\n$header = substr($result, 0, $h_size);\n$body = substr($result,$h_size,strlen($result));\ncurl_close($ch);\nexit($header."\r\n\r\n".json_encode($body));\n?>'.format(ip_type=ip_type)
        public.writeFile(tmp_file, http_php)
        data = json.dumps({'url': url, 'timeout': timeout, 'verify': verify, 'headers': self._php_headers(headers)})
        if php_version in ['53']:
            php_version = '/www/server/php/' + php_version + '/bin/php'
        if php_version.find('/www/server/php') != -1:
            result = public.ExecShell(php_version + ' ' + tmp_file + " --post='" + data + "'")[0]
        else:
            result = public.request_php(php_version, '/http.php', '/dev/shm', 'POST', {'data': data})
            if isinstance(result, bytes):
                result = result.decode('utf-8')
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        (r_body, r_headers, r_status_code) = self._curl_format(result)
        return response(json.loads(r_body).strip(), r_status_code, r_headers)

    def _get_php_version(self):
        if False:
            while True:
                i = 10
        php_versions = public.get_php_versions()
        php_versions = sorted(php_versions, reverse=True)
        php_path = '/www/server/php/{}/sbin/php-fpm'
        php_sock = '/tmp/php-cgi-{}.sock'
        for pv in php_versions:
            if not os.path.exists(php_path.format(pv)):
                continue
            if not os.path.exists(php_sock.format(pv)):
                continue
            return pv
        php_bin = '/www/server/php/{}/bin/php'
        for pv in php_versions:
            pb = php_bin.format(pv)
            if not os.path.exists(pb):
                continue
            return pb
        return None

    def _curl_bin(self):
        if False:
            while True:
                i = 10
        c_bin = ['/usr/local/curl2/bin/curl', '/usr/local/curl/bin/curl', '/usr/local/bin/curl', '/usr/bin/curl']
        curl_bin = 'curl'
        for cb in c_bin:
            if os.path.exists(cb):
                curl_bin = cb
        if self._ip_type != 'auto':
            v4_file = '{}/data/v4.pl'.format(public.get_panel_path())
            curl_bin += ' {}'.format(public.readFile(v4_file).strip())
        return curl_bin

    def _curl_format(self, req):
        if False:
            return 10
        match = re.search('(.|\n)+\r\n\r\n', req)
        if not match:
            return (req, {}, 0)
        tmp = match.group().split('\r\n')
        i = 0
        if tmp[i].find('Continue') != -1:
            i += 1
        if not tmp[i]:
            i += 1
        try:
            status_code = int(tmp[i].split()[1])
        except:
            status_code = 0
        body = req.replace(match.group(), '')
        return (body, tmp, status_code)

    def _php_headers(self, headers):
        if False:
            i = 10
            return i + 15
        php_headers = []
        for h in headers.keys():
            php_headers.append('{}: {}'.format(h, headers[h]))
        return php_headers

    def _str_headers(self, headers):
        if False:
            for i in range(10):
                print('nop')
        str_headers = ''
        for key in headers.keys():
            str_headers += " -H '{}: {}'".format(key, headers[key])
        return str_headers

    def _str_post(self, pdata, headers):
        if False:
            for i in range(10):
                print('nop')
        str_pdata = ''
        if headers.find('application/jose') != -1 or headers.find('application/josn') != -1:
            if type(pdata) == dict:
                pdata = json.dumps(pdata)
            if type(pdata) == bytes:
                pdata = pdata.decode('utf-8')
            str_pdata += " -d '{}'".format(pdata)
            return str_pdata
        for key in pdata.keys():
            str_pdata += " -F '{}={}'".format(key, pdata[key])
        return str_pdata

    def _str_py_post(self, pdata, headers):
        if False:
            return 10
        if 'Content-Type' in headers:
            if headers['Content-Type'].find('application/jose') != -1 or headers['Content-Type'].find('application/josn') != -1:
                if type(pdata) == dict:
                    pdata = json.dumps(pdata)
                if type(pdata) == str:
                    pdata = pdata.encode('utf-8')
                return pdata
        return public.url_encode(pdata)

class http_headers:

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        return getattr(self, key.lower(), None)

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        setattr(self, key.lower(), value)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return getattr(self, key.lower(), None)

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        delattr(self, key.lower())

    def __delattr__(self, key):
        if False:
            while True:
                i = 10
        delattr(self, key.lower())

    def get(self, key):
        if False:
            while True:
                i = 10
        return getattr(self, key.lower(), None)

    def get_items(self):
        if False:
            while True:
                i = 10
        return self

class response:
    status_code = None
    status = None
    code = None
    headers = {}
    text = None
    content = None

    def __init__(self, body, status_code, headers):
        if False:
            print('Hello World!')
        self.text = body
        self.content = body
        self.status_code = status_code
        self.status = status_code
        self.code = status_code
        self.headers = http_headers()
        self.format_headers(headers)

    def format_headers(self, raw_headers):
        if False:
            i = 10
            return i + 15
        raw = []
        for h in raw_headers:
            if not h:
                continue
            if type(h) == tuple:
                raw.append(h[0] + ': ' + h[1])
                if len(h) < 2:
                    continue
                self.headers[h[0]] = h[1].strip()
            else:
                raw.append(h.strip())
                tmp = h.split(': ')
                if len(tmp) < 2:
                    continue
                self.headers[tmp[0]] = tmp[1].strip()
        self.headers.raw = '\r\n'.join(raw)

    def close(self):
        if False:
            while True:
                i = 10
        self.text = None
        self.content = None
        self.status_code = None
        self.status = None
        self.code = None
        self.headers = None

    def json(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return json.loads(self.text)
        except:
            return self.text
DEFAULT_HEADERS = {'Content-type': 'application/x-www-form-urlencoded', 'User-Agent': 'BT-Panel'}
s_types = ['python', 'php', 'curl', 'src']
DEFAULT_TYPE = 'python'
__version__ = 1.0

def get_stype(s_type):
    if False:
        i = 10
        return i + 15
    if not s_type:
        s_type_file = '/www/server/panel/data/http_type.pl'
        if os.path.exists(s_type_file):
            tmp_type = public.readFile(s_type_file)
            if tmp_type:
                tmp_type = tmp_type.strip().lower()
                if tmp_type in s_types:
                    s_type = tmp_type
    else:
        s_type = s_type.lower()
        if not s_type in s_types:
            s_type = DEFAULT_TYPE
    if not s_type:
        s_type = DEFAULT_TYPE
    return s_type

def get_headers(headers):
    if False:
        print('Hello World!')
    if type(headers) != dict:
        headers = {}
    if not 'User-Agent' in headers:
        headers['User-Agent'] = DEFAULT_HEADERS['User-Agent']
    return headers

def post(url, data={}, timeout=60, headers={}, verify=False, s_type=None):
    if False:
        return 10
    '\n        POST请求\n        @param [url] string URL地址\n        @parma [data] dict POST参数\n        @param [timeout] int 超时时间 默认60秒\n        @param [headers] dict 请求头 默认{"Content-type":"application/x-www-form-urlencoded","User-Agent":"BT-Panel"}\n        @param [verify] bool 是否验证ssl证书 默认False\n        @param [s_type] string 请求方法 默认python 可选：curl或php\n    '
    p = http()
    try:
        return p.post(url, data, timeout, get_headers(headers), verify, get_stype(s_type))
    except:
        raise Exception(public.get_error_info())

def get(url, timeout=60, headers={}, verify=False, s_type=None):
    if False:
        for i in range(10):
            print('nop')
    '\n        GET请求\n        @param [url] string URL地址\n        @param [timeout] int 超时时间 默认60秒\n        @param [headers] dict 请求头 默认{"Content-type":"application/x-www-form-urlencoded","User-Agent":"BT-Panel"}\n        @param [verify] bool 是否验证ssl证书 默认False\n        @param [s_type] string 请求方法 默认python 可选：curl或php\n    '
    p = http()
    try:
        return p.get(url, timeout, get_headers(headers), verify, get_stype(s_type))
    except:
        raise Exception(public.get_error_info())