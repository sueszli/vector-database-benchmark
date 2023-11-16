import json, os, public, time, re, sys
import time
import fastcgiClient as fcgi_client
import struct
FCGI_Header = '!BBHHBx'
if sys.version_info[0] == 2:
    try:
        from cStringIO import StringIO
    except:
        from StringIO import StringIO
else:
    from io import BytesIO as StringIO

class panelPHP:
    re_io = None

    def __init__(self, plugin_name=None):
        if False:
            for i in range(10):
                print('nop')
        if plugin_name:
            self.__plugin_name = plugin_name
            self.__plugin_path = '/www/server/panel/plugin/%s' % plugin_name
            self.__args_dir = self.__plugin_path + '/args'
            self.__args_tmp = self.__args_dir + '/' + public.GetRandomString(32)
            if not os.path.exists(self.__args_dir):
                os.makedirs(self.__args_dir, 384)

    def exec_php_script(self, args):
        if False:
            return 10
        php_bin = self.__get_php_bin()
        if not php_bin:
            return public.returnMsg(False, '没有找到兼容的PHP版本，请先安装')
        self.__write_args(args)
        result = os.popen('cd ' + self.__plugin_path + ' && %s /www/server/panel/class/panel_php_run.php --args_tmp="%s" --plugin_name="%s" --fun="%s"' % (php_bin, self.__args_tmp, self.__plugin_name, args.s)).read()
        try:
            result = json.loads(result)
        except:
            pass
        if os.path.exists(self.__args_tmp):
            os.remove(self.__args_tmp)
        return result

    def __write_args(self, args):
        if False:
            print('Hello World!')
        from BTPanel import request
        if os.path.exists(self.__args_tmp):
            os.remove(self.__args_tmp)
        self.__clean_args_file()
        data = {}
        data['GET'] = request.args.to_dict()
        data['POST'] = {}
        for key in request.form.keys():
            data['POST'][key] = str(request.form.get(key, ''))
        data['POST']['client_ip'] = public.GetClientIp()
        data = json.dumps(data)
        public.writeFile(self.__args_tmp, data)

    def __clean_args_file(self):
        if False:
            while True:
                i = 10
        args_dir = self.__plugin_path + '/args'
        if not os.path.exists(args_dir):
            return False
        now_time = time.time()
        for f_name in os.listdir(args_dir):
            filename = args_dir + '/' + f_name
            if not os.path.exists(filename):
                continue
            if now_time - os.path.getctime(filename) > 60:
                os.remove(filename)

    def __get_php_bin(self):
        if False:
            for i in range(10):
                print('nop')
        php_v_file = self.__plugin_path + '/php_version.json'
        if os.path.exists(php_v_file):
            php_vs = json.loads(public.readFile(php_v_file).replace('.', ''))
        else:
            php_vs = public.get_php_versions(True)
        php_path = '/www/server/php/'
        php_v = None
        for pv in php_vs:
            php_bin = php_path + pv + '/bin/php'
            if os.path.exists(php_bin):
                php_v = pv
                break
        if not php_v:
            return False
        php_ini = self.__plugin_path + '/php_cli_' + php_v + '.ini'
        if not os.path.exists(php_ini):
            src_php_ini = php_path + php_v + '/etc/php.ini'
            import shutil
            shutil.copy(src_php_ini, php_ini)
            php_ini_body = public.readFile(php_ini)
            php_ini_body = re.sub('disable_functions\\s*=.*', 'disable_functions = ', php_ini_body)
            php_ini_body = re.sub('.*bt_filter.+', '', php_ini_body)
            public.writeFile(php_ini, php_ini_body)
        return php_path + php_v + '/bin/php -c ' + php_ini

    def get_php_version(self, php_version):
        if False:
            i = 10
            return i + 15
        if php_version:
            if not isinstance(php_version, list):
                php_vs = [php_version]
            else:
                php_vs = sorted(php_version, reverse=True)
        else:
            php_vs = public.get_php_versions(True)
        php_path = '/www/server/php/'
        php_v = None
        for pv in php_vs:
            php_bin = php_path + pv + '/bin/php'
            if os.path.exists(php_bin) and os.path.exists('/tmp/php-cgi-{}.sock'.format(pv)):
                php_v = pv
                break
        return php_v

    def get_header_data(self, sock):
        if False:
            while True:
                i = 10
        '\n            @name 获取头部32KB数据\n            @author hwliang<2020-07-11>\n            @param sock socketobject(fastcgi套接字对象)\n            @return bytes\n        '
        headers_data = b''
        total_len = 0
        header_len = 1024 * 128
        while True:
            fastcgi_header = sock.recv(8)
            if not fastcgi_header:
                break
            if len(fastcgi_header) != 8:
                headers_data += fastcgi_header
                break
            fast_pack = struct.unpack(FCGI_Header, fastcgi_header)
            if fast_pack[1] == 3:
                break
            tlen = fast_pack[3]
            while tlen > 0:
                sd = sock.recv(tlen)
                if not sd:
                    break
                headers_data += sd
                tlen -= len(sd)
            total_len += fast_pack[3]
            if fast_pack[4]:
                sock.recv(fast_pack[4])
            if total_len > header_len:
                break
        return headers_data

    def format_header_data(self, headers_data):
        if False:
            return 10
        '\n            @name 格式化响应头\n            @author hwliang<2020-07-11>\n            @param headers_data bytes(fastcgi头部32KB数据)\n            @return status int(响应状态), headers dict(响应头), bdata bytes(格式化响应头后的多余数据)\n        '
        status = '200 OK'
        headers = {}
        pos = 0
        while True:
            eolpos = headers_data.find(b'\n', pos)
            if eolpos < 0:
                break
            line = headers_data[pos:eolpos - 1]
            pos = eolpos + 1
            line = line.strip()
            if len(line) < 2:
                break
            if line.find(b':') == -1:
                continue
            (header, value) = line.split(b':', 1)
            header = header.strip()
            value = value.strip()
            if isinstance(header, bytes):
                header = header.decode()
                value = value.decode()
            if header == 'Status':
                status = value
                if status.find(' ') < 0:
                    status += ' BTPanel'
            else:
                headers[header] = value
        bdata = headers_data[pos:]
        status = int(status.split(' ')[0])
        return (status, headers, bdata)

    def resp_sock(self, sock, bdata):
        if False:
            return 10
        '\n            @name 以流的方式发送剩余数据\n            @author hwliang<2020-07-11>\n            @param sock socketobject(fastcgi套接字对象)\n            @param bdata bytes(格式化响应头后的多余数据)\n            @return yield bytes\n        '
        yield bdata
        while True:
            fastcgi_header = sock.recv(8)
            if not fastcgi_header:
                break
            if len(fastcgi_header) != 8:
                yield fastcgi_header
                break
            fast_pack = struct.unpack(FCGI_Header, fastcgi_header)
            if fast_pack[1] == 3:
                break
            tlen = fast_pack[3]
            while tlen > 0:
                sd = sock.recv(tlen)
                if not sd:
                    break
                tlen -= len(sd)
                if sd:
                    yield sd
            if fast_pack[4]:
                sock.recv(fast_pack[4])
        sock.close()

class FPM(object):

    def __init__(self, sock=None, document_root='', last_path=''):
        if False:
            for i in range(10):
                print('nop')
        '\n            @name 实例化FPM对象\n            @author hwliang<2020-07-11>\n            @param sock string(unixsocket路径)\n            @param document_root string(PHP文档根目录)\n            @return FPM\n        '
        if sock:
            self.fcgi_sock = sock
            if document_root[-1:] != '/':
                document_root += '/'
            self.document_root = document_root
            self.last_path = last_path

    def load_url(self, url, content=b''):
        if False:
            return 10
        '\n            @name 转发URL到PHP-FPM\n            @author hwliang<2020-07-11>\n            @param url string(URI地址)\n            @param content stream(POST数据io对象)\n            @return fastcgi-socket\n        '
        fcgi = fcgi_client.FCGIApp(connect=self.fcgi_sock)
        try:
            (script_name, query_string) = url.split('?')
        except ValueError:
            script_name = url
            query_string = ''
        from BTPanel import request
        env = {'SCRIPT_FILENAME': '%s%s' % (self.document_root, script_name), 'QUERY_STRING': query_string, 'REQUEST_METHOD': request.method, 'SCRIPT_NAME': self.last_path + script_name, 'REQUEST_URI': self.last_path + url, 'GATEWAY_INTERFACE': 'CGI/1.1', 'SERVER_SOFTWARE': 'BT-Panel', 'REDIRECT_STATUS': '200', 'CONTENT_TYPE': request.headers.get('Content-Type', 'application/x-www-form-urlencoded'), 'CONTENT_LENGTH': str(request.headers.get('Content-Length', '0')), 'DOCUMENT_URI': request.path, 'DOCUMENT_ROOT': self.document_root, 'SERVER_PROTOCOL': 'HTTP/1.1', 'REMOTE_ADDR': request.remote_addr.replace('::ffff:', ''), 'REMOTE_PORT': str(request.environ.get('REMOTE_PORT')), 'SERVER_ADDR': request.headers.get('host'), 'SERVER_PORT': '80', 'SERVER_NAME': 'BT-Panel'}
        for k in request.headers.keys():
            key = 'HTTP_' + k.replace('-', '_').upper()
            env[key] = request.headers[k]
        fpm_sock = fcgi(env, content)
        return fpm_sock

    def load_url_public(self, url, content=b'', method='GET', content_type='application/x-www-form-urlencoded'):
        if False:
            print('Hello World!')
        '\n            @name 转发URL到PHP-FPM 公共\n            @author hwliang<2020-07-11>\n            @param url string(URI地址)\n            @param content stream(POST数据io对象)\n            @return fastcgi-socket\n        '
        fcgi = fcgi_client.FCGIApp(connect=self.fcgi_sock)
        try:
            (script_name, query_string) = url.split('?')
        except ValueError:
            script_name = url
            query_string = ''
        content_length = len(content)
        if content:
            content = StringIO(content)
        env = {'SCRIPT_FILENAME': '%s%s' % (self.document_root, script_name), 'QUERY_STRING': query_string, 'REQUEST_METHOD': method, 'SCRIPT_NAME': self.last_path + script_name, 'REQUEST_URI': url, 'GATEWAY_INTERFACE': 'CGI/1.1', 'SERVER_SOFTWARE': 'BT-Panel', 'REDIRECT_STATUS': '200', 'CONTENT_TYPE': content_type, 'CONTENT_LENGTH': str(content_length), 'DOCUMENT_URI': script_name, 'DOCUMENT_ROOT': self.document_root, 'SERVER_PROTOCOL': 'HTTP/1.1', 'REMOTE_ADDR': '127.0.0.1', 'REMOTE_PORT': '8888', 'SERVER_ADDR': '127.0.0.1', 'SERVER_PORT': '80', 'SERVER_NAME': 'BT-Panel'}
        fpm_sock = fcgi(env, content)
        _data = b''
        while True:
            fastcgi_header = fpm_sock.recv(8)
            if not fastcgi_header:
                break
            if len(fastcgi_header) != 8:
                _data += fastcgi_header
                break
            fast_pack = struct.unpack(FCGI_Header, fastcgi_header)
            if fast_pack[1] == 3:
                break
            tlen = fast_pack[3]
            while tlen > 0:
                sd = fpm_sock.recv(tlen)
                if not sd:
                    break
                tlen -= len(sd)
                _data += sd
            if fast_pack[4]:
                fpm_sock.recv(fast_pack[4])
        (status, headers, data) = panelPHP().format_header_data(_data)
        return data