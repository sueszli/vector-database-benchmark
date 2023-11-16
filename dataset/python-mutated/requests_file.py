"""
Copyright 2015 Red Hat, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import errno
import io
import locale
import os
import os.path
import stat
import sys
from io import BytesIO
from urllib.parse import unquote, urljoin, urlparse
from requests import Response, codes
from requests.adapters import BaseAdapter
from streamlink.compat import is_win32

class FileAdapter(BaseAdapter):

    def send(self, request, **kwargs):
        if False:
            print('Hello World!')
        ' Wraps a file, described in request, in a Response object.\n\n            :param request: The PreparedRequest` being "sent".\n            :returns: a Response object containing the file\n        '
        if request.method not in ('GET', 'HEAD'):
            raise ValueError(f'Invalid request method {request.method}')
        url_parts = urlparse(request.url)
        if is_win32 and url_parts.netloc.endswith(':'):
            url_parts = url_parts._replace(path=f'/{url_parts.netloc}{url_parts.path}', netloc='')
        if url_parts.netloc and url_parts.netloc not in ('localhost', '.', '..', '-'):
            raise ValueError('file: URLs with hostname components are not permitted')
        if url_parts.netloc in ('.', '..'):
            pwd = os.path.abspath(url_parts.netloc).replace(os.sep, '/') + '/'
            if is_win32:
                pwd = f'/{pwd}'
            url_parts = url_parts._replace(path=urljoin(pwd, url_parts.path.lstrip('/')))
        resp = Response()
        resp.url = request.url
        try:
            if url_parts.netloc == '-':
                resp.raw = sys.stdin.buffer
                resp.url = 'file://' + os.path.abspath('.').replace(os.sep, '/') + '/'
            else:
                path_parts = [unquote(p) for p in url_parts.path.split('/')]
                while path_parts and (not path_parts[0]):
                    path_parts.pop(0)
                if any((os.sep in p for p in path_parts)):
                    raise IOError(errno.ENOENT, os.strerror(errno.ENOENT))
                if path_parts and (path_parts[0].endswith('|') or path_parts[0].endswith(':')):
                    path_drive = path_parts.pop(0)
                    if path_drive.endswith('|'):
                        path_drive = f'{path_drive[:-1]}:'
                    while path_parts and (not path_parts[0]):
                        path_parts.pop(0)
                else:
                    path_drive = ''
                path = path_drive + os.sep + os.path.join(*path_parts)
                if path_drive and (not os.path.splitdrive(path)):
                    path = os.sep + os.path.join(path_drive, *path_parts)
                resp.raw = io.open(path, 'rb')
                resp.raw.release_conn = resp.raw.close
        except IOError as e:
            if e.errno == errno.EACCES:
                resp.status_code = codes.forbidden
            elif e.errno == errno.ENOENT:
                resp.status_code = codes.not_found
            else:
                resp.status_code = codes.bad_request
            resp_str = str(e).encode(locale.getpreferredencoding(False))
            resp.raw = BytesIO(resp_str)
            resp.headers['Content-Length'] = len(resp_str)
            resp.raw.release_conn = resp.raw.close
        else:
            resp.status_code = codes.ok
            resp_stat = os.fstat(resp.raw.fileno())
            if stat.S_ISREG(resp_stat.st_mode):
                resp.headers['Content-Length'] = resp_stat.st_size
        return resp

    def close(self):
        if False:
            while True:
                i = 10
        pass