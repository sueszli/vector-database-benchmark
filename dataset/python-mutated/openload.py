import collections
import contextlib
import json
import os
import subprocess
import tempfile
from ..compat import compat_urlparse
from ..utils import ExtractorError, Popen, check_executable, format_field, get_exe_version, is_outdated_version, shell_quote

def cookie_to_dict(cookie):
    if False:
        print('Hello World!')
    cookie_dict = {'name': cookie.name, 'value': cookie.value}
    if cookie.port_specified:
        cookie_dict['port'] = cookie.port
    if cookie.domain_specified:
        cookie_dict['domain'] = cookie.domain
    if cookie.path_specified:
        cookie_dict['path'] = cookie.path
    if cookie.expires is not None:
        cookie_dict['expires'] = cookie.expires
    if cookie.secure is not None:
        cookie_dict['secure'] = cookie.secure
    if cookie.discard is not None:
        cookie_dict['discard'] = cookie.discard
    with contextlib.suppress(TypeError):
        if cookie.has_nonstandard_attr('httpOnly') or cookie.has_nonstandard_attr('httponly') or cookie.has_nonstandard_attr('HttpOnly'):
            cookie_dict['httponly'] = True
    return cookie_dict

def cookie_jar_to_list(cookie_jar):
    if False:
        return 10
    return [cookie_to_dict(cookie) for cookie in cookie_jar]

class PhantomJSwrapper:
    """PhantomJS wrapper class

    This class is experimental.
    """
    INSTALL_HINT = 'Please download it from https://phantomjs.org/download.html'
    _BASE_JS = "\n        phantom.onError = function(msg, trace) {{\n          var msgStack = ['PHANTOM ERROR: ' + msg];\n          if(trace && trace.length) {{\n            msgStack.push('TRACE:');\n            trace.forEach(function(t) {{\n              msgStack.push(' -> ' + (t.file || t.sourceURL) + ': ' + t.line\n                + (t.function ? ' (in function ' + t.function +')' : ''));\n            }});\n          }}\n          console.error(msgStack.join('\\n'));\n          phantom.exit(1);\n        }};\n    "
    _TEMPLATE = '\n        var page = require(\'webpage\').create();\n        var fs = require(\'fs\');\n        var read = {{ mode: \'r\', charset: \'utf-8\' }};\n        var write = {{ mode: \'w\', charset: \'utf-8\' }};\n        JSON.parse(fs.read("{cookies}", read)).forEach(function(x) {{\n          phantom.addCookie(x);\n        }});\n        page.settings.resourceTimeout = {timeout};\n        page.settings.userAgent = "{ua}";\n        page.onLoadStarted = function() {{\n          page.evaluate(function() {{\n            delete window._phantom;\n            delete window.callPhantom;\n          }});\n        }};\n        var saveAndExit = function() {{\n          fs.write("{html}", page.content, write);\n          fs.write("{cookies}", JSON.stringify(phantom.cookies), write);\n          phantom.exit();\n        }};\n        page.onLoadFinished = function(status) {{\n          if(page.url === "") {{\n            page.setContent(fs.read("{html}", read), "{url}");\n          }}\n          else {{\n            {jscode}\n          }}\n        }};\n        page.open("");\n    '
    _TMP_FILE_NAMES = ['script', 'html', 'cookies']

    @staticmethod
    def _version():
        if False:
            return 10
        return get_exe_version('phantomjs', version_re='([0-9.]+)')

    def __init__(self, extractor, required_version=None, timeout=10000):
        if False:
            while True:
                i = 10
        self._TMP_FILES = {}
        self.exe = check_executable('phantomjs', ['-v'])
        if not self.exe:
            raise ExtractorError(f'PhantomJS not found, {self.INSTALL_HINT}', expected=True)
        self.extractor = extractor
        if required_version:
            version = self._version()
            if is_outdated_version(version, required_version):
                self.extractor._downloader.report_warning('Your copy of PhantomJS is outdated, update it to version %s or newer if you encounter any errors.' % required_version)
        for name in self._TMP_FILE_NAMES:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.close()
            self._TMP_FILES[name] = tmp
        self.options = collections.ChainMap({'timeout': timeout}, {x: self._TMP_FILES[x].name.replace('\\', '\\\\').replace('"', '\\"') for x in self._TMP_FILE_NAMES})

    def __del__(self):
        if False:
            while True:
                i = 10
        for name in self._TMP_FILE_NAMES:
            with contextlib.suppress(OSError, KeyError):
                os.remove(self._TMP_FILES[name].name)

    def _save_cookies(self, url):
        if False:
            print('Hello World!')
        cookies = cookie_jar_to_list(self.extractor.cookiejar)
        for cookie in cookies:
            if 'path' not in cookie:
                cookie['path'] = '/'
            if 'domain' not in cookie:
                cookie['domain'] = compat_urlparse.urlparse(url).netloc
        with open(self._TMP_FILES['cookies'].name, 'wb') as f:
            f.write(json.dumps(cookies).encode('utf-8'))

    def _load_cookies(self):
        if False:
            while True:
                i = 10
        with open(self._TMP_FILES['cookies'].name, 'rb') as f:
            cookies = json.loads(f.read().decode('utf-8'))
        for cookie in cookies:
            if cookie['httponly'] is True:
                cookie['rest'] = {'httpOnly': None}
            if 'expiry' in cookie:
                cookie['expire_time'] = cookie['expiry']
            self.extractor._set_cookie(**cookie)

    def get(self, url, html=None, video_id=None, note=None, note2='Executing JS on webpage', headers={}, jscode='saveAndExit();'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Downloads webpage (if needed) and executes JS\n\n        Params:\n            url: website url\n            html: optional, html code of website\n            video_id: video id\n            note: optional, displayed when downloading webpage\n            note2: optional, displayed when executing JS\n            headers: custom http headers\n            jscode: code to be executed when page is loaded\n\n        Returns tuple with:\n            * downloaded website (after JS execution)\n            * anything you print with `console.log` (but not inside `page.execute`!)\n\n        In most cases you don't need to add any `jscode`.\n        It is executed in `page.onLoadFinished`.\n        `saveAndExit();` is mandatory, use it instead of `phantom.exit()`\n        It is possible to wait for some element on the webpage, e.g.\n            var check = function() {\n              var elementFound = page.evaluate(function() {\n                return document.querySelector('#b.done') !== null;\n              });\n              if(elementFound)\n                saveAndExit();\n              else\n                window.setTimeout(check, 500);\n            }\n\n            page.evaluate(function(){\n              document.querySelector('#a').click();\n            });\n            check();\n        "
        if 'saveAndExit();' not in jscode:
            raise ExtractorError('`saveAndExit();` not found in `jscode`')
        if not html:
            html = self.extractor._download_webpage(url, video_id, note=note, headers=headers)
        with open(self._TMP_FILES['html'].name, 'wb') as f:
            f.write(html.encode('utf-8'))
        self._save_cookies(url)
        user_agent = headers.get('User-Agent') or self.extractor.get_param('http_headers')['User-Agent']
        jscode = self._TEMPLATE.format_map(self.options.new_child({'url': url, 'ua': user_agent.replace('"', '\\"'), 'jscode': jscode}))
        stdout = self.execute(jscode, video_id, note=note2)
        with open(self._TMP_FILES['html'].name, 'rb') as f:
            html = f.read().decode('utf-8')
        self._load_cookies()
        return (html, stdout)

    def execute(self, jscode, video_id=None, *, note='Executing JS'):
        if False:
            while True:
                i = 10
        'Execute JS and return stdout'
        if 'phantom.exit();' not in jscode:
            jscode += ';\nphantom.exit();'
        jscode = self._BASE_JS + jscode
        with open(self._TMP_FILES['script'].name, 'w', encoding='utf-8') as f:
            f.write(jscode)
        self.extractor.to_screen(f"{format_field(video_id, None, '%s: ')}{note}")
        cmd = [self.exe, '--ssl-protocol=any', self._TMP_FILES['script'].name]
        self.extractor.write_debug(f'PhantomJS command line: {shell_quote(cmd)}')
        try:
            (stdout, stderr, returncode) = Popen.run(cmd, timeout=self.options['timeout'] / 1000, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise ExtractorError(f'{note} failed: Unable to run PhantomJS binary', cause=e)
        if returncode:
            raise ExtractorError(f'{note} failed with returncode {returncode}:\n{stderr.strip()}')
        return stdout