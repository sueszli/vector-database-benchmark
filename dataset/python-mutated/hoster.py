import inspect
import re
import time
import urllib.parse
from pyload.core.network.exceptions import Abort, Fail, Reconnect, Retry, Skip
from pyload.core.utils import format, parse
from pyload.core.utils.old import fixurl
from ..helpers import parse_html_form, replace_patterns
from .captcha import BaseCaptcha
from .plugin import BasePlugin

def get_info(urls):
    if False:
        print('Hello World!')
    pass

def parse_file_info(klass, url='', html=''):
    if False:
        for i in range(10):
            print('nop')
    info = klass.get_info(url, html)
    return (info['name'], info['size'], info['status'], info['url'])

class BaseHoster(BasePlugin):
    __name__ = 'BaseHoster'
    __type__ = 'base'
    __version__ = '0.41'
    __status__ = 'stable'
    __pattern__ = '^unmatchable$'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True)]
    __description__ = 'Base hoster plugin'
    __license__ = 'GPLv3'
    __authors__ = [('Walter Purcaro', 'vuolter@gmail.com')]
    URL_REPLACEMENTS = []

    def get_info(self, url='', html=''):
        if False:
            i = 10
            return i + 15
        url = fixurl(url, unquote=True)
        info = {'name': parse.name(url), 'hash': {}, 'pattern': {}, 'size': 0, 'status': 7 if url else 8, 'url': replace_patterns(url, self.URL_REPLACEMENTS)}
        try:
            info['pattern'] = re.match(self.__pattern__, url).groupdict()
        except Exception:
            pass
        return info

    def __init__(self, pyfile):
        if False:
            return 10
        self._init(pyfile.m.pyload)
        self.want_reconnect = False
        self.multi_dl = True
        self.waiting = False
        self.account = None
        self.premium = None
        self.pyfile = pyfile
        self.thread = None
        self.captcha = BaseCaptcha(pyfile)
        self.data = ''
        self.retries = {}
        self.init_base()
        self.init()

    def _log(self, level, plugintype, pluginname, args, kwargs):
        if False:
            print('Hello World!')
        log = getattr(self.pyload.log, level)
        try:
            user = self.account.user
            hidden_user = '{:*<{}}'.format(self.account.user[:3], 7)
            args = tuple((arg.replace(user, hidden_user) for arg in args if arg))
        except (AttributeError, KeyError, TypeError):
            pass
        try:
            pw = self.account.info['login']['password']
            hidden_pw = '*' * 10
            args = tuple((arg.replace(pw, hidden_pw) for arg in args if arg))
        except (AttributeError, KeyError, TypeError):
            pass
        log('{plugintype} {pluginname}[{id}]: {msg}'.format(plugintype=plugintype.upper(), pluginname=pluginname, id=self.pyfile.id, msg=' | '.join(['%s'] * len(args))), *args, **kwargs)

    def init_base(self):
        if False:
            return 10
        pass

    def setup_base(self):
        if False:
            i = 10
            return i + 15
        pass

    def setup(self):
        if False:
            while True:
                i = 10
        '\n        Setup for enviroment and other things, called before downloading (possibly more\n        than one time)\n        '
        pass

    def _setup(self):
        if False:
            print('Hello World!')
        self.pyfile.error = ''
        self.data = ''
        self.last_html = ''
        self.last_header = {}
        if self.config.get('use_premium', True):
            self.load_account()
        else:
            self.account = False
        try:
            self.req.close()
        except Exception:
            pass
        if self.account:
            self.req = self.pyload.request_factory.get_request(self.classname, self.account.user)
            self.premium = self.account.info['data']['premium']
        else:
            self.req = self.pyload.request_factory.get_request(self.classname)
            self.premium = False
        self.setup_base()
        self.grab_info()
        self.setup()
        self.check_status()

    def load_account(self):
        if False:
            return 10
        if self.account is None:
            self.account = self.pyload.account_manager.get_account_plugin(self.classname)
        if self.account:
            self.account.choose()

    def _update_name(self):
        if False:
            for i in range(10):
                print('nop')
        name = self.info.get('name')
        if name and name != self.info.get('url'):
            self.pyfile.name = name
        else:
            name = self.pyfile.name
        self.log_info(self._('Link name: {}').format(name))

    def _update_size(self):
        if False:
            print('Hello World!')
        size = self.info.get('size')
        if size > 0:
            self.pyfile.size = int(self.info.get('size'))
        else:
            size = self.pyfile.size
        if size:
            self.log_info(self._('Link size: {} ({} bytes)').format(format.size(size), size))
        else:
            self.log_info(self._('Link size: N/D'))

    def _update_status(self):
        if False:
            return 10
        self.pyfile.status = self.info.get('status', 14)
        self.pyfile.sync()
        self.log_info(self._('Link status: ') + self.pyfile.get_status_name())

    def sync_info(self):
        if False:
            print('Hello World!')
        self._update_name()
        self._update_size()
        self._update_status()

    def grab_info(self):
        if False:
            i = 10
            return i + 15
        if self.pyfile.status != 2:
            self.log_info(self._('Grabbing link info...'))
            old_info = dict(self.info)
            new_info = self.get_info(replace_patterns(self.pyfile.url, self.URL_REPLACEMENTS), self.data)
            self.info.update(new_info)
            self.log_debug(f'Link info: {self.info}')
            self.log_debug(f'Previous link info: {old_info}')
            self.sync_info()

    def check_status(self):
        if False:
            while True:
                i = 10
        status = self.pyfile.status
        if status == 1:
            self.offline()
        elif status == 4:
            self.skip(self.pyfile.statusname)
        elif status == 6:
            self.temp_offline()
        elif status == 8:
            self.fail()
        elif status == 9 or self.pyfile.abort:
            self.abort()

    def _initialize(self):
        if False:
            i = 10
            return i + 15
        self.log_debug('Plugin version: ' + self.__version__)
        self.log_debug('Plugin status: ' + self.__status__)
        if self.__status__ == 'broken':
            self.abort(self._('Plugin is temporarily unavailable'))
        elif self.__status__ == 'testing':
            self.log_warning(self._('Plugin may be unstable'))

    def _process(self, thread):
        if False:
            return 10
        '\n        Handles important things to do before starting.\n        '
        self.thread = thread
        self._initialize()
        self._setup()
        if self.__type__ == 'decrypter':
            self.pyload.addon_manager.download_preparing(self.pyfile)
            self.check_status()
        self.pyfile.set_status('starting')
        self.log_info(self._('Processing url: ') + self.pyfile.url)
        self.process(self.pyfile)
        self.check_status()

    def preprocessing(self, *args, **kwargs):
        if False:
            return 10
        return self._process(*args, **kwargs)

    def process(self, pyfile):
        if False:
            i = 10
            return i + 15
        '\n        The "main" method of every downloader plugin, you **have to** overwrite it.\n        '
        raise NotImplementedError

    def set_reconnect(self, reconnect):
        if False:
            while True:
                i = 10
        if self.pyload.config.get('reconnect', 'enabled'):
            reconnect = reconnect and self.pyload.api.is_time_reconnect()
            self.log_debug('RECONNECT{} required'.format('' if reconnect else ' not'), 'Previous want_reconnect: {}'.format(self.want_reconnect))
            self.want_reconnect = bool(reconnect)

    def set_wait(self, seconds, strict=False):
        if False:
            i = 10
            return i + 15
        '\n        Set a specific wait time later used with wait()\n\n        :param seconds: wait time in seconds\n        :param strict: strict mode\n        '
        wait_time = float(seconds)
        if wait_time < 0:
            return False
        old_wait_until = self.pyfile.wait_until
        new_wait_until = time.time() + wait_time + float(not strict)
        self.log_debug('WAIT set to timestamp {}'.format(new_wait_until), 'Previous wait_until: {}'.format(old_wait_until))
        self.pyfile.wait_until = new_wait_until
        return True

    def wait(self, seconds=None, reconnect=None):
        if False:
            while True:
                i = 10
        '\n        Waits the time previously set.\n\n        :param seconds: How many seconds to wait or if equals to None then use the value from set_wait()\n        :param reconnect: True if reconnect would avoid wait time\n        '
        if seconds is not None:
            self.set_wait(seconds)
        wait_time = self.pyfile.wait_until - time.time()
        if wait_time < 1:
            self.log_warning(self._('Invalid wait time interval'))
            return
        if reconnect is None:
            reconnect = wait_time > self.config.get('max_wait', 10) * 60
        self.set_reconnect(reconnect)
        self.waiting = True
        status = self.pyfile.status
        self.pyfile.set_status('waiting')
        self.log_info(self._('Waiting {}...').format(format.time(wait_time)))
        if self.want_reconnect:
            self.log_info(self._('Requiring reconnection...'))
            if self.account:
                self.log_warning(self._('Reconnection ignored due logged account'))
        if not self.want_reconnect or self.account:
            while self.pyfile.wait_until > time.time():
                self.check_status()
                time.sleep(2)
        else:
            while self.pyfile.wait_until > time.time():
                self.check_status()
                self.thread.m.reconnecting.wait(1)
                if self.thread.m.reconnecting.is_set():
                    self.waiting = False
                    self.want_reconnect = False
                    self.req.clear_cookies()
                    raise Reconnect
                time.sleep(2)
        self.waiting = False
        self.pyfile.status = status

    def skip(self, msg=''):
        if False:
            while True:
                i = 10
        '\n        Skip and give msg.\n        '
        raise Skip(msg or self.pyfile.error or self.pyfile.pluginname)

    def fail(self, msg=''):
        if False:
            i = 10
            return i + 15
        '\n        Fail and give msg.\n        '
        msg = msg.strip()
        if msg:
            self.pyfile.error = msg
        else:
            msg = self.pyfile.error or self.info.get('error') or self.pyfile.get_status_name()
        raise Fail(msg)

    def error(self, msg='', type='Parse'):
        if False:
            print('Hello World!')
        type = self._('{} error').format(type.strip().capitalize() if type else self._('Unknown'))
        msg = self._('{type}: {msg} | Plugin may be out of date').format(type=type, msg=msg or self.pyfile.error)
        self.fail(msg)

    def abort(self, msg=''):
        if False:
            while True:
                i = 10
        '\n        Abort and give msg.\n        '
        if msg:
            self.pyfile.error = msg
        raise Abort

    def offline(self, msg=''):
        if False:
            i = 10
            return i + 15
        '\n        Fail and indicate file is offline.\n        '
        self.fail('offline')

    def temp_offline(self, msg=''):
        if False:
            print('Hello World!')
        '\n        Fail and indicates file ist temporary offline, the core may take consequences.\n        '
        self.fail('temp. offline')

    def restart(self, msg='', premium=True):
        if False:
            print('Hello World!')
        if not msg:
            msg = self._('Restart plugin') if premium else self._('Fallback to free processing')
        if not premium:
            if self.premium:
                self.restart_free = True
            else:
                self.fail('{} | {}'.format(msg, self._('Url was already processed as free')))
        raise Retry(msg)

    def retry(self, attempts=5, wait=1, msg='', msgfail='Max retries reached'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retries and begin again from the beginning.\n\n        :param attempts: number of maximum retries\n        :param wait: time to wait in seconds before retry\n        :param msg: message to pass to retry if attempts value was not yet reached\n        :param msgfail: message passed to fail if attempts value was reached\n        '
        frame = inspect.currentframe()
        try:
            id = frame.f_back.f_lineno
        finally:
            del frame
        if id not in self.retries:
            self.retries[id] = 0
        if 0 < attempts <= self.retries[id]:
            self.fail(msgfail)
        self.retries[id] += 1
        self.wait(wait)
        raise Retry(msg)

    def retry_captcha(self, attempts=10, wait=1, msg='', msgfail='Max captcha retries reached'):
        if False:
            print('Hello World!')
        self.captcha.invalid(msg)
        self.retry(attempts, wait, msg=self._('Retry Captcha'), msgfail=msgfail)

    def fixurl(self, url, baseurl=None):
        if False:
            for i in range(10):
                print('nop')
        baseurl = baseurl or self.pyfile.url
        if not urllib.parse.urlparse(url).scheme:
            url_p = urllib.parse.urlparse(baseurl)
            baseurl = '{}://{}'.format(url_p.scheme, url_p.netloc)
            url = urllib.parse.urljoin(baseurl, url)
        return url

    def load(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.check_status()
        return super().load(*args, **kwargs)

    def parse_html_form(self, attr_str='', input_names={}):
        if False:
            while True:
                i = 10
        return parse_html_form(attr_str, self.data, input_names)

    def get_password(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the password the user provided in the package.\n        '
        return self.pyfile.package().password or ''

    def clean(self):
        if False:
            print('Hello World!')
        '\n        Clean everything and remove references.\n        '
        super().clean()
        for attr in ('account', 'html', 'pyfile', 'thread'):
            if hasattr(self, attr):
                setattr(self, attr, None)