import locale
import re
import time
import urllib.parse
from pyload.core.utils import parse
from ..helpers import parse_html_form, set_cookie, search_pattern
from .account import BaseAccount

class XFSAccount(BaseAccount):
    __name__ = 'XFSAccount'
    __type__ = 'account'
    __version__ = '0.65'
    __status__ = 'stable'
    __config__ = [('enabled', 'bool', 'Activated', True), ('multi', 'bool', 'Multi-hoster', True), ('multi_mode', 'all;listed;unlisted', 'Hosters to use', 'all'), ('multi_list', 'str', 'Hoster list (comma separated)', ''), ('multi_interval', 'int', 'Reload interval in hours', 12)]
    __description__ = 'XFileSharing account plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zoidberg', 'zoidberg@mujmail.cz'), ('Walter Purcaro', 'vuolter@gmail.com')]
    PLUGIN_DOMAIN = None
    PLUGIN_URL = None
    LOGIN_URL = None
    COOKIES = True
    PREMIUM_PATTERN = '\\(Premium only\\)'
    VALID_UNTIL_PATTERN = 'Premium.[Aa]ccount expire:.*?(\\d{1,2} [\\w^_]+ \\d{4})'
    TRAFFIC_LEFT_PATTERN = 'Traffic available today:.*?<b>\\s*(?P<S>[\\d.,]+|[Uu]nlimited)\\s*(?:(?P<U>[\\w^_]+)\\s*)?</b>'
    TRAFFIC_LEFT_UNIT = 'MiB'
    LEECH_TRAFFIC_PATTERN = 'Leech Traffic left:<b>.*?(?P<S>[\\d.,]+|[Uu]nlimited)\\s*(?:(?P<U>[\\w^_]+)\\s*)?</b>'
    LEECH_TRAFFIC_UNIT = 'MiB'
    LOGIN_FAIL_PATTERN = 'Incorrect Login or Password|account was banned|Error<'
    LOGIN_BAN_PATTERN = '>(Your IP.+?)<a'
    LOGIN_SKIP_PATTERN = 'op=logout'

    def _set_xfs_cookie(self):
        if False:
            print('Hello World!')
        cookie = (self.PLUGIN_DOMAIN, 'lang', 'english')
        if isinstance(self.COOKIES, list) and cookie not in self.COOKIES:
            self.COOKIES.insert(cookie)
        else:
            set_cookie(self.req.cj, *cookie)

    def setup(self):
        if False:
            print('Hello World!')
        if not self.PLUGIN_DOMAIN:
            self.fail_login(self._('Missing PLUGIN DOMAIN'))
        if not self.PLUGIN_URL:
            self.PLUGIN_URL = 'http://www.{}/'.format(self.PLUGIN_DOMAIN)
        if not self.LOGIN_URL:
            self.LOGIN_URL = urllib.parse.urljoin(self.PLUGIN_URL, 'login.html')
        if self.COOKIES:
            self._set_xfs_cookie()

    def grab_info(self, user, password, data):
        if False:
            for i in range(10):
                print('nop')
        validuntil = None
        trafficleft = None
        leechtraffic = None
        premium = None
        if not self.PLUGIN_URL:
            return
        self.data = self.load(self.PLUGIN_URL, get={'op': 'my_account'}, cookies=self.COOKIES)
        premium = True if search_pattern(self.PREMIUM_PATTERN, self.data) is not None else False
        m = search_pattern(self.VALID_UNTIL_PATTERN, self.data)
        if m is not None:
            expiredate = m.group(1).strip()
            self.log_debug('Expire date: ' + expiredate)
            previous_locale = locale.getlocale(locale.LC_TIME)
            try:
                locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
                validuntil = time.mktime(time.strptime(expiredate, '%d %B %Y'))
            except Exception as exc:
                self.log_error(exc)
            else:
                self.log_debug(f'Valid until: {validuntil}')
                if validuntil > time.mktime(time.gmtime()):
                    premium = True
                    trafficleft = -1
                else:
                    premium = False
                    validuntil = None
            finally:
                locale.setlocale(locale.LC_TIME, previous_locale)
        else:
            self.log_debug('VALID UNTIL PATTERN not found')
        m = search_pattern(self.TRAFFIC_LEFT_PATTERN, self.data)
        if m is not None:
            try:
                traffic = m.groupdict()
                size = traffic['S']
                if 'nlimited' in size:
                    trafficleft = -1
                    if validuntil is None:
                        validuntil = -1
                else:
                    if 'U' in traffic:
                        unit = traffic['U']
                    elif isinstance(self.TRAFFIC_LEFT_UNIT, str):
                        unit = self.TRAFFIC_LEFT_UNIT
                    else:
                        unit = ''
                    trafficleft = max(0, self.parse_traffic(size, unit))
            except Exception as exc:
                self.log_error(exc)
        else:
            self.log_debug('TRAFFIC LEFT PATTERN not found')
        leech = [m.groupdict() for m in re.finditer(self.LEECH_TRAFFIC_PATTERN, self.data)]
        if leech:
            leechtraffic = 0
            try:
                for traffic in leech:
                    size = traffic['S']
                    if 'nlimited' in size:
                        leechtraffic = -1
                        if validuntil is None:
                            validuntil = -1
                        break
                    else:
                        if 'U' in traffic:
                            unit = traffic['U']
                        elif isinstance(self.LEECH_TRAFFIC_UNIT, str):
                            unit = self.LEECH_TRAFFIC_UNIT
                        else:
                            unit = ''
                        leechtraffic += self.parse_traffic(size + unit)
            except Exception as exc:
                self.log_error(exc)
        else:
            self.log_debug('LEECH TRAFFIC PATTERN not found')
        return {'validuntil': validuntil, 'trafficleft': trafficleft, 'leechtraffic': leechtraffic, 'premium': premium}

    def signin(self, user, password, data):
        if False:
            for i in range(10):
                print('nop')
        self.data = self.load(self.LOGIN_URL, cookies=self.COOKIES)
        if search_pattern(self.LOGIN_SKIP_PATTERN, self.data):
            self.skip_login()
        (action, inputs) = parse_html_form('name="FL"', self.data)
        if not inputs:
            inputs = {'op': 'login', 'redirect': self.PLUGIN_URL}
        inputs.update({'login': user, 'password': password})
        if action:
            url = urllib.parse.urljoin(self.LOGIN_URL, action)
        else:
            url = self.LOGIN_URL
        self.data = self.load(url, post=inputs, cookies=self.COOKIES)
        self.check_errors()

    def check_errors(self, data=None):
        if False:
            return 10
        self.log_info(self._('Checking for link errors...'))
        data = data or self.data
        if not data:
            self.log_warning(self._('No data to check'))
            return
        m = search_pattern(self.LOGIN_BAN_PATTERN, data)
        if m is not None:
            try:
                errmsg = m.group(1)
            except (AttributeError, IndexError):
                errmsg = m.group(0)
            finally:
                errmsg = re.sub('<.*?>', ' ', errmsg.strip())
            new_timeout = parse.seconds(errmsg)
            if new_timeout > self.timeout:
                self.timeout = new_timeout
            self.fail_login(errmsg)
        m = search_pattern(self.LOGIN_FAIL_PATTERN, data)
        if m is not None:
            try:
                errmsg = m.group(1)
            except (AttributeError, IndexError):
                errmsg = m.group(0)
            finally:
                errmsg = re.sub('<.*?>', ' ', errmsg.strip())
            self.timeout = self.LOGIN_TIMEOUT
            self.fail_login(errmsg)
        self.log_info(self._('No errors found'))