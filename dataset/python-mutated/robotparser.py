""" robotparser.py

    Copyright (C) 2000  Bastian Kleineidam

    You can choose between two licenses when using this package:
    1) GNU GPLv2
    2) PSF license for Python 2.2

    The robots.txt Exclusion Protocol is implemented as specified in
    http://www.robotstxt.org/norobots-rfc.txt
"""
import collections
import urllib.parse
import urllib.request
__all__ = ['RobotFileParser']
RequestRate = collections.namedtuple('RequestRate', 'requests seconds')

class RobotFileParser:
    """ This class provides a set of methods to read, parse and answer
    questions about a single robots.txt file.

    """

    def __init__(self, url=''):
        if False:
            i = 10
            return i + 15
        self.entries = []
        self.sitemaps = []
        self.default_entry = None
        self.disallow_all = False
        self.allow_all = False
        self.set_url(url)
        self.last_checked = 0

    def mtime(self):
        if False:
            i = 10
            return i + 15
        'Returns the time the robots.txt file was last fetched.\n\n        This is useful for long-running web spiders that need to\n        check for new robots.txt files periodically.\n\n        '
        return self.last_checked

    def modified(self):
        if False:
            i = 10
            return i + 15
        'Sets the time the robots.txt file was last fetched to the\n        current time.\n\n        '
        import time
        self.last_checked = time.time()

    def set_url(self, url):
        if False:
            return 10
        'Sets the URL referring to a robots.txt file.'
        self.url = url
        (self.host, self.path) = urllib.parse.urlparse(url)[1:3]

    def read(self):
        if False:
            i = 10
            return i + 15
        'Reads the robots.txt URL and feeds it to the parser.'
        try:
            f = urllib.request.urlopen(self.url)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                self.disallow_all = True
            elif err.code >= 400 and err.code < 500:
                self.allow_all = True
        else:
            raw = f.read()
            self.parse(raw.decode('utf-8').splitlines())

    def _add_entry(self, entry):
        if False:
            while True:
                i = 10
        if '*' in entry.useragents:
            if self.default_entry is None:
                self.default_entry = entry
        else:
            self.entries.append(entry)

    def parse(self, lines):
        if False:
            while True:
                i = 10
        'Parse the input lines from a robots.txt file.\n\n        We allow that a user-agent: line is not preceded by\n        one or more blank lines.\n        '
        state = 0
        entry = Entry()
        self.modified()
        for line in lines:
            if not line:
                if state == 1:
                    entry = Entry()
                    state = 0
                elif state == 2:
                    self._add_entry(entry)
                    entry = Entry()
                    state = 0
            i = line.find('#')
            if i >= 0:
                line = line[:i]
            line = line.strip()
            if not line:
                continue
            line = line.split(':', 1)
            if len(line) == 2:
                line[0] = line[0].strip().lower()
                line[1] = urllib.parse.unquote(line[1].strip())
                if line[0] == 'user-agent':
                    if state == 2:
                        self._add_entry(entry)
                        entry = Entry()
                    entry.useragents.append(line[1])
                    state = 1
                elif line[0] == 'disallow':
                    if state != 0:
                        entry.rulelines.append(RuleLine(line[1], False))
                        state = 2
                elif line[0] == 'allow':
                    if state != 0:
                        entry.rulelines.append(RuleLine(line[1], True))
                        state = 2
                elif line[0] == 'crawl-delay':
                    if state != 0:
                        if line[1].strip().isdigit():
                            entry.delay = int(line[1])
                        state = 2
                elif line[0] == 'request-rate':
                    if state != 0:
                        numbers = line[1].split('/')
                        if len(numbers) == 2 and numbers[0].strip().isdigit() and numbers[1].strip().isdigit():
                            entry.req_rate = RequestRate(int(numbers[0]), int(numbers[1]))
                        state = 2
                elif line[0] == 'sitemap':
                    self.sitemaps.append(line[1])
        if state == 2:
            self._add_entry(entry)

    def can_fetch(self, useragent, url):
        if False:
            while True:
                i = 10
        'using the parsed robots.txt decide if useragent can fetch url'
        if self.disallow_all:
            return False
        if self.allow_all:
            return True
        if not self.last_checked:
            return False
        parsed_url = urllib.parse.urlparse(urllib.parse.unquote(url))
        url = urllib.parse.urlunparse(('', '', parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
        url = urllib.parse.quote(url)
        if not url:
            url = '/'
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.allowance(url)
        if self.default_entry:
            return self.default_entry.allowance(url)
        return True

    def crawl_delay(self, useragent):
        if False:
            print('Hello World!')
        if not self.mtime():
            return None
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.delay
        if self.default_entry:
            return self.default_entry.delay
        return None

    def request_rate(self, useragent):
        if False:
            for i in range(10):
                print('nop')
        if not self.mtime():
            return None
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.req_rate
        if self.default_entry:
            return self.default_entry.req_rate
        return None

    def site_maps(self):
        if False:
            while True:
                i = 10
        if not self.sitemaps:
            return None
        return self.sitemaps

    def __str__(self):
        if False:
            i = 10
            return i + 15
        entries = self.entries
        if self.default_entry is not None:
            entries = entries + [self.default_entry]
        return '\n\n'.join(map(str, entries))

class RuleLine:
    """A rule line is a single "Allow:" (allowance==True) or "Disallow:"
       (allowance==False) followed by a path."""

    def __init__(self, path, allowance):
        if False:
            for i in range(10):
                print('nop')
        if path == '' and (not allowance):
            allowance = True
        path = urllib.parse.urlunparse(urllib.parse.urlparse(path))
        self.path = urllib.parse.quote(path)
        self.allowance = allowance

    def applies_to(self, filename):
        if False:
            while True:
                i = 10
        return self.path == '*' or filename.startswith(self.path)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return ('Allow' if self.allowance else 'Disallow') + ': ' + self.path

class Entry:
    """An entry has one or more user-agents and zero or more rulelines"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.useragents = []
        self.rulelines = []
        self.delay = None
        self.req_rate = None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        for agent in self.useragents:
            ret.append(f'User-agent: {agent}')
        if self.delay is not None:
            ret.append(f'Crawl-delay: {self.delay}')
        if self.req_rate is not None:
            rate = self.req_rate
            ret.append(f'Request-rate: {rate.requests}/{rate.seconds}')
        ret.extend(map(str, self.rulelines))
        return '\n'.join(ret)

    def applies_to(self, useragent):
        if False:
            print('Hello World!')
        'check if this entry applies to the specified agent'
        useragent = useragent.split('/')[0].lower()
        for agent in self.useragents:
            if agent == '*':
                return True
            agent = agent.lower()
            if agent in useragent:
                return True
        return False

    def allowance(self, filename):
        if False:
            print('Hello World!')
        'Preconditions:\n        - our agent applies to this entry\n        - filename is URL decoded'
        for line in self.rulelines:
            if line.applies_to(filename):
                return line.allowance
        return True