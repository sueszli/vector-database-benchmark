import collections
import re
from typing import List
from typing import cast
Parsed = collections.namedtuple('Parsed', ['pathname', 'protocols', 'protocol', 'href', 'resource', 'user', 'port', 'name', 'owner'])
POSSIBLE_REGEXES = (re.compile('^(?P<protocol>https?|git|ssh|rsync)\\://(?:(?P<user>[^\\n@]+)@)*(?P<resource>[a-z0-9_.-]*)[:/]*(?P<port>(?<=:)[\\d]+){0,1}(?P<pathname>\\/((?P<owner>[\\w\\-%\\/]+)\\/)?((?P<name>[\\w\\-%\\.]+?)(\\.git|\\/)?)?)$'), re.compile('(git\\+)?((?P<protocol>\\w+)://)((?P<user>\\w+)@)?((?P<resource>[\\w\\.\\-]+))(:(?P<port>\\d+))?(?P<pathname>(\\/(?P<owner>\\w+)/)?(\\/?(?P<name>[\\w\\-]+)(\\.git|\\/)?)?)$'), re.compile('^(?:(?P<user>[^\\n@]+)@)*(?P<resource>[a-z0-9_.-]*)[:]*(?P<port>(?<=:)[\\d]+){0,1}(?P<pathname>\\/?(?P<owner>.+)/(?P<name>.+).git)$'), re.compile('((?P<user>\\w+)@)?((?P<resource>[\\w\\.\\-]+))[\\:\\/]{1,2}(?P<pathname>((?P<owner>([\\w\\-]+\\/)?\\w+)/)?((?P<name>[\\w\\-]+)(\\.git|\\/)?)?)$'), re.compile('((?P<user>\\w+)@)?((?P<resource>[\\w\\.\\-]+))[\\:\\/]{1,2}(?P<pathname>((?P<owner>\\w+)/)?((?P<name>[\\w\\-\\.]+)(\\.git|\\/)?)?)$'))

class ParserError(Exception):
    """ Error raised when a URL can't be parsed. """
    pass

class Parser(str):
    """
    A class responsible for parsing a GIT URL and return a `Parsed` object.
    """

    def __init__(self, url: str):
        if False:
            i = 10
            return i + 15
        self._url: str = url
        if url[-1] == '/':
            self._url = url[:-1]

    def parse(self) -> Parsed:
        if False:
            while True:
                i = 10
        '\n        Parses a GIT URL and returns an object.  Raises an exception on invalid\n        URL.\n        :returns: Parsed object\n        :raise: :class:`.ParserError`\n        '
        d = {'pathname': None, 'protocols': self._get_protocols(), 'protocol': 'ssh', 'href': self._url, 'resource': None, 'user': None, 'port': None, 'name': None, 'owner': None}
        if len(self._url) > 1024:
            msg = f'URL exceeds maximum supported length of 1024: {self._url}'
            raise ParserError(msg)
        for regex in POSSIBLE_REGEXES:
            match = regex.search(self._url)
            if match:
                d.update(match.groupdict())
                break
        else:
            msg = "Invalid URL '{}'".format(self._url)
            raise ParserError(msg)
        if d['owner'] is not None and cast(str, d['owner']).endswith('/_git'):
            d['owner'] = d['owner'][:-len('/_git')]
        return Parsed(**d)

    def _get_protocols(self) -> List[str]:
        if False:
            while True:
                i = 10
        try:
            index = self._url.index('://')
        except ValueError:
            return []
        return self._url[:index].split('+')