from pyquery import PyQuery
from ..database import ProxyIP

class BaseProvider(object):
    """BaseProvider is the abstract class for the proxy providers

    :raises NotImplementedError: [if urls() or parse() is not implemented]
    """
    _sleep = 0

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__

    def sleep_seconds(self) -> int:
        if False:
            while True:
                i = 10
        'Return a sleep time for each request, by default it is 0\n\n        :return: sleep time in seconds\n        '
        return self._sleep

    def urls(self) -> [str]:
        if False:
            print('Hello World!')
        'Return a list of url strings for crawling\n\n        :return: [a list of url strings]\n        :rtype: [str]\n        '
        raise NotImplementedError

    def parse(self, document: PyQuery) -> [ProxyIP]:
        if False:
            while True:
                i = 10
        'Parse the document in order to get a list of proxies\n\n        :param document: the HTML object from requests-html\n        :return: a list of proxy ips\n        '
        raise NotImplementedError

    @staticmethod
    def should_render_js() -> bool:
        if False:
            i = 10
            return i + 15
        'Whether needs js rendering\n        By default, it is False.\n\n        :return: a boolean value indicating whether or not js rendering is needed\n        :rtype: bool\n        '
        return False