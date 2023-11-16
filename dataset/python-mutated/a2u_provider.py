import re
from pyquery import PyQuery
from scylla.database import ProxyIP
from .base_provider import BaseProvider

class A2uProvider(BaseProvider):

    def urls(self) -> [str]:
        if False:
            i = 10
            return i + 15
        return ['https://raw.githubusercontent.com/a2u/free-proxy-list/master/free-proxy-list.txt']

    def parse(self, document: PyQuery) -> [ProxyIP]:
        if False:
            while True:
                i = 10
        ip_list: [ProxyIP] = []
        raw_html = document.html()
        ip_port_str_list = re.findall('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}:\\d{2,5}', raw_html)
        for ip_port in ip_port_str_list:
            ip = re.search('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', ip_port).group(0)
            port = re.search(':(\\d{2,5})', ip_port).group(1)
            if ip and port:
                p = ProxyIP(ip=ip, port=port)
                ip_list.append(p)
        return ip_list

    @staticmethod
    def should_render_js() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False