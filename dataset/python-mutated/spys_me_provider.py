import re
from pyquery import PyQuery
from scylla.database import ProxyIP
from scylla.providers import BaseProvider

class SpyMeProvider(BaseProvider):

    def urls(self) -> [str]:
        if False:
            while True:
                i = 10
        return ['https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list.txt']

    def parse(self, document: PyQuery) -> [ProxyIP]:
        if False:
            for i in range(10):
                print('nop')
        ip_list: [ProxyIP] = []
        text = document.html()
        ip_port_str_list = re.findall('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}:\\d{2,5}', text)
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