import re
from pyquery import PyQuery
from scylla.database import ProxyIP
from scylla.providers import BaseProvider

class SpysOneProvider(BaseProvider):

    def urls(self) -> [str]:
        if False:
            i = 10
            return i + 15
        return ['http://spys.one/en/anonymous-proxy-list/']

    def parse(self, document: PyQuery) -> [ProxyIP]:
        if False:
            for i in range(10):
                print('nop')
        ip_list: [ProxyIP] = []
        for ip_row in document.find('table tr[onmouseover]'):
            ip_row: PyQuery = ip_row
            ip_port_text_elem = ip_row.find('.spy14')
            if ip_port_text_elem:
                ip_port_text = ip_port_text_elem.text()
                ip = re.search('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', ip_port_text).group(0)
                port = re.search(':\\n(\\d{2,5})', ip_port_text).group(1)
                if ip and port:
                    p = ProxyIP(ip=ip, port=port)
                    ip_list.append(p)
        return ip_list

    @staticmethod
    def should_render_js() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True