from pyquery import PyQuery
from scylla.database import ProxyIP
from scylla.providers import BaseProvider

class KuaidailiProvider(BaseProvider):

    def urls(self) -> [str]:
        if False:
            i = 10
            return i + 15
        return ['https://www.kuaidaili.com/free/', 'https://www.kuaidaili.com/free/inha/2/']

    def parse(self, document: PyQuery) -> [ProxyIP]:
        if False:
            return 10
        ip_list: [ProxyIP] = []
        for ip_row in document.find('#list table tr'):
            ip_row: PyQuery = ip_row
            ip_element = ip_row.find('td[data-title="IP"]')
            port_element = ip_row.find('td[data-title="PORT"]')
            if ip_element and port_element:
                p = ProxyIP(ip=ip_element.text(), port=port_element.text())
                ip_list.append(p)
        return ip_list

    @staticmethod
    def should_render_js() -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True