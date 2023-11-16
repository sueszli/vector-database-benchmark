"""Serves as an abstraction layer in front of iptables."""
from __future__ import absolute_import, division, print_function, unicode_literals
from wifiphisher.common.constants import NETWORK_GW_IP, PORT, SSL_PORT
from wifiphisher.common.utilities import execute_commands

class Fw:
    """Handles all iptables operations."""

    @staticmethod
    def nat(internal_interface, external_interface):
        if False:
            while True:
                i = 10
        'Do NAT.'
        execute_commands(['iptables -t nat -A POSTROUTING -o {} -j MASQUERADE'.format(external_interface), 'iptables -A FORWARD -i {} -o {} -j ACCEPT'.format(internal_interface, external_interface)])

    @staticmethod
    def clear_rules():
        if False:
            for i in range(10):
                print('nop')
        'Clear all rules.'
        execute_commands(['iptables -F', 'iptables -X', 'iptables -t nat -F', 'iptables -t nat -X'])

    @staticmethod
    def redirect_requests_localhost():
        if False:
            print('Hello World!')
        'Redirect HTTP, HTTPS & DNS requests to localhost.\n\n        Redirect the following requests to localhost:\n            * HTTP (Port 80)\n            * HTTPS (Port 443)\n            * DNS (Port 53)\n        '
        execute_commands(['iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination {}:{}'.format(NETWORK_GW_IP, PORT), 'iptables -t nat -A PREROUTING -p udp --dport 53 -j DNAT --to-destination {}:{}'.format(NETWORK_GW_IP, 53), 'iptables -t nat -A PREROUTING -p tcp --dport 53 -j DNAT --to-destination {}:{}'.format(NETWORK_GW_IP, 53), 'iptables -t nat -A PREROUTING -p tcp --dport 443 -j DNAT --to-destination {}:{}'.format(NETWORK_GW_IP, SSL_PORT)])

    def on_exit(self):
        if False:
            return 10
        'Start the clean up.'
        self.clear_rules()