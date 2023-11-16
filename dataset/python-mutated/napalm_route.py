"""
NAPALM Route
============

Retrieves route details from network devices.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------
- :mod:`NAPALM proxy minion <salt.proxy.napalm>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
log = logging.getLogger(__file__)
__virtualname__ = 'route'
__proxyenabled__ = ['napalm']
__virtual_aliases__ = ('napalm_route',)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

@proxy_napalm_wrap
def show(destination, protocol=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Displays all details for a certain route learned via a specific protocol.\n    If the protocol is not specified, will return all possible routes.\n\n    .. note::\n\n        This function return the routes from the RIB.\n        In case the destination prefix is too short,\n        there may be too many routes matched.\n        Therefore in cases of devices having a very high number of routes\n        it may be necessary to adjust the prefix length and request\n        using a longer prefix.\n\n    destination\n        destination prefix.\n\n    protocol (optional)\n        protocol used to learn the routes to the destination.\n\n    .. versionchanged:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my_router' route.show 172.16.0.0/25\n        salt 'my_router' route.show 172.16.0.0/25 bgp\n\n    Output example:\n\n    .. code-block:: python\n\n        {\n            '172.16.0.0/25': [\n                {\n                    'protocol': 'BGP',\n                    'last_active': True,\n                    'current_active': True,\n                    'age': 1178693,\n                    'routing_table': 'inet.0',\n                    'next_hop': '192.168.0.11',\n                    'outgoing_interface': 'xe-1/1/1.100',\n                    'preference': 170,\n                    'selected_next_hop': False,\n                    'protocol_attributes': {\n                        'remote_as': 65001,\n                        'metric': 5,\n                        'local_as': 13335,\n                        'as_path': '',\n                        'remote_address': '192.168.0.11',\n                        'metric2': 0,\n                        'local_preference': 0,\n                        'communities': [\n                            '0:2',\n                            'no-export'\n                        ],\n                        'preference2': -1\n                    },\n                    'inactive_reason': ''\n                },\n                {\n                    'protocol': 'BGP',\n                    'last_active': False,\n                    'current_active': False,\n                    'age': 2359429,\n                    'routing_table': 'inet.0',\n                    'next_hop': '192.168.0.17',\n                    'outgoing_interface': 'xe-1/1/1.100',\n                    'preference': 170,\n                    'selected_next_hop': True,\n                    'protocol_attributes': {\n                        'remote_as': 65001,\n                        'metric': 5,\n                        'local_as': 13335,\n                        'as_path': '',\n                        'remote_address': '192.168.0.17',\n                        'metric2': 0,\n                        'local_preference': 0,\n                        'communities': [\n                            '0:3',\n                            'no-export'\n                        ],\n                        'preference2': -1\n                    },\n                    'inactive_reason': 'Not Best in its group - Router ID'\n                }\n            ]\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_route_to', **{'destination': destination, 'protocol': protocol})