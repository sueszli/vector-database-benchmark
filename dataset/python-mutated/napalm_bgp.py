"""
NAPALM BGP
==========

Manages BGP configuration on network devices and provides statistics.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net> & Jerome Fleury <jf@cloudflare.com>
:maturity:   new
:depends:    napalm
:platform:   unix

Dependencies
------------

- :mod:`napalm proxy minion <salt.proxy.napalm>`

.. versionadded:: 2016.11.0
"""
import logging
import salt.utils.napalm
from salt.utils.napalm import proxy_napalm_wrap
log = logging.getLogger(__file__)
__virtualname__ = 'bgp'
__proxyenabled__ = ['napalm']
__virtual_aliases__ = ('napalm_bgp',)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    NAPALM library must be installed for this module to work and run in a (proxy) minion.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

@proxy_napalm_wrap
def config(group=None, neighbor=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Provides the BGP configuration on the device.\n\n    :param group: Name of the group selected to display the configuration.\n    :param neighbor: IP Address of the neighbor to display the configuration.\n        If the group parameter is not specified, the neighbor setting will be\n        ignored.\n\n    :return: A dictionary containing the BGP configuration from the network\n        device. The keys of the main dictionary are the group names.\n\n    Each group has the following properties:\n\n        * type (string)\n        * description (string)\n        * apply_groups (string list)\n        * multihop_ttl (int)\n        * multipath (True/False)\n        * local_address (string)\n        * local_as (int)\n        * remote_as (int)\n        * import_policy (string)\n        * export_policy (string)\n        * remove_private_as (True/False)\n        * prefix_limit (dictionary)\n        * neighbors (dictionary)\n\n    Each neighbor in the dictionary of neighbors provides:\n\n        * description (string)\n        * import_policy (string)\n        * export_policy (string)\n        * local_address (string)\n        * local_as (int)\n        * remote_as (int)\n        * authentication_key (string)\n        * prefix_limit (dictionary)\n        * route_reflector_client (True/False)\n        * nhs (True/False)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bgp.config # entire BGP config\n        salt '*' bgp.config PEERS-GROUP-NAME # provides detail only about BGP group PEERS-GROUP-NAME\n        salt '*' bgp.config PEERS-GROUP-NAME 172.17.17.1 # provides details only about BGP neighbor 172.17.17.1,\n        # configured in the group PEERS-GROUP-NAME\n\n    Output Example:\n\n    .. code-block:: python\n\n        {\n            'PEERS-GROUP-NAME':{\n                'type'          : 'external',\n                'description'   : 'Here we should have a nice description',\n                'apply_groups'  : ['BGP-PREFIX-LIMIT'],\n                'import_policy' : 'PUBLIC-PEER-IN',\n                'export_policy' : 'PUBLIC-PEER-OUT',\n                'remove_private': True,\n                'multipath'     : True,\n                'multihop_ttl'  : 30,\n                'neighbors'     : {\n                    '192.168.0.1': {\n                        'description'   : 'Facebook [CDN]',\n                        'prefix_limit'  : {\n                            'inet': {\n                                'unicast': {\n                                    'limit': 100,\n                                    'teardown': {\n                                        'threshold' : 95,\n                                        'timeout'   : 5\n                                    }\n                                }\n                            }\n                        }\n                        'peer-as'        : 32934,\n                        'route_reflector': False,\n                        'nhs'            : True\n                    },\n                    '172.17.17.1': {\n                        'description'   : 'Twitter [CDN]',\n                        'prefix_limit'  : {\n                            'inet': {\n                                'unicast': {\n                                    'limit': 500,\n                                    'no-validate': 'IMPORT-FLOW-ROUTES'\n                                }\n                            }\n                        }\n                        'peer_as'        : 13414\n                        'route_reflector': False,\n                        'nhs'            : False\n                    }\n                }\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_bgp_config', **{'group': group, 'neighbor': neighbor})

@proxy_napalm_wrap
def neighbors(neighbor=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Provides details regarding the BGP sessions configured on the network device.\n\n    :param neighbor: IP Address of a specific neighbor.\n\n    :return: A dictionary with the statistics of the all/selected BGP\n        neighbors. Outer dictionary keys represent the VRF name.  Keys of inner\n        dictionary represent the AS numbers, while the values are lists of\n        dictionaries, having the following keys:\n\n        - up (True/False)\n        - local_as (int)\n        - remote_as (int)\n        - local_address (string)\n        - routing_table (string)\n        - local_address_configured (True/False)\n        - local_port (int)\n        - remote_address (string)\n        - remote_port (int)\n        - multihop (True/False)\n        - multipath (True/False)\n        - remove_private_as (True/False)\n        - import_policy (string)\n        - export_policy (string)\n        - input_messages (int)\n        - output_messages (int)\n        - input_updates (int)\n        - output_updates (int)\n        - messages_queued_out (int)\n        - connection_state (string)\n        - previous_connection_state (string)\n        - last_event (string)\n        - suppress_4byte_as (True/False)\n        - local_as_prepend (True/False)\n        - holdtime (int)\n        - configured_holdtime (int)\n        - keepalive (int)\n        - configured_keepalive (int)\n        - active_prefix_count (int)\n        - received_prefix_count (int)\n        - accepted_prefix_count (int)\n        - suppressed_prefix_count (int)\n        - advertised_prefix_count (int)\n        - flap_count (int)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' bgp.neighbors  # all neighbors\n        salt '*' bgp.neighbors 172.17.17.1  # only session with BGP neighbor(s) 172.17.17.1\n\n    Output Example:\n\n    .. code-block:: python\n\n        {\n            'default': {\n                8121: [\n                    {\n                        'up'                        : True,\n                        'local_as'                  : 13335,\n                        'remote_as'                 : 8121,\n                        'local_address'             : '172.101.76.1',\n                        'local_address_configured'  : True,\n                        'local_port'                : 179,\n                        'remote_address'            : '192.247.78.0',\n                        'router_id'                 : '192.168.0.1',\n                        'remote_port'               : 58380,\n                        'multihop'                  : False,\n                        'import_policy'             : '4-NTT-TRANSIT-IN',\n                        'export_policy'             : '4-NTT-TRANSIT-OUT',\n                        'input_messages'            : 123,\n                        'output_messages'           : 13,\n                        'input_updates'             : 123,\n                        'output_updates'            : 5,\n                        'messages_queued_out'       : 23,\n                        'connection_state'          : 'Established',\n                        'previous_connection_state' : 'EstabSync',\n                        'last_event'                : 'RecvKeepAlive',\n                        'suppress_4byte_as'         : False,\n                        'local_as_prepend'          : False,\n                        'holdtime'                  : 90,\n                        'configured_holdtime'       : 90,\n                        'keepalive'                 : 30,\n                        'configured_keepalive'      : 30,\n                        'active_prefix_count'       : 132808,\n                        'received_prefix_count'     : 566739,\n                        'accepted_prefix_count'     : 566479,\n                        'suppressed_prefix_count'   : 0,\n                        'advertise_prefix_count'    : 0,\n                        'flap_count'                : 27\n                    }\n                ]\n            }\n        }\n    "
    return salt.utils.napalm.call(napalm_device, 'get_bgp_neighbors_detail', **{'neighbor_address': neighbor})