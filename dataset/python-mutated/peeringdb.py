"""
PeeringDB Module
================

.. versionadded:: 2019.2.0

Execution module for the basic interaction with the
`PeeringDB <https://www.peeringdb.com/>`_ API.

While for GET operations (the functions prefixed by ``get_``) the credentials
are optional, there are some specific details that are visible only to
authenticated users. Moreover, the credentials are required when adding or
updating information. That means, the module can equally work out of the box
without any further configuration with the limitations imposed by the PeeringDB
API.

For complete API documentation, please refer to https://www.peeringdb.com/apidocs/.

Configuration (in the opts or Pillar):

.. code-block:: yaml

    peeringdb:
      username: salt
      password: 5@1t
"""
import logging
import salt.utils.http
from salt.utils.args import clean_kwargs
log = logging.getLogger(__name__)
__virtualname__ = 'peeringdb'
__proxyenabled__ = ['*']
PEERINGDB_URL = 'https://www.peeringdb.com/api'

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__

def _get_auth(username=None, password=None):
    if False:
        print('Hello World!')
    peeringdb_cfg = __salt__['config.merge']('peeringdb', default={})
    if not username:
        username = peeringdb_cfg.get('username', username)
    if not password:
        password = peeringdb_cfg.get('password', password)
    return (username, password)

def _build_url(endpoint, id=None):
    if False:
        print('Hello World!')
    if id:
        return '{base}/{endp}/{id}'.format(base=PEERINGDB_URL, endp=endpoint, id=id)
    return '{base}/{endp}'.format(base=PEERINGDB_URL, endp=endpoint)

def _get_endpoint(endpoint, id=None, **kwargs):
    if False:
        while True:
            i = 10
    (username, password) = _get_auth(kwargs.pop('username', None), kwargs.pop('password', None))
    kwargs = clean_kwargs(**kwargs)
    url = _build_url(endpoint, id=id)
    ret = {'comment': '', 'result': True, 'out': None}
    res = salt.utils.http.query(url, method='GET', decode=True, username=username, password=password, params=kwargs)
    if 'error' in res:
        ret.update({'result': False, 'comment': res['error']})
        return ret
    ret['out'] = res['dict']['data']
    return ret

def get_net(**kwargs):
    if False:
        print('Hello World!')
    "\n    Return the details of a network identified using the search filters\n    specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible networks registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/net/net_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_net id=4224\n        salt '*' peeringdb.get_net asn=13335\n        salt '*' peeringdb.get_net city='Salt Lake City'\n        salt '*' peeringdb.get_net name__startswith=GTT\n    "
    return _get_endpoint('net', **kwargs)

def get_fac(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the details of the facility identified using the search\n    filters specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible facilities registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/netfac/netfac_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_fac id=1774\n        salt '*' peeringdb.get_fac state=UT\n    "
    return _get_endpoint('fac', **kwargs)

def get_ix(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the details of an IX (Internet Exchange) using the search filters\n    specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible IXs registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/ix/ix_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_ix id=1\n        salt '*' peeringdb.get_ix city='Milwaukee'\n    "
    return _get_endpoint('ix', **kwargs)

def get_ixfac(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the details of an IX (Internet Exchange) facility using the search\n    filters specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible IX facilities registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/ixfac/ixfac_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_ixfac id=1\n        salt '*' peeringdb.get_ixfac city='Milwaukee'\n    "
    return _get_endpoint('ixfac', **kwargs)

def get_ixlan(**kwargs):
    if False:
        print('Hello World!')
    "\n    Return the details of an IX (Internet Exchange) together with the networks\n    available in this location (and their details), using the search filters\n    specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible IX LAN facilities registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/ixlan/ixlan_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_ixlan id=780\n        salt '*' peeringdb.get_ixlan city='Milwaukee'\n    "
    return _get_endpoint('ixlan', **kwargs)

def get_ixpfx(**kwargs):
    if False:
        return 10
    "\n    Return the details of an IX (Internet Exchange) together with the PeeringDB\n    IDs of the networks available in this location, using the search filters\n    specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible IX LAN facilities registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/ixpfx/ixpfx_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_ixpfx id=780\n        salt '*' peeringdb.get_ixpfx city='Milwaukee'\n    "
    return _get_endpoint('ixpfx', **kwargs)

def get_netfac(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the list of facilities used by a particular network, given the ``id``\n    or other filters specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible network facilities registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/netfac/netfac_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_netfac id=780\n        salt '*' peeringdb.get_netfac city='Milwaukee'\n    "
    return _get_endpoint('netfac', **kwargs)

def get_netixlan(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the IP addresses used by a particular network at all the IXs where it\n    is available. The network is selected either via the ``id`` argument or the\n    other filters specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible IP addresses, of all networks, at all IXs, registered in\n        PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/netixlan/netixlan_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_netixlan asn=13335\n        salt '*' peeringdb.get_netixlan ipaddr4=185.1.114.25\n    "
    return _get_endpoint('netixlan', **kwargs)

def get_org(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the details of an organisation together with the networks\n    available in this location, using the search filters specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible organisations registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/org/org_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_org id=2\n        salt '*' peeringdb.get_org city=Duesseldorf\n    "
    return _get_endpoint('org', **kwargs)

def get_poc(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the details of a person of contact together using the search filters\n    specified in the query.\n\n    .. note::\n        If no ``id`` or filter arguments are specified, it will return all the\n        possible contacts registered in PeeringDB.\n\n        The available filters are documented at:\n        https://www.peeringdb.com/apidocs/#!/poc/poc_list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' peeringdb.get_poc id=6721\n        salt '*' peeringdb.get_poc email__contains='@cloudflare.com'\n    "
    return _get_endpoint('poc', **kwargs)