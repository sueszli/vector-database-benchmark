"""Implementation of the ipaddres-based network types adaptation
"""
from psycopg2.extensions import new_type, new_array_type, register_type, register_adapter, QuotedString
ipaddress = None
_casters = None

def register_ipaddress(conn_or_curs=None):
    if False:
        i = 10
        return i + 15
    '\n    Register conversion support between `ipaddress` objects and `network types`__.\n\n    :param conn_or_curs: the scope where to register the type casters.\n        If `!None` register them globally.\n\n    After the function is called, PostgreSQL :sql:`inet` values will be\n    converted into `~ipaddress.IPv4Interface` or `~ipaddress.IPv6Interface`\n    objects, :sql:`cidr` values into into `~ipaddress.IPv4Network` or\n    `~ipaddress.IPv6Network`.\n\n    .. __: https://www.postgresql.org/docs/current/static/datatype-net-types.html\n    '
    global ipaddress
    import ipaddress
    global _casters
    if _casters is None:
        _casters = _make_casters()
    for c in _casters:
        register_type(c, conn_or_curs)
    for t in [ipaddress.IPv4Interface, ipaddress.IPv6Interface, ipaddress.IPv4Network, ipaddress.IPv6Network]:
        register_adapter(t, adapt_ipaddress)

def _make_casters():
    if False:
        i = 10
        return i + 15
    inet = new_type((869,), 'INET', cast_interface)
    ainet = new_array_type((1041,), 'INET[]', inet)
    cidr = new_type((650,), 'CIDR', cast_network)
    acidr = new_array_type((651,), 'CIDR[]', cidr)
    return [inet, ainet, cidr, acidr]

def cast_interface(s, cur=None):
    if False:
        i = 10
        return i + 15
    if s is None:
        return None
    return ipaddress.ip_interface(str(s))

def cast_network(s, cur=None):
    if False:
        while True:
            i = 10
    if s is None:
        return None
    return ipaddress.ip_network(str(s))

def adapt_ipaddress(obj):
    if False:
        return 10
    return QuotedString(str(obj))