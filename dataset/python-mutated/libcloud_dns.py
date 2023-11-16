"""
Apache Libcloud DNS Management
==============================

Connection module for Apache Libcloud DNS management

.. versionadded:: 2016.11.0

:configuration:
    This module uses a configuration profile for one or multiple DNS providers

    .. code-block:: yaml

        libcloud_dns:
            profile_test1:
              driver: cloudflare
              key: 12345
              secret: mysecret
            profile_test2:
              driver: godaddy
              key: 12345
              secret: mysecret
              shopper_id: 12345

:depends: apache-libcloud
"""
import logging
from salt.utils.versions import Version
log = logging.getLogger(__name__)
REQUIRED_LIBCLOUD_VERSION = '2.0.0'
try:
    import libcloud
    from libcloud.dns.providers import get_driver
    from libcloud.dns.types import RecordType
    if hasattr(libcloud, '__version__') and Version(libcloud.__version__) < Version(REQUIRED_LIBCLOUD_VERSION):
        raise ImportError()
    logging.getLogger('libcloud').setLevel(logging.CRITICAL)
    HAS_LIBCLOUD = True
except ImportError:
    HAS_LIBCLOUD = False

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if libcloud libraries exist.\n    '
    if not HAS_LIBCLOUD:
        return (False, 'A apache-libcloud library with version at least {} was not found'.format(REQUIRED_LIBCLOUD_VERSION))
    return True

def _get_driver(profile):
    if False:
        return 10
    config = __salt__['config.option']('libcloud_dns')[profile]
    cls = get_driver(config['driver'])
    args = config.copy()
    del args['driver']
    args['key'] = config.get('key')
    args['secret'] = config.get('secret', None)
    args['secure'] = config.get('secure', True)
    args['host'] = config.get('host', None)
    args['port'] = config.get('port', None)
    return cls(**args)

def list_record_types(profile):
    if False:
        return 10
    '\n    List available record types for the given profile, e.g. A, AAAA\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.list_record_types profile1\n    '
    conn = _get_driver(profile=profile)
    return conn.list_record_types()

def list_zones(profile):
    if False:
        print('Hello World!')
    '\n    List zones for the given profile\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.list_zones profile1\n    '
    conn = _get_driver(profile=profile)
    return [_simple_zone(zone) for zone in conn.list_zones()]

def list_records(zone_id, profile, type=None):
    if False:
        return 10
    '\n    List records for the given zone_id on the given profile\n\n    :param zone_id: Zone to export.\n    :type  zone_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param type: The record type, e.g. A, NS\n    :type  type: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.list_records google.com profile1\n    '
    conn = _get_driver(profile=profile)
    zone = conn.get_zone(zone_id)
    if type is not None:
        return [_simple_record(record) for record in conn.list_records(zone) if record.type == type]
    else:
        return [_simple_record(record) for record in conn.list_records(zone)]

def get_zone(zone_id, profile):
    if False:
        while True:
            i = 10
    '\n    Get zone information for the given zone_id on the given profile\n\n    :param zone_id: Zone to export.\n    :type  zone_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.get_zone google.com profile1\n    '
    conn = _get_driver(profile=profile)
    return _simple_zone(conn.get_zone(zone_id))

def get_record(zone_id, record_id, profile):
    if False:
        return 10
    '\n    Get record information for the given zone_id on the given profile\n\n    :param zone_id: Zone to export.\n    :type  zone_id: ``str``\n\n    :param record_id: Record to delete.\n    :type  record_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.get_record google.com www profile1\n    '
    conn = _get_driver(profile=profile)
    return _simple_record(conn.get_record(zone_id, record_id))

def create_zone(domain, profile, type='master', ttl=None):
    if False:
        print('Hello World!')
    '\n    Create a new zone.\n\n    :param domain: Zone domain name (e.g. example.com)\n    :type domain: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param type: Zone type (master / slave).\n    :type  type: ``str``\n\n    :param ttl: TTL for new records. (optional)\n    :type  ttl: ``int``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.create_zone google.com profile1\n    '
    conn = _get_driver(profile=profile)
    zone = conn.create_record(domain, type=type, ttl=ttl)
    return _simple_zone(zone)

def update_zone(zone_id, domain, profile, type='master', ttl=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update an existing zone.\n\n    :param zone_id: Zone ID to update.\n    :type  zone_id: ``str``\n\n    :param domain: Zone domain name (e.g. example.com)\n    :type  domain: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param type: Zone type (master / slave).\n    :type  type: ``str``\n\n    :param ttl: TTL for new records. (optional)\n    :type  ttl: ``int``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.update_zone google.com google.com profile1 type=slave\n    '
    conn = _get_driver(profile=profile)
    zone = conn.get_zone(zone_id)
    return _simple_zone(conn.update_zone(zone=zone, domain=domain, type=type, ttl=ttl))

def create_record(name, zone_id, type, data, profile):
    if False:
        return 10
    "\n    Create a new record.\n\n    :param name: Record name without the domain name (e.g. www).\n                 Note: If you want to create a record for a base domain\n                 name, you should specify empty string ('') for this\n                 argument.\n    :type  name: ``str``\n\n    :param zone_id: Zone where the requested record is created.\n    :type  zone_id: ``str``\n\n    :param type: DNS record type (A, AAAA, ...).\n    :type  type: ``str``\n\n    :param data: Data for the record (depends on the record type).\n    :type  data: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.create_record www google.com A 12.32.12.2 profile1\n    "
    conn = _get_driver(profile=profile)
    record_type = _string_to_record_type(type)
    zone = conn.get_zone(zone_id)
    return _simple_record(conn.create_record(name, zone, record_type, data))

def delete_zone(zone_id, profile):
    if False:
        print('Hello World!')
    '\n    Delete a zone.\n\n    :param zone_id: Zone to delete.\n    :type  zone_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.delete_zone google.com profile1\n    '
    conn = _get_driver(profile=profile)
    zone = conn.get_zone(zone_id=zone_id)
    return conn.delete_zone(zone)

def delete_record(zone_id, record_id, profile):
    if False:
        while True:
            i = 10
    '\n    Delete a record.\n\n    :param zone_id: Zone to delete.\n    :type  zone_id: ``str``\n\n    :param record_id: Record to delete.\n    :type  record_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :rtype: ``bool``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.delete_record google.com www profile1\n    '
    conn = _get_driver(profile=profile)
    record = conn.get_record(zone_id=zone_id, record_id=record_id)
    return conn.delete_record(record)

def get_bind_data(zone_id, profile):
    if False:
        for i in range(10):
            print('nop')
    '\n    Export Zone to the BIND compatible format.\n\n    :param zone_id: Zone to export.\n    :type  zone_id: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :return: Zone data in BIND compatible format.\n    :rtype: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.get_bind_data google.com profile1\n    '
    conn = _get_driver(profile=profile)
    zone = conn.get_zone(zone_id)
    return conn.export_zone_to_bind_format(zone)

def extra(method, profile, **libcloud_kwargs):
    if False:
        print('Hello World!')
    "\n    Call an extended method on the driver\n\n    :param method: Driver's method name\n    :type  method: ``str``\n\n    :param profile: The profile key\n    :type  profile: ``str``\n\n    :param libcloud_kwargs: Extra arguments for the driver's delete_container method\n    :type  libcloud_kwargs: ``dict``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion libcloud_dns.extra ex_get_permissions google container_name=my_container object_name=me.jpg --out=yaml\n    "
    _sanitize_kwargs(libcloud_kwargs)
    conn = _get_driver(profile=profile)
    connection_method = getattr(conn, method)
    return connection_method(**libcloud_kwargs)

def _string_to_record_type(string):
    if False:
        while True:
            i = 10
    '\n    Return a string representation of a DNS record type to a\n    libcloud RecordType ENUM.\n\n    :param string: A record type, e.g. A, TXT, NS\n    :type  string: ``str``\n\n    :rtype: :class:`RecordType`\n    '
    string = string.upper()
    record_type = getattr(RecordType, string)
    return record_type

def _simple_zone(zone):
    if False:
        return 10
    return {'id': zone.id, 'domain': zone.domain, 'type': zone.type, 'ttl': zone.ttl, 'extra': zone.extra}

def _simple_record(record):
    if False:
        print('Hello World!')
    return {'id': record.id, 'name': record.name, 'type': record.type, 'data': record.data, 'zone': _simple_zone(record.zone), 'ttl': record.ttl, 'extra': record.extra}