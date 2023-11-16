"""
NetBox
======

Module to query NetBox

:codeauthor: Zach Moody <zmoody@do.co>
:maturity:   new
:depends:    pynetbox

The following config should be in the minion config file. In order to
work with ``secrets`` you should provide a token and path to your
private key file:

.. code-block:: yaml

  netbox:
    url: <NETBOX_URL>
    token: <NETBOX_USERNAME_API_TOKEN (OPTIONAL)>
    keyfile: </PATH/TO/NETBOX/KEY (OPTIONAL)>

.. versionadded:: 2018.3.0
"""
import logging
import re
from salt.exceptions import CommandExecutionError
try:
    import pynetbox
    HAS_PYNETBOX = True
except ImportError:
    HAS_PYNETBOX = False
log = logging.getLogger(__name__)
AUTH_ENDPOINTS = ('secrets',)
__func_alias__ = {'filter_': 'filter', 'get_': 'get'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    pynetbox must be installed.\n    '
    if not HAS_PYNETBOX:
        return (False, 'The netbox execution module cannot be loaded: pynetbox library is not installed.')
    else:
        return True

def _config():
    if False:
        while True:
            i = 10
    config = __salt__['config.get']('netbox')
    if not config:
        raise CommandExecutionError('NetBox execution module configuration could not be found')
    return config

def _nb_obj(auth_required=False):
    if False:
        while True:
            i = 10
    pynb_kwargs = {}
    pynb_kwargs['token'] = _config().get('token')
    if auth_required:
        pynb_kwargs['private_key_file'] = _config().get('keyfile')
    return pynetbox.api(_config().get('url'), **pynb_kwargs)

def _strip_url_field(input_dict):
    if False:
        while True:
            i = 10
    if 'url' in input_dict.keys():
        del input_dict['url']
    for (k, v) in input_dict.items():
        if isinstance(v, dict):
            _strip_url_field(v)
    return input_dict

def _dict(iterable):
    if False:
        print('Hello World!')
    if iterable:
        return dict(iterable)
    else:
        return {}

def _add(app, endpoint, payload):
    if False:
        while True:
            i = 10
    '\n    POST a payload\n    '
    nb = _nb_obj(auth_required=True)
    try:
        return getattr(getattr(nb, app), endpoint).create(**payload)
    except pynetbox.RequestError as e:
        log.error('%s, %s, %s', e.req.request.headers, e.request_body, e.error)
        return False

def slugify(value):
    if False:
        while True:
            i = 10
    "'\n    Slugify given value.\n    Credit to Djangoproject https://docs.djangoproject.com/en/2.0/_modules/django/utils/text/#slugify\n    "
    value = re.sub('[^\\w\\s-]', '', value).strip().lower()
    return re.sub('[-\\s]+', '-', value)

def _get(app, endpoint, id=None, auth_required=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to do a GET request to Netbox.\n    Returns the actual pynetbox object, which allows manipulation from other functions.\n    '
    nb = _nb_obj(auth_required=auth_required)
    if id:
        item = getattr(getattr(nb, app), endpoint).get(id)
    else:
        kwargs = __utils__['args.clean_kwargs'](**kwargs)
        item = getattr(getattr(nb, app), endpoint).get(**kwargs)
    return item

def _if_name_unit(if_name):
    if False:
        for i in range(10):
            print('nop')
    if_name_split = if_name.split('.')
    if len(if_name_split) == 2:
        return if_name_split
    return (if_name, '0')

def filter_(app, endpoint, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Get a list of items from NetBox.\n\n    app\n        String of netbox app, e.g., ``dcim``, ``circuits``, ``ipam``\n    endpoint\n        String of app endpoint, e.g., ``sites``, ``regions``, ``devices``\n    kwargs\n        Optional arguments that can be used to filter.\n        All filter keywords are available in Netbox,\n        which can be found by surfing to the corresponding API endpoint,\n        and clicking Filters. e.g., ``role=router``\n\n    Returns a list of dictionaries\n\n    .. code-block:: bash\n\n        salt myminion netbox.filter dcim devices status=1 role=router\n    '
    ret = []
    nb = _nb_obj(auth_required=True if app in AUTH_ENDPOINTS else False)
    nb_query = getattr(getattr(nb, app), endpoint).filter(**__utils__['args.clean_kwargs'](**kwargs))
    if nb_query:
        ret = [_strip_url_field(dict(i)) for i in nb_query]
    return ret

def get_(app, endpoint, id=None, **kwargs):
    if False:
        return 10
    '\n    Get a single item from NetBox.\n\n    app\n        String of netbox app, e.g., ``dcim``, ``circuits``, ``ipam``\n    endpoint\n        String of app endpoint, e.g., ``sites``, ``regions``, ``devices``\n\n    Returns a single dictionary\n\n    To get an item based on ID.\n\n    .. code-block:: bash\n\n        salt myminion netbox.get dcim devices id=123\n\n    Or using named arguments that correspond with accepted filters on\n    the NetBox endpoint.\n\n    .. code-block:: bash\n\n        salt myminion netbox.get dcim devices name=my-router\n    '
    return _dict(_get(app, endpoint, id=id, auth_required=True if app in AUTH_ENDPOINTS else False, **kwargs))

def create_manufacturer(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Create a device manufacturer.\n\n    name\n        The name of the manufacturer, e.g., ``Juniper``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_manufacturer Juniper\n    '
    nb_man = get_('dcim', 'manufacturers', name=name)
    if nb_man:
        return False
    else:
        payload = {'name': name, 'slug': slugify(name)}
        man = _add('dcim', 'manufacturers', payload)
        if man:
            return {'dcim': {'manufacturers': payload}}
        else:
            return False

def create_device_type(model, manufacturer):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Create a device type. If the manufacturer doesn't exist, create a new manufacturer.\n\n    model\n        String of device model, e.g., ``MX480``\n    manufacturer\n        String of device manufacturer, e.g., ``Juniper``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_device_type MX480 Juniper\n    "
    nb_type = get_('dcim', 'device-types', model=model)
    if nb_type:
        return False
    nb_man = get_('dcim', 'manufacturers', name=manufacturer)
    new_man = None
    if not nb_man:
        new_man = create_manufacturer(manufacturer)
    payload = {'model': model, 'manufacturer': nb_man['id'], 'slug': slugify(model)}
    typ = _add('dcim', 'device-types', payload)
    ret_dict = {'dcim': {'device-types': payload}}
    if new_man:
        ret_dict['dcim'].update(new_man['dcim'])
    if typ:
        return ret_dict
    else:
        return False

def create_device_role(role, color):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2019.2.0\n\n    Create a device role\n\n    role\n        String of device role, e.g., ``router``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_device_role router\n    '
    nb_role = get_('dcim', 'device-roles', name=role)
    if nb_role:
        return False
    else:
        payload = {'name': role, 'slug': slugify(role), 'color': color}
        role = _add('dcim', 'device-roles', payload)
        if role:
            return {'dcim': {'device-roles': payload}}
        else:
            return False

def create_platform(platform):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new device platform\n\n    platform\n        String of device platform, e.g., ``junos``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_platform junos\n    '
    nb_platform = get_('dcim', 'platforms', slug=slugify(platform))
    if nb_platform:
        return False
    else:
        payload = {'name': platform, 'slug': slugify(platform)}
        plat = _add('dcim', 'platforms', payload)
        if plat:
            return {'dcim': {'platforms': payload}}
        else:
            return False

def create_site(site):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new device site\n\n    site\n        String of device site, e.g., ``BRU``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_site BRU\n    '
    nb_site = get_('dcim', 'sites', name=site)
    if nb_site:
        return False
    else:
        payload = {'name': site, 'slug': slugify(site)}
        site = _add('dcim', 'sites', payload)
        if site:
            return {'dcim': {'sites': payload}}
        else:
            return False

def create_device(name, role, model, manufacturer, site):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new device with a name, role, model, manufacturer and site.\n    All these components need to be already in Netbox.\n\n    name\n        The name of the device, e.g., ``edge_router``\n    role\n        String of device role, e.g., ``router``\n    model\n        String of device model, e.g., ``MX480``\n    manufacturer\n        String of device manufacturer, e.g., ``Juniper``\n    site\n        String of device site, e.g., ``BRU``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_device edge_router router MX480 Juniper BRU\n    '
    try:
        nb_role = get_('dcim', 'device-roles', name=role)
        if not nb_role:
            return False
        nb_type = get_('dcim', 'device-types', model=model)
        if not nb_type:
            return False
        nb_site = get_('dcim', 'sites', name=site)
        if not nb_site:
            return False
        status = {'label': 'Active', 'value': 1}
    except pynetbox.RequestError as e:
        log.error('%s, %s, %s', e.req.request.headers, e.request_body, e.error)
        return False
    payload = {'name': name, 'display_name': name, 'slug': slugify(name), 'device_type': nb_type['id'], 'device_role': nb_role['id'], 'site': nb_site['id']}
    new_dev = _add('dcim', 'devices', payload)
    if new_dev:
        return {'dcim': {'devices': payload}}
    else:
        return False

def update_device(name, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Add attributes to an existing device, identified by name.\n\n    name\n        The name of the device, e.g., ``edge_router``\n    kwargs\n       Arguments to change in device, e.g., ``serial=JN2932930``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.update_device edge_router serial=JN2932920\n    '
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    nb_device = _get('dcim', 'devices', auth_required=True, name=name)
    for (k, v) in kwargs.items():
        setattr(nb_device, k, v)
    try:
        nb_device.save()
        return {'dcim': {'devices': kwargs}}
    except pynetbox.RequestError as e:
        log.error('%s, %s, %s', e.req.request.headers, e.request_body, e.error)
        return False

def create_inventory_item(device_name, item_name, manufacturer_name=None, serial='', part_id='', description=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Add an inventory item to an existing device.\n\n    device_name\n        The name of the device, e.g., ``edge_router``.\n    item_name\n        String of inventory item name, e.g., ``Transceiver``.\n\n    manufacturer_name\n        String of inventory item manufacturer, e.g., ``Fiberstore``.\n\n    serial\n        String of inventory item serial, e.g., ``FS1238931``.\n\n    part_id\n        String of inventory item part id, e.g., ``740-01234``.\n\n    description\n        String of inventory item description, e.g., ``SFP+-10G-LR``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_inventory_item edge_router Transceiver part_id=740-01234\n    '
    nb_device = get_('dcim', 'devices', name=device_name)
    if not nb_device:
        return False
    if manufacturer_name:
        nb_man = get_('dcim', 'manufacturers', name=manufacturer_name)
        if not nb_man:
            create_manufacturer(manufacturer_name)
            nb_man = get_('dcim', 'manufacturers', name=manufacturer_name)
    payload = {'device': nb_device['id'], 'name': item_name, 'description': description, 'serial': serial, 'part_id': part_id, 'parent': None}
    if manufacturer_name:
        payload['manufacturer'] = nb_man['id']
    done = _add('dcim', 'inventory-items', payload)
    if done:
        return {'dcim': {'inventory-items': payload}}
    else:
        return done

def delete_inventory_item(item_id):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Remove an item from a devices inventory. Identified by the netbox id\n\n    item_id\n        Integer of item to be deleted\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.delete_inventory_item 1354\n    '
    nb_inventory_item = _get('dcim', 'inventory-items', auth_required=True, id=item_id)
    nb_inventory_item.delete()
    return {'DELETE': {'dcim': {'inventory-items': item_id}}}

def create_interface_connection(interface_a, interface_b):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Create an interface connection between 2 interfaces\n\n    interface_a\n        Interface id for Side A\n    interface_b\n        Interface id for Side B\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_interface_connection 123 456\n    '
    payload = {'interface_a': interface_a, 'interface_b': interface_b}
    ret = _add('dcim', 'interface-connections', payload)
    if ret:
        return {'dcim': {'interface-connections': {ret['id']: payload}}}
    else:
        return ret

def get_interfaces(device_name=None, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Returns interfaces for a specific device using arbitrary netbox filters\n\n    device_name\n        The name of the device, e.g., ``edge_router``\n    kwargs\n        Optional arguments to be used for filtering\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.get_interfaces edge_router name="et-0/0/5"\n\n    '
    if not device_name:
        device_name = __opts__['id']
    netbox_device = get_('dcim', 'devices', name=device_name)
    return filter_('dcim', 'interfaces', device_id=netbox_device['id'], **kwargs)

def openconfig_interfaces(device_name=None):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return a dictionary structured as standardised in the\n    `openconfig-interfaces <http://ops.openconfig.net/branches/models/master/openconfig-interfaces.html>`_\n    YANG model, containing physical and configuration data available in Netbox,\n    e.g., IP addresses, MTU, enabled / disabled, etc.\n\n    device_name: ``None``\n        The name of the device to query the interface data for. If not provided,\n        will use the Minion ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' netbox.openconfig_interfaces\n        salt '*' netbox.openconfig_interfaces device_name=cr1.thn.lon\n    "
    oc_if = {}
    interfaces = get_interfaces(device_name=device_name)
    ipaddresses = get_ipaddresses(device_name=device_name)
    for interface in interfaces:
        (if_name, if_unit) = _if_name_unit(interface['name'])
        if if_name not in oc_if:
            oc_if[if_name] = {'config': {'name': if_name}, 'subinterfaces': {'subinterface': {}}}
        if if_unit == '0':
            oc_if[if_name]['config']['enabled'] = interface['enabled']
            if interface['description']:
                if if_name == interface['name']:
                    oc_if[if_name]['config']['description'] = str(interface['description'])
                else:
                    subif_descr = {'subinterfaces': {'subinterface': {if_unit: {'config': {'description': str(interface['description'])}}}}}
                    oc_if[if_name] = __utils__['dictupdate.update'](oc_if[if_name], subif_descr)
            if interface['mtu']:
                oc_if[if_name]['config']['mtu'] = int(interface['mtu'])
        else:
            oc_if[if_name]['subinterfaces']['subinterface'][if_unit] = {'config': {'index': int(if_unit), 'enabled': interface['enabled']}}
            if interface['description']:
                oc_if[if_name]['subinterfaces']['subinterface'][if_unit]['config']['description'] = str(interface['description'])
    for ipaddress in ipaddresses:
        (ip, prefix_length) = ipaddress['address'].split('/')
        if_name = ipaddress['interface']['name']
        (if_name, if_unit) = _if_name_unit(if_name)
        ipvkey = 'ipv{}'.format(ipaddress['family'])
        if if_unit not in oc_if[if_name]['subinterfaces']['subinterface']:
            oc_if[if_name]['subinterfaces']['subinterface'][if_unit] = {'config': {'index': int(if_unit), 'enabled': True}}
        if ipvkey not in oc_if[if_name]['subinterfaces']['subinterface'][if_unit]:
            oc_if[if_name]['subinterfaces']['subinterface'][if_unit][ipvkey] = {'addresses': {'address': {}}}
        oc_if[if_name]['subinterfaces']['subinterface'][if_unit][ipvkey]['addresses']['address'][ip] = {'config': {'ip': ip, 'prefix_length': int(prefix_length)}}
    return {'interfaces': {'interface': oc_if}}

def openconfig_lacp(device_name=None):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Return a dictionary structured as standardised in the\n    `openconfig-lacp <http://ops.openconfig.net/branches/models/master/openconfig-lacp.html>`_\n    YANG model, with configuration data for Link Aggregation Control Protocol\n    (LACP) for aggregate interfaces.\n\n    .. note::\n        The ``interval`` and ``lacp_mode`` keys have the values set as ``SLOW``\n        and ``ACTIVE`` respectively, as this data is not currently available\n        in Netbox, therefore defaulting to the values defined in the standard.\n        See `interval <http://ops.openconfig.net/branches/models/master/docs/openconfig-lacp.html#lacp-interfaces-interface-config-interval>`_\n        and `lacp-mode <http://ops.openconfig.net/branches/models/master/docs/openconfig-lacp.html#lacp-interfaces-interface-config-lacp-mode>`_\n        for further details.\n\n    device_name: ``None``\n        The name of the device to query the LACP information for. If not provided,\n        will use the Minion ID.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' netbox.openconfig_lacp\n        salt '*' netbox.openconfig_lacp device_name=cr1.thn.lon\n    "
    oc_lacp = {}
    interfaces = get_interfaces(device_name=device_name)
    for interface in interfaces:
        if not interface['lag']:
            continue
        (if_name, if_unit) = _if_name_unit(interface['name'])
        parent_if = interface['lag']['name']
        if parent_if not in oc_lacp:
            oc_lacp[parent_if] = {'config': {'name': parent_if, 'interval': 'SLOW', 'lacp_mode': 'ACTIVE'}, 'members': {'member': {}}}
        oc_lacp[parent_if]['members']['member'][if_name] = {}
    return {'lacp': {'interfaces': {'interface': oc_lacp}}}

def create_interface(device_name, interface_name, mac_address=None, description=None, enabled=None, lag=None, lag_parent=None, form_factor=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2019.2.0\n\n    Attach an interface to a device. If not all arguments are provided,\n    they will default to Netbox defaults.\n\n    device_name\n        The name of the device, e.g., ``edge_router``\n    interface_name\n        The name of the interface, e.g., ``TenGigE0/0/0/0``\n    mac_address\n        String of mac address, e.g., ``50:87:89:73:92:C8``\n    description\n        String of interface description, e.g., ``NTT``\n    enabled\n        String of boolean interface status, e.g., ``True``\n    lag:\n        Boolean of interface lag status, e.g., ``True``\n    lag_parent\n        String of interface lag parent name, e.g., ``ae13``\n    form_factor\n        Integer of form factor id, obtained through _choices API endpoint, e.g., ``200``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_interface edge_router ae13 description="Core uplink"\n    '
    nb_device = get_('dcim', 'devices', name=device_name)
    if not nb_device:
        return False
    if lag_parent:
        lag_interface = get_('dcim', 'interfaces', device_id=nb_device['id'], name=lag_parent)
        if not lag_interface:
            return False
    if not description:
        description = ''
    if not enabled:
        enabled = 'false'
    payload = {'device': nb_device['id'], 'name': interface_name, 'description': description, 'enabled': enabled, 'form_factor': 1200}
    if form_factor is not None:
        payload['form_factor'] = form_factor
    if lag:
        payload['form_factor'] = 200
    if lag_parent:
        payload['lag'] = lag_interface['id']
    if mac_address:
        payload['mac_address'] = mac_address
    nb_interface = get_('dcim', 'interfaces', device_id=nb_device['id'], name=interface_name)
    if not nb_interface:
        nb_interface = _add('dcim', 'interfaces', payload)
    if nb_interface:
        return {'dcim': {'interfaces': {nb_interface['id']: payload}}}
    else:
        return nb_interface

def update_interface(device_name, interface_name, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2019.2.0\n\n    Update an existing interface with new attributes.\n\n    device_name\n        The name of the device, e.g., ``edge_router``\n    interface_name\n        The name of the interface, e.g., ``ae13``\n    kwargs\n        Arguments to change in interface, e.g., ``mac_address=50:87:69:53:32:D0``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.update_interface edge_router ae13 mac_address=50:87:69:53:32:D0\n    '
    nb_device = get_('dcim', 'devices', name=device_name)
    nb_interface = _get('dcim', 'interfaces', auth_required=True, device_id=nb_device['id'], name=interface_name)
    if not nb_device:
        return False
    if not nb_interface:
        return False
    else:
        for (k, v) in __utils__['args.clean_kwargs'](**kwargs).items():
            setattr(nb_interface, k, v)
        try:
            nb_interface.save()
            return {'dcim': {'interfaces': {nb_interface.id: dict(nb_interface)}}}
        except pynetbox.RequestError as e:
            log.error('%s, %s, %s', e.req.request.headers, e.request_body, e.error)
            return False

def delete_interface(device_name, interface_name):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2019.2.0\n\n    Delete an interface from a device.\n\n    device_name\n        The name of the device, e.g., ``edge_router``.\n\n    interface_name\n        The name of the interface, e.g., ``ae13``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.delete_interface edge_router ae13\n    '
    nb_device = get_('dcim', 'devices', name=device_name)
    nb_interface = _get('dcim', 'interfaces', auth_required=True, device_id=nb_device['id'], name=interface_name)
    if nb_interface:
        nb_interface.delete()
        return {'DELETE': {'dcim': {'interfaces': {nb_interface.id: nb_interface.name}}}}
    return False

def make_interface_lag(device_name, interface_name):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Update an interface to be a LAG.\n\n    device_name\n        The name of the device, e.g., ``edge_router``.\n\n    interface_name\n        The name of the interface, e.g., ``ae13``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.make_interface_lag edge_router ae13\n    '
    return update_interface(device_name, interface_name, form_factor=200)

def make_interface_child(device_name, interface_name, parent_name):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Set an interface as part of a LAG.\n\n    device_name\n        The name of the device, e.g., ``edge_router``.\n\n    interface_name\n        The name of the interface to be attached to LAG, e.g., ``xe-1/0/2``.\n\n    parent_name\n        The name of the LAG interface, e.g., ``ae13``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.make_interface_child xe-1/0/2 ae13\n    '
    nb_device = get_('dcim', 'devices', name=device_name)
    nb_parent = get_('dcim', 'interfaces', device_id=nb_device['id'], name=parent_name)
    if nb_device and nb_parent:
        return update_interface(device_name, interface_name, lag=nb_parent['id'])
    else:
        return False

def get_ipaddresses(device_name=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2019.2.0\n\n    Filters for an IP address using specified filters\n\n    device_name\n        The name of the device to check for the IP address\n    kwargs\n        Optional arguments that can be used to filter, e.g., ``family=4``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.get_ipaddresses device_name family=4\n    '
    if not device_name:
        device_name = __opts__['id']
    netbox_device = get_('dcim', 'devices', name=device_name)
    return filter_('ipam', 'ip-addresses', device_id=netbox_device['id'], **kwargs)

def create_ipaddress(ip_address, family, device=None, interface=None):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2019.2.0\n\n    Add an IP address, and optionally attach it to an interface.\n\n    ip_address\n        The IP address and CIDR, e.g., ``192.168.1.1/24``\n    family\n        Integer of IP family, e.g., ``4``\n    device\n        The name of the device to attach IP to, e.g., ``edge_router``\n    interface\n        The name of the interface to attach IP to, e.g., ``ae13``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_ipaddress 192.168.1.1/24 4 device=edge_router interface=ae13\n    '
    nb_addr = None
    payload = {'family': family, 'address': ip_address}
    if interface and device:
        nb_device = get_('dcim', 'devices', name=device)
        if not nb_device:
            return False
        nb_interface = get_('dcim', 'interfaces', device_id=nb_device['id'], name=interface)
        if not nb_interface:
            return False
        nb_addr = get_('ipam', 'ip-addresses', q=ip_address, interface_id=nb_interface['id'], family=family)
        if nb_addr:
            log.error(nb_addr)
            return False
        else:
            payload['interface'] = nb_interface['id']
    ipaddr = _add('ipam', 'ip-addresses', payload)
    if ipaddr:
        return {'ipam': {'ip-addresses': payload}}
    else:
        return ipaddr

def delete_ipaddress(ipaddr_id):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Delete an IP address. IP addresses in Netbox are a combination of address\n    and the interface it is assigned to.\n\n    id\n        The Netbox id for the IP address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.delete_ipaddress 9002\n    '
    nb_ipaddr = _get('ipam', 'ip-addresses', auth_required=True, id=ipaddr_id)
    if nb_ipaddr:
        nb_ipaddr.delete()
        return {'DELETE': {'ipam': {'ip-address': ipaddr_id}}}
    return False

def create_circuit_provider(name, asn=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new Netbox circuit provider\n\n    name\n        The name of the circuit provider\n    asn\n        The ASN of the circuit provider\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_circuit_provider Telia 1299\n    '
    nb_circuit_provider = get_('circuits', 'providers', name=name)
    payload = {}
    if nb_circuit_provider:
        if nb_circuit_provider['asn'] == asn:
            return False
        else:
            log.error('Duplicate provider with different ASN: %s: %s', name, asn)
            raise CommandExecutionError('Duplicate provider with different ASN: {}: {}'.format(name, asn))
    else:
        payload = {'name': name, 'slug': slugify(name)}
        if asn:
            payload['asn'] = asn
        circuit_provider = _add('circuits', 'providers', payload)
        if circuit_provider:
            return {'circuits': {'providers': {circuit_provider['id']: payload}}}
        else:
            return circuit_provider

def get_circuit_provider(name, asn=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Get a circuit provider with a given name and optional ASN.\n\n    name\n        The name of the circuit provider\n    asn\n        The ASN of the circuit provider\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.get_circuit_provider Telia 1299\n    '
    if asn:
        nb_circuit_provider = get_('circuits', 'providers', asn=asn)
    else:
        nb_circuit_provider = get_('circuits', 'providers', name=name)
    return nb_circuit_provider

def create_circuit_type(name):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new Netbox circuit type.\n\n    name\n        The name of the circuit type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_circuit_type Transit\n    '
    nb_circuit_type = get_('circuits', 'circuit-types', slug=slugify(name))
    if nb_circuit_type:
        return False
    else:
        payload = {'name': name, 'slug': slugify(name)}
        circuit_type = _add('circuits', 'circuit-types', payload)
        if circuit_type:
            return {'circuits': {'circuit-types': {circuit_type['id']: payload}}}
        else:
            return circuit_type

def create_circuit(name, provider_id, circuit_type, description=None):
    if False:
        return 10
    '\n    .. versionadded:: 2019.2.0\n\n    Create a new Netbox circuit\n\n    name\n        Name of the circuit\n    provider_id\n        The netbox id of the circuit provider\n    circuit_type\n        The name of the circuit type\n    asn\n        The ASN of the circuit provider\n    description\n        The description of the circuit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_circuit NEW_CIRCUIT_01 Telia Transit 1299 "New Telia circuit"\n    '
    nb_circuit_provider = get_('circuits', 'providers', provider_id)
    nb_circuit_type = get_('circuits', 'circuit-types', slug=slugify(circuit_type))
    if nb_circuit_provider and nb_circuit_type:
        payload = {'cid': name, 'provider': nb_circuit_provider['id'], 'type': nb_circuit_type['id']}
        if description:
            payload['description'] = description
        nb_circuit = get_('circuits', 'circuits', cid=name)
        if nb_circuit:
            return False
        circuit = _add('circuits', 'circuits', payload)
        if circuit:
            return {'circuits': {'circuits': {circuit['id']: payload}}}
        else:
            return circuit
    else:
        return False

def create_circuit_termination(circuit, interface, device, speed, xconnect_id=None, term_side='A'):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2019.2.0\n\n    Terminate a circuit on an interface\n\n    circuit\n        The name of the circuit\n    interface\n        The name of the interface to terminate on\n    device\n        The name of the device the interface belongs to\n    speed\n        The speed of the circuit, in Kbps\n    xconnect_id\n        The cross-connect identifier\n    term_side\n        The side of the circuit termination\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion netbox.create_circuit_termination NEW_CIRCUIT_01 xe-0/0/1 myminion 10000 xconnect_id=XCON01\n    '
    nb_device = get_('dcim', 'devices', name=device)
    nb_interface = get_('dcim', 'interfaces', device_id=nb_device['id'], name=interface)
    nb_circuit = get_('circuits', 'circuits', cid=circuit)
    if nb_circuit and nb_device:
        nb_termination = get_('circuits', 'circuit-terminations', q=nb_circuit['cid'])
        if nb_termination:
            return False
        payload = {'circuit': nb_circuit['id'], 'interface': nb_interface['id'], 'site': nb_device['site']['id'], 'port_speed': speed, 'term_side': term_side}
        if xconnect_id:
            payload['xconnect_id'] = xconnect_id
        circuit_termination = _add('circuits', 'circuit-terminations', payload)
        if circuit_termination:
            return {'circuits': {'circuit-terminations': {circuit_termination['id']: payload}}}
        else:
            return circuit_termination