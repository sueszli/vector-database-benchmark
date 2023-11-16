"""
Namecheap Nameserver Management

.. versionadded:: 2017.7.0

Prerequisites
-------------

This module uses the ``requests`` Python module to communicate to the namecheap
API.

Configuration
-------------

The Namecheap username, API key and URL should be set in the minion configuration
file, or in the Pillar data.

.. code-block:: yaml

    namecheap.name: companyname
    namecheap.key: a1b2c3d4e5f67a8b9c0d1e2f3
    namecheap.client_ip: 162.155.30.172
    #Real url
    namecheap.url: https://api.namecheap.com/xml.response
    #Sandbox url
    #namecheap.url: https://api.sandbox.namecheap.xml.response
"""
CAN_USE_NAMECHEAP = True
try:
    import salt.utils.namecheap
except ImportError:
    CAN_USE_NAMECHEAP = False
__virtualname__ = 'namecheap_domains_ns'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check to make sure requests and xml are installed and requests\n    '
    if CAN_USE_NAMECHEAP:
        return 'namecheap_domains_ns'
    return False

def get_info(sld, tld, nameserver):
    if False:
        while True:
            i = 10
    "\n    Retrieves information about a registered nameserver. Returns the following\n    information:\n\n    - IP Address set for the nameserver\n    - Domain name which was queried\n    - A list of nameservers and their statuses\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    nameserver\n        Nameserver to retrieve\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' namecheap_domains_ns.get_info sld tld nameserver\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.ns.delete')
    opts['SLD'] = sld
    opts['TLD'] = tld
    opts['Nameserver'] = nameserver
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    domainnsinforesult = response_xml.getElementsByTagName('DomainNSInfoResult')[0]
    return salt.utils.namecheap.xml_to_dict(domainnsinforesult)

def update(sld, tld, nameserver, old_ip, new_ip):
    if False:
        return 10
    "\n    Deletes a nameserver. Returns ``True`` if the nameserver was updated\n    successfully.\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    nameserver\n        Nameserver to create\n\n    old_ip\n        Current ip address\n\n    new_ip\n        New ip address\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' namecheap_domains_ns.update sld tld nameserver old_ip new_ip\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.ns.update')
    opts['SLD'] = sld
    opts['TLD'] = tld
    opts['Nameserver'] = nameserver
    opts['OldIP'] = old_ip
    opts['IP'] = new_ip
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    domainnsupdateresult = response_xml.getElementsByTagName('DomainNSUpdateResult')[0]
    return salt.utils.namecheap.string_to_value(domainnsupdateresult.getAttribute('IsSuccess'))

def delete(sld, tld, nameserver):
    if False:
        i = 10
        return i + 15
    "\n    Deletes a nameserver. Returns ``True`` if the nameserver was deleted\n    successfully\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    nameserver\n        Nameserver to delete\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' namecheap_domains_ns.delete sld tld nameserver\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.ns.delete')
    opts['SLD'] = sld
    opts['TLD'] = tld
    opts['Nameserver'] = nameserver
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    domainnsdeleteresult = response_xml.getElementsByTagName('DomainNSDeleteResult')[0]
    return salt.utils.namecheap.string_to_value(domainnsdeleteresult.getAttribute('IsSuccess'))

def create(sld, tld, nameserver, ip):
    if False:
        i = 10
        return i + 15
    "\n    Creates a new nameserver. Returns ``True`` if the nameserver was created\n    successfully.\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    nameserver\n        Nameserver to create\n\n    ip\n        Nameserver IP address\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' namecheap_domains_ns.create sld tld nameserver ip\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.ns.create')
    opts['SLD'] = sld
    opts['TLD'] = tld
    opts['Nameserver'] = nameserver
    opts['IP'] = ip
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    domainnscreateresult = response_xml.getElementsByTagName('DomainNSCreateResult')[0]
    return salt.utils.namecheap.string_to_value(domainnscreateresult.getAttribute('IsSuccess'))