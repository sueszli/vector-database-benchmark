"""
Namecheap DNS Management

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
__virtualname__ = 'namecheap_domains_dns'

def __virtual__():
    if False:
        return 10
    '\n    Check to make sure requests and xml are installed and requests\n    '
    if CAN_USE_NAMECHEAP:
        return 'namecheap_domains_dns'
    return False

def get_hosts(sld, tld):
    if False:
        return 10
    "\n    Retrieves DNS host record settings for the requested domain.\n\n    returns a dictionary of information about the requested domain\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains_dns.get_hosts sld tld\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.dns.gethosts')
    opts['TLD'] = tld
    opts['SLD'] = sld
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return {}
    domaindnsgethostsresult = response_xml.getElementsByTagName('DomainDNSGetHostsResult')[0]
    return salt.utils.namecheap.xml_to_dict(domaindnsgethostsresult)

def get_list(sld, tld):
    if False:
        print('Hello World!')
    "\n    Gets a list of DNS servers associated with the requested domain.\n\n    returns a dictionary of information about requested domain\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains_dns.get_list sld tld\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.dns.getlist')
    opts['TLD'] = tld
    opts['SLD'] = sld
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return {}
    domaindnsgetlistresult = response_xml.getElementsByTagName('DomainDNSGetListResult')[0]
    return salt.utils.namecheap.xml_to_dict(domaindnsgetlistresult)

def set_hosts(sld, tld, hosts):
    if False:
        print('Hello World!')
    "\n    Sets DNS host records settings for the requested domain.\n\n    returns True if the host records were set successfully\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    hosts\n        Must be passed as a list of Python dictionaries, with each dictionary\n        containing the following keys:\n\n        - **hostname**\n        - **recordtype** - One of ``A``, ``AAAA``, ``CNAME``, ``MX``, ``MXE``,\n          ``TXT``, ``URL``, ``URL301``, or ``FRAME``\n        - **address** - URL or IP address\n        - **ttl** - An integer between 60 and 60000 (default: ``1800``)\n\n        Additionally, the ``mxpref`` key can be present, but must be accompanied\n        by an ``emailtype`` key.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains_dns.set_hosts sld tld hosts\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.dns.setHosts')
    opts['SLD'] = sld
    opts['TLD'] = tld
    i = 1
    for hostrecord in hosts:
        str_i = str(i)
        opts['HostName' + str_i] = hostrecord['hostname']
        opts['RecordType' + str_i] = hostrecord['recordtype']
        opts['Address' + str_i] = hostrecord['address']
        if 'ttl' in hostrecord:
            opts['TTL' + str_i] = hostrecord['ttl']
        if 'mxpref' in hostrecord:
            opts['MXPref' + str_i] = hostrecord['mxpref']
            opts['EmailType'] = hostrecord['emailtype']
        i += 1
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    dnsresult = response_xml.getElementsByTagName('DomainDNSSetHostsResult')[0]
    return salt.utils.namecheap.string_to_value(dnsresult.getAttribute('IsSuccess'))

def set_custom(sld, tld, nameservers):
    if False:
        print('Hello World!')
    "\n    Sets domain to use custom DNS servers.\n\n    returns True if the custom nameservers were set successfully\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    nameservers\n        array of strings  List of nameservers to be associated with this domain\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains_dns.set_custom sld tld nameserver\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.dns.setCustom')
    opts['SLD'] = sld
    opts['TLD'] = tld
    opts['Nameservers'] = ','.join(nameservers)
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    dnsresult = response_xml.getElementsByTagName('DomainDNSSetCustomResult')[0]
    return salt.utils.namecheap.string_to_value(dnsresult.getAttribute('Update'))

def set_default(sld, tld):
    if False:
        print('Hello World!')
    "\n    Sets domain to use namecheap default DNS servers. Required for free\n    services like Host record management, URL forwarding, email forwarding,\n    dynamic DNS and other value added services.\n\n    sld\n        SLD of the domain name\n\n    tld\n        TLD of the domain name\n\n    Returns ``True`` if the domain was successfully pointed at the default DNS\n    servers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains_dns.set_default sld tld\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.dns.setDefault')
    opts['SLD'] = sld
    opts['TLD'] = tld
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return False
    dnsresult = response_xml.getElementsByTagName('DomainDNSSetDefaultResult')[0]
    return salt.utils.namecheap.string_to_value(dnsresult.getAttribute('Updated'))