"""
Namecheap Domain Management

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
import logging
CAN_USE_NAMECHEAP = True
try:
    import salt.utils.namecheap
except ImportError:
    CAN_USE_NAMECHEAP = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Check to make sure requests and xml are installed and requests\n    '
    if CAN_USE_NAMECHEAP:
        return 'namecheap_domains'
    return False

def reactivate(domain_name):
    if False:
        while True:
            i = 10
    "\n    Try to reactivate the expired domain name\n\n    Returns the following information:\n\n    - Whether or not the domain was reactivated successfully\n    - The amount charged for reactivation\n    - The order ID\n    - The transaction ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.reactivate my-domain-name\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.reactivate')
    opts['DomainName'] = domain_name
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    domainreactivateresult = response_xml.getElementsByTagName('DomainReactivateResult')[0]
    return salt.utils.namecheap.xml_to_dict(domainreactivateresult)

def renew(domain_name, years, promotion_code=None):
    if False:
        i = 10
        return i + 15
    "\n    Try to renew the specified expiring domain name for a specified number of years\n\n    domain_name\n        The domain name to be renewed\n\n    years\n        Number of years to renew\n\n    Returns the following information:\n\n    - Whether or not the domain was renewed successfully\n    - The domain ID\n    - The order ID\n    - The transaction ID\n    - The amount charged for renewal\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.renew my-domain-name 5\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.renew')
    opts['DomainName'] = domain_name
    opts['Years'] = years
    if promotion_code is not None:
        opts['PromotionCode'] = promotion_code
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    domainrenewresult = response_xml.getElementsByTagName('DomainRenewResult')[0]
    return salt.utils.namecheap.xml_to_dict(domainrenewresult)

def create(domain_name, years, **kwargs):
    if False:
        print('Hello World!')
    "\n    Try to register the specified domain name\n\n    domain_name\n        The domain name to be registered\n\n    years\n        Number of years to register\n\n    Returns the following information:\n\n    - Whether or not the domain was renewed successfully\n    - Whether or not WhoisGuard is enabled\n    - Whether or not registration is instant\n    - The amount charged for registration\n    - The domain ID\n    - The order ID\n    - The transaction ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.create my-domain-name 2\n    "
    idn_codes = ('afr', 'alb', 'ara', 'arg', 'arm', 'asm', 'ast', 'ave', 'awa', 'aze', 'bak', 'bal', 'ban', 'baq', 'bas', 'bel', 'ben', 'bho', 'bos', 'bul', 'bur', 'car', 'cat', 'che', 'chi', 'chv', 'cop', 'cos', 'cze', 'dan', 'div', 'doi', 'dut', 'eng', 'est', 'fao', 'fij', 'fin', 'fre', 'fry', 'geo', 'ger', 'gla', 'gle', 'gon', 'gre', 'guj', 'heb', 'hin', 'hun', 'inc', 'ind', 'inh', 'isl', 'ita', 'jav', 'jpn', 'kas', 'kaz', 'khm', 'kir', 'kor', 'kur', 'lao', 'lav', 'lit', 'ltz', 'mal', 'mkd', 'mlt', 'mol', 'mon', 'mri', 'msa', 'nep', 'nor', 'ori', 'oss', 'pan', 'per', 'pol', 'por', 'pus', 'raj', 'rum', 'rus', 'san', 'scr', 'sin', 'slo', 'slv', 'smo', 'snd', 'som', 'spa', 'srd', 'srp', 'swa', 'swe', 'syr', 'tam', 'tel', 'tgk', 'tha', 'tib', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'wel', 'yid')
    require_opts = ['AdminAddress1', 'AdminCity', 'AdminCountry', 'AdminEmailAddress', 'AdminFirstName', 'AdminLastName', 'AdminPhone', 'AdminPostalCode', 'AdminStateProvince', 'AuxBillingAddress1', 'AuxBillingCity', 'AuxBillingCountry', 'AuxBillingEmailAddress', 'AuxBillingFirstName', 'AuxBillingLastName', 'AuxBillingPhone', 'AuxBillingPostalCode', 'AuxBillingStateProvince', 'RegistrantAddress1', 'RegistrantCity', 'RegistrantCountry', 'RegistrantEmailAddress', 'RegistrantFirstName', 'RegistrantLastName', 'RegistrantPhone', 'RegistrantPostalCode', 'RegistrantStateProvince', 'TechAddress1', 'TechCity', 'TechCountry', 'TechEmailAddress', 'TechFirstName', 'TechLastName', 'TechPhone', 'TechPostalCode', 'TechStateProvince', 'Years']
    opts = salt.utils.namecheap.get_opts('namecheap.domains.create')
    opts['DomainName'] = domain_name
    opts['Years'] = str(years)

    def add_to_opts(opts_dict, kwargs, value, suffix, prefices):
        if False:
            print('Hello World!')
        for prefix in prefices:
            nextkey = prefix + suffix
            if nextkey not in kwargs:
                opts_dict[nextkey] = value
    for (key, value) in kwargs.items():
        if key.startswith('Registrant'):
            add_to_opts(opts, kwargs, value, key[10:], ['Tech', 'Admin', 'AuxBilling', 'Billing'])
        if key.startswith('Tech'):
            add_to_opts(opts, kwargs, value, key[4:], ['Registrant', 'Admin', 'AuxBilling', 'Billing'])
        if key.startswith('Admin'):
            add_to_opts(opts, kwargs, value, key[5:], ['Registrant', 'Tech', 'AuxBilling', 'Billing'])
        if key.startswith('AuxBilling'):
            add_to_opts(opts, kwargs, value, key[10:], ['Registrant', 'Tech', 'Admin', 'Billing'])
        if key.startswith('Billing'):
            add_to_opts(opts, kwargs, value, key[7:], ['Registrant', 'Tech', 'Admin', 'AuxBilling'])
        if key == 'IdnCode' and key not in idn_codes:
            log.error('Invalid IdnCode')
            raise Exception('Invalid IdnCode')
        opts[key] = value
    for requiredkey in require_opts:
        if requiredkey not in opts:
            log.error("Missing required parameter '%s'", requiredkey)
            raise Exception("Missing required parameter '{}'".format(requiredkey))
    response_xml = salt.utils.namecheap.post_request(opts)
    if response_xml is None:
        return {}
    domainresult = response_xml.getElementsByTagName('DomainCreateResult')[0]
    return salt.utils.namecheap.atts_to_dict(domainresult)

def check(*domains_to_check):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks the availability of domains\n\n    domains_to_check\n        array of strings  List of domains to check\n\n    Returns a dictionary mapping the each domain name to a boolean denoting\n    whether or not it is available.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.check domain-to-check\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.check')
    opts['DomainList'] = ','.join(domains_to_check)
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return {}
    domains_checked = {}
    for result in response_xml.getElementsByTagName('DomainCheckResult'):
        available = result.getAttribute('Available')
        domains_checked[result.getAttribute('Domain').lower()] = salt.utils.namecheap.string_to_value(available)
    return domains_checked

def get_info(domain_name):
    if False:
        print('Hello World!')
    "\n    Returns information about the requested domain\n\n    returns a dictionary of information about the domain_name\n\n    domain_name\n        string  Domain name to get information about\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.get_info my-domain-name\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.getinfo')
    opts['DomainName'] = domain_name
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return []
    domaingetinforesult = response_xml.getElementsByTagName('DomainGetInfoResult')[0]
    return salt.utils.namecheap.xml_to_dict(domaingetinforesult)

def get_tld_list():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a list of TLDs as objects\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.get_tld_list\n    "
    response_xml = salt.utils.namecheap.get_request(salt.utils.namecheap.get_opts('namecheap.domains.gettldlist'))
    if response_xml is None:
        return []
    tldresult = response_xml.getElementsByTagName('Tlds')[0]
    tlds = []
    for e in tldresult.getElementsByTagName('Tld'):
        tld = salt.utils.namecheap.atts_to_dict(e)
        tld['data'] = e.firstChild.data
        categories = []
        subcategories = e.getElementsByTagName('Categories')[0]
        for c in subcategories.getElementsByTagName('TldCategory'):
            categories.append(salt.utils.namecheap.atts_to_dict(c))
        tld['categories'] = categories
        tlds.append(tld)
    return tlds

def get_list(list_type=None, search_term=None, page=None, page_size=None, sort_by=None):
    if False:
        while True:
            i = 10
    "\n    Returns a list of domains for the particular user as a list of objects\n    offset by ``page`` length of ``page_size``\n\n    list_type : ALL\n        One of ``ALL``, ``EXPIRING``, ``EXPIRED``\n\n    search_term\n        Keyword to look for on the domain list\n\n    page : 1\n        Number of result page to return\n\n    page_size : 20\n        Number of domains to be listed per page (minimum: ``10``, maximum:\n        ``100``)\n\n    sort_by\n        One of ``NAME``, ``NAME_DESC``, ``EXPIREDATE``, ``EXPIREDATE_DESC``,\n        ``CREATEDATE``, or ``CREATEDATE_DESC``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'my-minion' namecheap_domains.get_list\n    "
    opts = salt.utils.namecheap.get_opts('namecheap.domains.getList')
    if list_type is not None:
        if list_type not in ['ALL', 'EXPIRING', 'EXPIRED']:
            log.error('Invalid option for list_type')
            raise Exception('Invalid option for list_type')
        opts['ListType'] = list_type
    if search_term is not None:
        if len(search_term) > 70:
            log.warning('search_term trimmed to first 70 characters')
            search_term = search_term[0:70]
        opts['SearchTerm'] = search_term
    if page is not None:
        opts['Page'] = page
    if page_size is not None:
        if page_size > 100 or page_size < 10:
            log.error('Invalid option for page')
            raise Exception('Invalid option for page')
        opts['PageSize'] = page_size
    if sort_by is not None:
        if sort_by not in ['NAME', 'NAME_DESC', 'EXPIREDATE', 'EXPIREDATE_DESC', 'CREATEDATE', 'CREATEDATE_DESC']:
            log.error('Invalid option for sort_by')
            raise Exception('Invalid option for sort_by')
        opts['SortBy'] = sort_by
    response_xml = salt.utils.namecheap.get_request(opts)
    if response_xml is None:
        return []
    domainresult = response_xml.getElementsByTagName('DomainGetListResult')[0]
    domains = []
    for d in domainresult.getElementsByTagName('Domain'):
        domains.append(salt.utils.namecheap.atts_to_dict(d))
    return domains