from . import Net
from .asn import IPASN
from .nir import NIRWhois
import logging
log = logging.getLogger(__name__)

class IPWhois:
    """
    The wrapper class for performing whois/RDAP lookups and parsing for
    IPv4 and IPv6 addresses.

    Args:
        address (:obj:`str`/:obj:`int`/:obj:`IPv4Address`/:obj:`IPv6Address`):
            An IPv4 or IPv6 address
        timeout (:obj:`int`): The default timeout for socket connections in
            seconds. Defaults to 5.
        proxy_opener (:obj:`urllib.request.OpenerDirector`): The request for
            proxy support. Defaults to None.
    """

    def __init__(self, address, timeout=5, proxy_opener=None):
        if False:
            return 10
        self.net = Net(address=address, timeout=timeout, proxy_opener=proxy_opener)
        self.ipasn = IPASN(self.net)
        self.address = self.net.address
        self.timeout = self.net.timeout
        self.address_str = self.net.address_str
        self.version = self.net.version
        self.reversed = self.net.reversed
        self.dns_zone = self.net.dns_zone

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'IPWhois({0}, {1}, {2})'.format(self.address_str, str(self.timeout), repr(self.net.opener))

    def lookup_whois(self, inc_raw=False, retry_count=3, get_referral=False, extra_blacklist=None, ignore_referral_errors=False, field_list=None, extra_org_map=None, inc_nir=True, nir_field_list=None, asn_methods=None, get_asn_description=True):
        if False:
            while True:
                i = 10
        "\n        The function for retrieving and parsing whois information for an IP\n        address via port 43 (WHOIS).\n\n        Args:\n            inc_raw (:obj:`bool`): Whether to include the raw whois results in\n                the returned dictionary. Defaults to False.\n            retry_count (:obj:`int`): The number of times to retry in case\n                socket errors, timeouts, connection resets, etc. are\n                encountered. Defaults to 3.\n            get_referral (:obj:`bool`): Whether to retrieve referral whois\n                information, if available. Defaults to False.\n            extra_blacklist (:obj:`list`): Blacklisted whois servers in\n                addition to the global BLACKLIST. Defaults to None.\n            ignore_referral_errors (:obj:`bool`): Whether to ignore and\n                continue when an exception is encountered on referral whois\n                lookups. Defaults to False.\n            field_list (:obj:`list`): If provided, a list of fields to parse:\n                ['name', 'handle', 'description', 'country', 'state', 'city',\n                'address', 'postal_code', 'emails', 'created', 'updated']\n                If None, defaults to all.\n            extra_org_map (:obj:`dict`): Dictionary mapping org handles to\n                RIRs. This is for limited cases where ARIN REST (ASN fallback\n                HTTP lookup) does not show an RIR as the org handle e.g., DNIC\n                (which is now the built in ORG_MAP) e.g., {'DNIC': 'arin'}.\n                Valid RIR values are (note the case-sensitive - this is meant\n                to match the REST result):\n                'ARIN', 'RIPE', 'apnic', 'lacnic', 'afrinic'\n                Defaults to None.\n            inc_nir (:obj:`bool`): Whether to retrieve NIR (National Internet\n                Registry) information, if registry is JPNIC (Japan) or KRNIC\n                (Korea). If True, extra network requests will be required.\n                If False, the information returned for JP or KR IPs is\n                severely restricted. Defaults to True.\n            nir_field_list (:obj:`list`): If provided and inc_nir, a list of\n                fields to parse:\n                ['name', 'handle', 'country', 'address', 'postal_code',\n                'nameservers', 'created', 'updated', 'contacts']\n                If None, defaults to all.\n            asn_methods (:obj:`list`): ASN lookup types to attempt, in order.\n                If None, defaults to all ['dns', 'whois', 'http'].\n            get_asn_description (:obj:`bool`): Whether to run an additional\n                query when pulling ASN information via dns, in order to get\n                the ASN description. Defaults to True.\n\n        Returns:\n            dict: The IP whois lookup results\n\n            ::\n\n                {\n                    'query' (str) - The IP address\n                    'asn' (str) - The Autonomous System Number\n                    'asn_date' (str) - The ASN Allocation date\n                    'asn_registry' (str) - The assigned ASN registry\n                    'asn_cidr' (str) - The assigned ASN CIDR\n                    'asn_country_code' (str) - The assigned ASN country code\n                    'asn_description' (str) - The ASN description\n                    'nets' (list) - Dictionaries containing network\n                        information which consists of the fields listed in the\n                        ipwhois.whois.RIR_WHOIS dictionary.\n                    'raw' (str) - Raw whois results if the inc_raw parameter\n                        is True.\n                    'referral' (dict) - Referral whois information if\n                        get_referral is True and the server is not blacklisted.\n                        Consists of fields listed in the ipwhois.whois.RWHOIS\n                        dictionary.\n                    'raw_referral' (str) - Raw referral whois results if the\n                        inc_raw parameter is True.\n                    'nir' (dict) - ipwhois.nir.NIRWhois() results if inc_nir\n                        is True.\n                }\n        "
        from .whois import Whois
        results = {'nir': None}
        log.debug('ASN lookup for {0}'.format(self.address_str))
        asn_data = self.ipasn.lookup(inc_raw=inc_raw, retry_count=retry_count, extra_org_map=extra_org_map, asn_methods=asn_methods, get_asn_description=get_asn_description)
        results.update(asn_data)
        whois = Whois(self.net)
        log.debug('WHOIS lookup for {0}'.format(self.address_str))
        whois_data = whois.lookup(inc_raw=inc_raw, retry_count=retry_count, response=None, get_referral=get_referral, extra_blacklist=extra_blacklist, ignore_referral_errors=ignore_referral_errors, asn_data=asn_data, field_list=field_list)
        results.update(whois_data)
        if inc_nir:
            nir = None
            if 'JP' == asn_data['asn_country_code']:
                nir = 'jpnic'
            elif 'KR' == asn_data['asn_country_code']:
                nir = 'krnic'
            if nir:
                nir_whois = NIRWhois(self.net)
                nir_data = nir_whois.lookup(nir=nir, inc_raw=inc_raw, retry_count=retry_count, response=None, field_list=nir_field_list, is_offline=False)
                results['nir'] = nir_data
        return results

    def lookup_rdap(self, inc_raw=False, retry_count=3, depth=0, excluded_entities=None, bootstrap=False, rate_limit_timeout=120, extra_org_map=None, inc_nir=True, nir_field_list=None, asn_methods=None, get_asn_description=True, root_ent_check=True):
        if False:
            i = 10
            return i + 15
        "\n        The function for retrieving and parsing whois information for an IP\n        address via HTTP (RDAP).\n\n        **This is now the recommended method, as RDAP contains much better\n        information to parse.**\n\n        Args:\n            inc_raw (:obj:`bool`): Whether to include the raw whois results in\n                the returned dictionary. Defaults to False.\n            retry_count (:obj:`int`): The number of times to retry in case\n                socket errors, timeouts, connection resets, etc. are\n                encountered. Defaults to 3.\n            depth (:obj:`int`): How many levels deep to run queries when\n                additional referenced objects are found. Defaults to 0.\n            excluded_entities (:obj:`list`): Entity handles to not perform\n                lookups. Defaults to None.\n            bootstrap (:obj:`bool`): If True, performs lookups via ARIN\n                bootstrap rather than lookups based on ASN data. ASN lookups\n                are not performed and no output for any of the asn* fields is\n                provided. Defaults to False.\n            rate_limit_timeout (:obj:`int`): The number of seconds to wait\n                before retrying when a rate limit notice is returned via\n                rdap+json. Defaults to 120.\n            extra_org_map (:obj:`dict`): Dictionary mapping org handles to\n                RIRs. This is for limited cases where ARIN REST (ASN fallback\n                HTTP lookup) does not show an RIR as the org handle e.g., DNIC\n                (which is now the built in ORG_MAP) e.g., {'DNIC': 'arin'}.\n                Valid RIR values are (note the case-sensitive - this is meant\n                to match the REST result):\n                'ARIN', 'RIPE', 'apnic', 'lacnic', 'afrinic'\n                Defaults to None.\n            inc_nir (:obj:`bool`): Whether to retrieve NIR (National Internet\n                Registry) information, if registry is JPNIC (Japan) or KRNIC\n                (Korea). If True, extra network requests will be required.\n                If False, the information returned for JP or KR IPs is\n                severely restricted. Defaults to True.\n            nir_field_list (:obj:`list`): If provided and inc_nir, a list of\n                fields to parse:\n                ['name', 'handle', 'country', 'address', 'postal_code',\n                'nameservers', 'created', 'updated', 'contacts']\n                If None, defaults to all.\n            asn_methods (:obj:`list`): ASN lookup types to attempt, in order.\n                If None, defaults to all ['dns', 'whois', 'http'].\n            get_asn_description (:obj:`bool`): Whether to run an additional\n                query when pulling ASN information via dns, in order to get\n                the ASN description. Defaults to True.\n            root_ent_check (:obj:`bool`): If True, will perform\n                additional RDAP HTTP queries for missing entity data at the\n                root level. Defaults to True.\n\n        Returns:\n            dict: The IP RDAP lookup results\n\n            ::\n\n                {\n                    'query' (str) - The IP address\n                    'asn' (str) - The Autonomous System Number\n                    'asn_date' (str) - The ASN Allocation date\n                    'asn_registry' (str) - The assigned ASN registry\n                    'asn_cidr' (str) - The assigned ASN CIDR\n                    'asn_country_code' (str) - The assigned ASN country code\n                    'asn_description' (str) - The ASN description\n                    'entities' (list) - Entity handles referred by the top\n                        level query.\n                    'network' (dict) - Network information which consists of\n                        the fields listed in the ipwhois.rdap._RDAPNetwork\n                        dict.\n                    'objects' (dict) - Mapping of entity handle->entity dict\n                        which consists of the fields listed in the\n                        ipwhois.rdap._RDAPEntity dict. The raw result is\n                        included for each object if the inc_raw parameter\n                        is True.\n                    'raw' (dict) - Whois results in json format if the inc_raw\n                        parameter is True.\n                    'nir' (dict) - ipwhois.nir.NIRWhois results if inc_nir is\n                        True.\n                }\n        "
        from .rdap import RDAP
        results = {'nir': None}
        asn_data = None
        response = None
        if not bootstrap:
            log.debug('ASN lookup for {0}'.format(self.address_str))
            asn_data = self.ipasn.lookup(inc_raw=inc_raw, retry_count=retry_count, extra_org_map=extra_org_map, asn_methods=asn_methods, get_asn_description=get_asn_description)
            results.update(asn_data)
        rdap = RDAP(self.net)
        log.debug('RDAP lookup for {0}'.format(self.address_str))
        rdap_data = rdap.lookup(inc_raw=inc_raw, retry_count=retry_count, asn_data=asn_data, depth=depth, excluded_entities=excluded_entities, response=response, bootstrap=bootstrap, rate_limit_timeout=rate_limit_timeout, root_ent_check=root_ent_check)
        results.update(rdap_data)
        if inc_nir:
            nir = None
            if 'JP' == asn_data['asn_country_code']:
                nir = 'jpnic'
            elif 'KR' == asn_data['asn_country_code']:
                nir = 'krnic'
            if nir:
                nir_whois = NIRWhois(self.net)
                nir_data = nir_whois.lookup(nir=nir, inc_raw=inc_raw, retry_count=retry_count, response=None, field_list=nir_field_list, is_offline=False)
                results['nir'] = nir_data
        return results