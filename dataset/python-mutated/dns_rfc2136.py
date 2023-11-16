"""DNS Authenticator using RFC 2136 Dynamic Updates."""
import logging
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional
import dns.flags
import dns.message
import dns.name
import dns.query
import dns.rdataclass
import dns.rdatatype
import dns.tsig
import dns.tsigkeyring
import dns.update
from certbot import errors
from certbot.plugins import dns_common
from certbot.plugins.dns_common import CredentialsConfiguration
from certbot.util import is_ipaddress
logger = logging.getLogger(__name__)
DEFAULT_NETWORK_TIMEOUT = 45

class Authenticator(dns_common.DNSAuthenticator):
    """DNS Authenticator using RFC 2136 Dynamic Updates

    This Authenticator uses RFC 2136 Dynamic Updates to fulfill a dns-01 challenge.
    """
    ALGORITHMS = {'HMAC-MD5': dns.tsig.HMAC_MD5, 'HMAC-SHA1': dns.tsig.HMAC_SHA1, 'HMAC-SHA224': dns.tsig.HMAC_SHA224, 'HMAC-SHA256': dns.tsig.HMAC_SHA256, 'HMAC-SHA384': dns.tsig.HMAC_SHA384, 'HMAC-SHA512': dns.tsig.HMAC_SHA512}
    PORT = 53
    description = 'Obtain certificates using a DNS TXT record (if you are using BIND for DNS).'
    ttl = 120

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.credentials: Optional[CredentialsConfiguration] = None

    @classmethod
    def add_parser_arguments(cls, add: Callable[..., None], default_propagation_seconds: int=60) -> None:
        if False:
            return 10
        super().add_parser_arguments(add, default_propagation_seconds=60)
        add('credentials', help='RFC 2136 credentials INI file.')

    def more_info(self) -> str:
        if False:
            print('Hello World!')
        return 'This plugin configures a DNS TXT record to respond to a dns-01 challenge using ' + 'RFC 2136 Dynamic Updates.'

    def _validate_credentials(self, credentials: CredentialsConfiguration) -> None:
        if False:
            while True:
                i = 10
        server = cast(str, credentials.conf('server'))
        if not is_ipaddress(server):
            raise errors.PluginError('The configured target DNS server ({0}) is not a valid IPv4 or IPv6 address. A hostname is not allowed.'.format(server))
        algorithm = credentials.conf('algorithm')
        if algorithm:
            if not self.ALGORITHMS.get(algorithm.upper()):
                raise errors.PluginError('Unknown algorithm: {0}.'.format(algorithm))

    def _setup_credentials(self) -> None:
        if False:
            print('Hello World!')
        self.credentials = self._configure_credentials('credentials', 'RFC 2136 credentials INI file', {'name': 'TSIG key name', 'secret': 'TSIG key secret', 'server': 'The target DNS server'}, self._validate_credentials)

    def _perform(self, _domain: str, validation_name: str, validation: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._get_rfc2136_client().add_txt_record(validation_name, validation, self.ttl)

    def _cleanup(self, _domain: str, validation_name: str, validation: str) -> None:
        if False:
            i = 10
            return i + 15
        self._get_rfc2136_client().del_txt_record(validation_name, validation)

    def _get_rfc2136_client(self) -> '_RFC2136Client':
        if False:
            while True:
                i = 10
        if not self.credentials:
            raise errors.Error('Plugin has not been prepared.')
        return _RFC2136Client(cast(str, self.credentials.conf('server')), int(cast(str, self.credentials.conf('port')) or self.PORT), cast(str, self.credentials.conf('name')), cast(str, self.credentials.conf('secret')), self.ALGORITHMS.get(self.credentials.conf('algorithm') or '', dns.tsig.HMAC_MD5), (self.credentials.conf('sign_query') or '').upper() == 'TRUE')

class _RFC2136Client:
    """
    Encapsulates all communication with the target DNS server.
    """

    def __init__(self, server: str, port: int, key_name: str, key_secret: str, key_algorithm: dns.name.Name, sign_query: bool, timeout: int=DEFAULT_NETWORK_TIMEOUT) -> None:
        if False:
            return 10
        self.server = server
        self.port = port
        self.keyring = dns.tsigkeyring.from_text({key_name: key_secret})
        self.algorithm = key_algorithm
        self.sign_query = sign_query
        self._default_timeout = timeout

    def add_txt_record(self, record_name: str, record_content: str, record_ttl: int) -> None:
        if False:
            return 10
        "\n        Add a TXT record using the supplied information.\n\n        :param str record_name: The record name (typically beginning with '_acme-challenge.').\n        :param str record_content: The record content (typically the challenge validation).\n        :param int record_ttl: The record TTL (number of seconds that the record may be cached).\n        :raises certbot.errors.PluginError: if an error occurs communicating with the DNS server\n        "
        domain = self._find_domain(record_name)
        n = dns.name.from_text(record_name)
        o = dns.name.from_text(domain)
        rel = n.relativize(o)
        update = dns.update.Update(domain, keyring=self.keyring, keyalgorithm=self.algorithm)
        update.add(rel, record_ttl, dns.rdatatype.TXT, record_content)
        try:
            response = dns.query.tcp(update, self.server, self._default_timeout, self.port)
        except Exception as e:
            raise errors.PluginError('Encountered error adding TXT record: {0}'.format(e))
        rcode = response.rcode()
        if rcode == dns.rcode.NOERROR:
            logger.debug('Successfully added TXT record %s', record_name)
        else:
            raise errors.PluginError('Received response from server: {0}'.format(dns.rcode.to_text(rcode)))

    def del_txt_record(self, record_name: str, record_content: str) -> None:
        if False:
            return 10
        "\n        Delete a TXT record using the supplied information.\n\n        :param str record_name: The record name (typically beginning with '_acme-challenge.').\n        :param str record_content: The record content (typically the challenge validation).\n        :param int record_ttl: The record TTL (number of seconds that the record may be cached).\n        :raises certbot.errors.PluginError: if an error occurs communicating with the DNS server\n        "
        domain = self._find_domain(record_name)
        n = dns.name.from_text(record_name)
        o = dns.name.from_text(domain)
        rel = n.relativize(o)
        update = dns.update.Update(domain, keyring=self.keyring, keyalgorithm=self.algorithm)
        update.delete(rel, dns.rdatatype.TXT, record_content)
        try:
            response = dns.query.tcp(update, self.server, self._default_timeout, self.port)
        except Exception as e:
            raise errors.PluginError('Encountered error deleting TXT record: {0}'.format(e))
        rcode = response.rcode()
        if rcode == dns.rcode.NOERROR:
            logger.debug('Successfully deleted TXT record %s', record_name)
        else:
            raise errors.PluginError('Received response from server: {0}'.format(dns.rcode.to_text(rcode)))

    def _find_domain(self, record_name: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Find the closest domain with an SOA record for a given domain name.\n\n        :param str record_name: The record name for which to find the closest SOA record.\n        :returns: The domain, if found.\n        :rtype: str\n        :raises certbot.errors.PluginError: if no SOA record can be found.\n        '
        domain_name_guesses = dns_common.base_domain_name_guesses(record_name)
        for guess in domain_name_guesses:
            if self._query_soa(guess):
                return guess
        raise errors.PluginError('Unable to determine base domain for {0} using names: {1}.'.format(record_name, domain_name_guesses))

    def _query_soa(self, domain_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Query a domain name for an authoritative SOA record.\n\n        :param str domain_name: The domain name to query for an SOA record.\n        :returns: True if found, False otherwise.\n        :rtype: bool\n        :raises certbot.errors.PluginError: if no response is received.\n        '
        domain = dns.name.from_text(domain_name)
        request = dns.message.make_query(domain, dns.rdatatype.SOA, dns.rdataclass.IN)
        request.flags ^= dns.flags.RD
        if self.sign_query:
            request.use_tsig(self.keyring, algorithm=self.algorithm)
        try:
            try:
                response = dns.query.tcp(request, self.server, self._default_timeout, self.port)
            except (OSError, dns.exception.Timeout) as e:
                logger.debug('TCP query failed, fallback to UDP: %s', e)
                response = dns.query.udp(request, self.server, self._default_timeout, self.port)
            rcode = response.rcode()
            if rcode == dns.rcode.NOERROR and response.get_rrset(response.answer, domain, dns.rdataclass.IN, dns.rdatatype.SOA) and response.flags & dns.flags.AA:
                logger.debug('Received authoritative SOA response for %s', domain_name)
                return True
            logger.debug('No authoritative SOA record found for %s', domain_name)
            return False
        except Exception as e:
            raise errors.PluginError('Encountered error when making query: {0}'.format(e))