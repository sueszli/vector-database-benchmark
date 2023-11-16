"""Common code for DNS Authenticator Plugins built on Lexicon."""
import abc
import logging
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
import warnings
from requests.exceptions import HTTPError
from requests.exceptions import RequestException
from certbot import configuration
from certbot import errors
from certbot.plugins import dns_common
try:
    from lexicon.client import Client
    from lexicon.config import ConfigResolver
    from lexicon.interfaces import Provider
except ImportError:
    Client = None
    ConfigResolver = None
    Provider = None
logger = logging.getLogger(__name__)

class LexiconClient:
    """
    Encapsulates all communication with a DNS provider via Lexicon.

    .. deprecated:: 2.7.0
       Please use certbot.plugins.dns_common_lexicon.LexiconDNSAuthenticator instead.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.provider: Provider

    def add_txt_record(self, domain: str, record_name: str, record_content: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Add a TXT record using the supplied information.\n\n        :param str domain: The domain to use to look up the managed zone.\n        :param str record_name: The record name (typically beginning with '_acme-challenge.').\n        :param str record_content: The record content (typically the challenge validation).\n        :raises errors.PluginError: if an error occurs communicating with the DNS Provider API\n        "
        self._find_domain_id(domain)
        try:
            self.provider.create_record(rtype='TXT', name=record_name, content=record_content)
        except RequestException as e:
            logger.debug('Encountered error adding TXT record: %s', e, exc_info=True)
            raise errors.PluginError('Error adding TXT record: {0}'.format(e))

    def del_txt_record(self, domain: str, record_name: str, record_content: str) -> None:
        if False:
            while True:
                i = 10
        "\n        Delete a TXT record using the supplied information.\n\n        :param str domain: The domain to use to look up the managed zone.\n        :param str record_name: The record name (typically beginning with '_acme-challenge.').\n        :param str record_content: The record content (typically the challenge validation).\n        :raises errors.PluginError: if an error occurs communicating with the DNS Provider  API\n        "
        try:
            self._find_domain_id(domain)
        except errors.PluginError as e:
            logger.debug('Encountered error finding domain_id during deletion: %s', e, exc_info=True)
            return
        try:
            self.provider.delete_record(rtype='TXT', name=record_name, content=record_content)
        except RequestException as e:
            logger.debug('Encountered error deleting TXT record: %s', e, exc_info=True)

    def _find_domain_id(self, domain: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Find the domain_id for a given domain.\n\n        :param str domain: The domain for which to find the domain_id.\n        :raises errors.PluginError: if the domain_id cannot be found.\n        '
        domain_name_guesses = dns_common.base_domain_name_guesses(domain)
        for domain_name in domain_name_guesses:
            try:
                if hasattr(self.provider, 'options'):
                    self.provider.options['domain'] = domain_name
                else:
                    self.provider.domain = domain_name
                self.provider.authenticate()
                return
            except HTTPError as e:
                result1 = self._handle_http_error(e, domain_name)
                if result1:
                    raise result1
            except Exception as e:
                result2 = self._handle_general_error(e, domain_name)
                if result2:
                    raise result2
        raise errors.PluginError('Unable to determine zone identifier for {0} using zone names: {1}'.format(domain, domain_name_guesses))

    def _handle_http_error(self, e: HTTPError, domain_name: str) -> Optional[errors.PluginError]:
        if False:
            i = 10
            return i + 15
        return errors.PluginError('Error determining zone identifier for {0}: {1}.'.format(domain_name, e))

    def _handle_general_error(self, e: Exception, domain_name: str) -> Optional[errors.PluginError]:
        if False:
            return 10
        if not str(e).startswith('No domain found'):
            return errors.PluginError('Unexpected error determining zone identifier for {0}: {1}'.format(domain_name, e))
        return None

def build_lexicon_config(lexicon_provider_name: str, lexicon_options: Mapping[str, Any], provider_options: Mapping[str, Any]) -> Union[ConfigResolver, Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    '\n    Convenient function to build a Lexicon 2.x/3.x config object.\n\n    :param str lexicon_provider_name: the name of the lexicon provider to use\n    :param dict lexicon_options: options specific to lexicon\n    :param dict provider_options: options specific to provider\n    :return: configuration to apply to the provider\n    :rtype: ConfigurationResolver or dict\n\n    .. deprecated:: 2.7.0\n       Please use certbot.plugins.dns_common_lexicon.LexiconDNSAuthenticator instead.\n    '
    config_dict: Dict[str, Any] = {'provider_name': lexicon_provider_name}
    config_dict.update(lexicon_options)
    if ConfigResolver is None:
        config_dict.update(provider_options)
        return config_dict
    else:
        provider_config: Dict[str, Any] = {}
        provider_config.update(provider_options)
        config_dict[lexicon_provider_name] = provider_config
        return ConfigResolver().with_dict(config_dict).with_env()

class LexiconDNSAuthenticator(dns_common.DNSAuthenticator):
    """
    Base class for a DNS authenticator that uses Lexicon client
    as backend to execute DNS record updates
    """

    def __init__(self, config: configuration.NamespaceConfig, name: str):
        if False:
            return 10
        super().__init__(config, name)
        self._provider_options: List[Tuple[str, str, str]] = []
        self._credentials: dns_common.CredentialsConfiguration

    @property
    @abc.abstractmethod
    def _provider_name(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        The name of the Lexicon provider to use\n        '

    @property
    def _ttl(self) -> int:
        if False:
            return 10
        '\n        Time to live to apply to the DNS records created by this Authenticator\n        '
        return 60

    def _add_provider_option(self, creds_var_name: str, creds_var_label: str, lexicon_provider_option_name: str) -> None:
        if False:
            i = 10
            return i + 15
        self._provider_options.append((creds_var_name, creds_var_label, lexicon_provider_option_name))

    def _build_lexicon_config(self, domain: str) -> ConfigResolver:
        if False:
            while True:
                i = 10
        if not hasattr(self, '_credentials'):
            self._setup_credentials()
        dict_config = {'domain': domain, 'delegated': domain, 'provider_name': self._provider_name, 'ttl': self._ttl, self._provider_name: {item[2]: self._credentials.conf(item[0]) for item in self._provider_options}}
        return ConfigResolver().with_dict(dict_config).with_env()

    def _setup_credentials(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._credentials = self._configure_credentials(key='credentials', label=f'Credentials INI file for {self._provider_name} DNS authenticator', required_variables={item[0]: item[1] for item in self._provider_options})

    def _perform(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            i = 10
            return i + 15
        resolved_domain = self._resolve_domain(domain)
        try:
            with Client(self._build_lexicon_config(resolved_domain)) as operations:
                operations.create_record(rtype='TXT', name=validation_name, content=validation)
        except RequestException as e:
            logger.debug('Encountered error adding TXT record: %s', e, exc_info=True)
            raise errors.PluginError('Error adding TXT record: {0}'.format(e))

    def _cleanup(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            i = 10
            return i + 15
        try:
            resolved_domain = self._resolve_domain(domain)
        except errors.PluginError as e:
            logger.debug('Encountered error finding domain_id during deletion: %s', e, exc_info=True)
            return
        try:
            with Client(self._build_lexicon_config(resolved_domain)) as operations:
                operations.delete_record(rtype='TXT', name=validation_name, content=validation)
        except RequestException as e:
            logger.debug('Encountered error deleting TXT record: %s', e, exc_info=True)

    def _resolve_domain(self, domain: str) -> str:
        if False:
            return 10
        domain_name_guesses = dns_common.base_domain_name_guesses(domain)
        for domain_name in domain_name_guesses:
            try:
                with Client(self._build_lexicon_config(domain_name)):
                    return domain_name
            except HTTPError as e:
                result1 = self._handle_http_error(e, domain_name)
                if result1:
                    raise result1
            except Exception as e:
                result2 = self._handle_general_error(e, domain_name)
                if result2:
                    raise result2
        raise errors.PluginError('Unable to determine zone identifier for {0} using zone names: {1}'.format(domain, domain_name_guesses))

    def _handle_http_error(self, e: HTTPError, domain_name: str) -> Optional[errors.PluginError]:
        if False:
            print('Hello World!')
        return errors.PluginError('Error determining zone identifier for {0}: {1}.'.format(domain_name, e))

    def _handle_general_error(self, e: Exception, domain_name: str) -> Optional[errors.PluginError]:
        if False:
            for i in range(10):
                print('nop')
        if not str(e).startswith('No domain found'):
            return errors.PluginError('Unexpected error determining zone identifier for {0}: {1}'.format(domain_name, e))
        return None

class _DeprecationModule:
    """
    Internal class delegating to a module, and displaying warnings when attributes
    related to deprecated attributes in the current module.
    """

    def __init__(self, module: ModuleType):
        if False:
            print('Hello World!')
        self.__dict__['_module'] = module

    def __getattr__(self, attr: str) -> Any:
        if False:
            return 10
        if attr in ('LexiconClient', 'build_lexicon_config'):
            warnings.warn(f'{attr} attribute in {__name__} module is deprecated and will be removed soon.', DeprecationWarning, stacklevel=2)
        return getattr(self._module, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if False:
            return 10
        setattr(self._module, attr, value)

    def __delattr__(self, attr: str) -> Any:
        if False:
            print('Hello World!')
        delattr(self._module, attr)

    def __dir__(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        return ['_module'] + dir(self._module)
sys.modules[__name__] = cast(ModuleType, _DeprecationModule(sys.modules[__name__]))