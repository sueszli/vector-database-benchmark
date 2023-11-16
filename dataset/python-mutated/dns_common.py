"""Common code for DNS Authenticator Plugins."""
import abc
import logging
from time import sleep
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Type
import configobj
from acme import challenges
from certbot import achallenges
from certbot import configuration
from certbot import errors
from certbot import interfaces
from certbot.compat import filesystem
from certbot.compat import os
from certbot.display import ops
from certbot.display import util as display_util
from certbot.plugins import common
logger = logging.getLogger(__name__)

class DNSAuthenticator(common.Plugin, interfaces.Authenticator, metaclass=abc.ABCMeta):
    """Base class for DNS Authenticators"""

    def __init__(self, config: configuration.NamespaceConfig, name: str) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(config, name)
        self._attempt_cleanup = False

    @classmethod
    def add_parser_arguments(cls, add: Callable[..., None], default_propagation_seconds: int=10) -> None:
        if False:
            while True:
                i = 10
        add('propagation-seconds', default=default_propagation_seconds, type=int, help='The number of seconds to wait for DNS to propagate before asking the ACME server to verify the DNS record.')

    def auth_hint(self, failed_achalls: List[achallenges.AnnotatedChallenge]) -> str:
        if False:
            return 10
        'See certbot.plugins.common.Plugin.auth_hint.'
        delay = self.conf('propagation-seconds')
        return 'The Certificate Authority failed to verify the DNS TXT records created by --{name}. Ensure the above domains are hosted by this DNS provider, or try increasing --{name}-propagation-seconds (currently {secs} second{suffix}).'.format(name=self.name, secs=delay, suffix='s' if delay != 1 else '')

    def get_chall_pref(self, unused_domain: str) -> Iterable[Type[challenges.Challenge]]:
        if False:
            while True:
                i = 10
        return [challenges.DNS01]

    def prepare(self) -> None:
        if False:
            return 10
        pass

    def more_info(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def perform(self, achalls: List[achallenges.AnnotatedChallenge]) -> List[challenges.ChallengeResponse]:
        if False:
            return 10
        self._setup_credentials()
        self._attempt_cleanup = True
        responses = []
        for achall in achalls:
            domain = achall.domain
            validation_domain_name = achall.validation_domain_name(domain)
            validation = achall.validation(achall.account_key)
            self._perform(domain, validation_domain_name, validation)
            responses.append(achall.response(achall.account_key))
        display_util.notify('Waiting %d seconds for DNS changes to propagate' % self.conf('propagation-seconds'))
        sleep(self.conf('propagation-seconds'))
        return responses

    def cleanup(self, achalls: List[achallenges.AnnotatedChallenge]) -> None:
        if False:
            return 10
        if self._attempt_cleanup:
            for achall in achalls:
                domain = achall.domain
                validation_domain_name = achall.validation_domain_name(domain)
                validation = achall.validation(achall.account_key)
                self._cleanup(domain, validation_domain_name, validation)

    @abc.abstractmethod
    def _setup_credentials(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Establish credentials, prompting if necessary.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def _perform(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Performs a dns-01 challenge by creating a DNS TXT record.\n\n        :param str domain: The domain being validated.\n        :param str validation_domain_name: The validation record domain name.\n        :param str validation: The validation record content.\n        :raises errors.PluginError: If the challenge cannot be performed\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def _cleanup(self, domain: str, validation_name: str, validation: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Deletes the DNS TXT record which would have been created by `_perform_achall`.\n\n        Fails gracefully if no such record exists.\n\n        :param str domain: The domain being validated.\n        :param str validation_domain_name: The validation record domain name.\n        :param str validation: The validation record content.\n        '
        raise NotImplementedError()

    def _configure(self, key: str, label: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that a configuration value is available.\n\n        If necessary, prompts the user and stores the result.\n\n        :param str key: The configuration key.\n        :param str label: The user-friendly label for this piece of information.\n        '
        configured_value = self.conf(key)
        if not configured_value:
            new_value = self._prompt_for_data(label)
            setattr(self.config, self.dest(key), new_value)

    def _configure_file(self, key: str, label: str, validator: Optional[Callable[[str], None]]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Ensure that a configuration value is available for a path.\n\n        If necessary, prompts the user and stores the result.\n\n        :param str key: The configuration key.\n        :param str label: The user-friendly label for this piece of information.\n        '
        configured_value = self.conf(key)
        if not configured_value:
            new_value = self._prompt_for_file(label, validator)
            setattr(self.config, self.dest(key), os.path.abspath(os.path.expanduser(new_value)))

    def _configure_credentials(self, key: str, label: str, required_variables: Optional[Mapping[str, str]]=None, validator: Optional[Callable[['CredentialsConfiguration'], None]]=None) -> 'CredentialsConfiguration':
        if False:
            print('Hello World!')
        '\n        As `_configure_file`, but for a credential configuration file.\n\n        If necessary, prompts the user and stores the result.\n\n        Always stores absolute paths to avoid issues during renewal.\n\n        :param str key: The configuration key.\n        :param str label: The user-friendly label for this piece of information.\n        :param dict required_variables: Map of variable which must be present to error to display.\n        :param callable validator: A method which will be called to validate the\n            `CredentialsConfiguration` resulting from the supplied input after it has been validated\n            to contain the `required_variables`. Should throw a `~certbot.errors.PluginError` to\n            indicate any issue.\n        '

        def __validator(filename: str) -> None:
            if False:
                i = 10
                return i + 15
            applied_configuration = CredentialsConfiguration(filename, self.dest)
            if required_variables:
                applied_configuration.require(required_variables)
            if validator:
                validator(applied_configuration)
        self._configure_file(key, label, __validator)
        credentials_configuration = CredentialsConfiguration(self.conf(key), self.dest)
        if required_variables:
            credentials_configuration.require(required_variables)
        if validator:
            validator(credentials_configuration)
        return credentials_configuration

    @staticmethod
    def _prompt_for_data(label: str) -> str:
        if False:
            print('Hello World!')
        "\n        Prompt the user for a piece of information.\n\n        :param str label: The user-friendly label for this piece of information.\n        :returns: The user's response (guaranteed non-empty).\n        :rtype: str\n        "

        def __validator(i: str) -> None:
            if False:
                return 10
            if not i:
                raise errors.PluginError('Please enter your {0}.'.format(label))
        (code, response) = ops.validated_input(__validator, 'Input your {0}'.format(label), force_interactive=True)
        if code == display_util.OK:
            return response
        raise errors.PluginError('{0} required to proceed.'.format(label))

    @staticmethod
    def _prompt_for_file(label: str, validator: Optional[Callable[[str], None]]=None) -> str:
        if False:
            print('Hello World!')
        "\n        Prompt the user for a path.\n\n        :param str label: The user-friendly label for the file.\n        :param callable validator: A method which will be called to validate the supplied input\n            after it has been validated to be a non-empty path to an existing file. Should throw a\n            `~certbot.errors.PluginError` to indicate any issue.\n        :returns: The user's response (guaranteed to exist).\n        :rtype: str\n        "

        def __validator(filename: str) -> None:
            if False:
                print('Hello World!')
            if not filename:
                raise errors.PluginError('Please enter a valid path to your {0}.'.format(label))
            filename = os.path.expanduser(filename)
            validate_file(filename)
            if validator:
                validator(filename)
        (code, response) = ops.validated_directory(__validator, 'Input the path to your {0}'.format(label), force_interactive=True)
        if code == display_util.OK:
            return response
        raise errors.PluginError('{0} required to proceed.'.format(label))

class CredentialsConfiguration:
    """Represents a user-supplied filed which stores API credentials."""

    def __init__(self, filename: str, mapper: Callable[[str], str]=lambda x: x) -> None:
        if False:
            while True:
                i = 10
        '\n        :param str filename: A path to the configuration file.\n        :param callable mapper: A transformation to apply to configuration key names\n        :raises errors.PluginError: If the file does not exist or is not a valid format.\n        '
        validate_file_permissions(filename)
        try:
            self.confobj = configobj.ConfigObj(filename)
        except configobj.ConfigObjError as e:
            logger.debug("Error parsing credentials configuration '%s': %s", filename, e, exc_info=True)
            raise errors.PluginError("Error parsing credentials configuration '{}': {}".format(filename, e))
        self.mapper = mapper

    def require(self, required_variables: Mapping[str, str]) -> None:
        if False:
            return 10
        'Ensures that the supplied set of variables are all present in the file.\n\n        :param dict required_variables: Map of variable which must be present to error to display.\n        :raises errors.PluginError: If one or more are missing.\n        '
        messages = []
        for var in required_variables:
            if not self._has(var):
                messages.append('Property "{0}" not found (should be {1}).'.format(self.mapper(var), required_variables[var]))
            elif not self._get(var):
                messages.append('Property "{0}" not set (should be {1}).'.format(self.mapper(var), required_variables[var]))
        if messages:
            raise errors.PluginError('Missing {0} in credentials configuration file {1}:\n * {2}'.format('property' if len(messages) == 1 else 'properties', self.confobj.filename, '\n * '.join(messages)))

    def conf(self, var: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'Find a configuration value for variable `var`, as transformed by `mapper`.\n\n        :param str var: The variable to get.\n        :returns: The value of the variable, if it exists.\n        :rtype: str or None\n        '
        return self._get(var)

    def _has(self, var: str) -> bool:
        if False:
            while True:
                i = 10
        return self.mapper(var) in self.confobj

    def _get(self, var: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.confobj.get(self.mapper(var))

def validate_file(filename: str) -> None:
    if False:
        print('Hello World!')
    'Ensure that the specified file exists.'
    if not os.path.exists(filename):
        raise errors.PluginError('File not found: {0}'.format(filename))
    if os.path.isdir(filename):
        raise errors.PluginError('Path is a directory: {0}'.format(filename))

def validate_file_permissions(filename: str) -> None:
    if False:
        print('Hello World!')
    'Ensure that the specified file exists and warn about unsafe permissions.'
    validate_file(filename)
    if filesystem.has_world_permissions(filename):
        logger.warning('Unsafe permissions on credentials configuration file: %s', filename)

def base_domain_name_guesses(domain: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    "Return a list of progressively less-specific domain names.\n\n    One of these will probably be the domain name known to the DNS provider.\n\n    :Example:\n\n    >>> base_domain_name_guesses('foo.bar.baz.example.com')\n    ['foo.bar.baz.example.com', 'bar.baz.example.com', 'baz.example.com', 'example.com', 'com']\n\n    :param str domain: The domain for which to return guesses.\n    :returns: The a list of less specific domain names.\n    :rtype: list\n    "
    fragments = domain.split('.')
    return ['.'.join(fragments[i:]) for i in range(0, len(fragments))]