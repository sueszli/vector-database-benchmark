"""New interface style Certbot enhancements"""
import abc
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from certbot import configuration
from certbot import interfaces
from certbot._internal import constants
ENHANCEMENTS = ['redirect', 'ensure-http-header', 'ocsp-stapling']
'List of possible :class:`certbot.interfaces.Installer`\nenhancements.\n\nList of expected options parameters:\n- redirect: None\n- ensure-http-header: name of header (i.e. Strict-Transport-Security)\n- ocsp-stapling: certificate chain file path\n\n'

def enabled_enhancements(config: configuration.NamespaceConfig) -> Generator[Dict[str, Any], None, None]:
    if False:
        print('Hello World!')
    '\n    Generator to yield the enabled new style enhancements.\n\n    :param config: Configuration.\n    :type config: certbot.configuration.NamespaceConfig\n    '
    for enh in _INDEX:
        if getattr(config, enh['cli_dest']):
            yield enh

def are_requested(config: configuration.NamespaceConfig) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Checks if one or more of the requested enhancements are those of the new\n    enhancement interfaces.\n\n    :param config: Configuration.\n    :type config: certbot.configuration.NamespaceConfig\n    '
    return any(enabled_enhancements(config))

def are_supported(config: configuration.NamespaceConfig, installer: Optional[interfaces.Installer]) -> bool:
    if False:
        print('Hello World!')
    '\n    Checks that all of the requested enhancements are supported by the\n    installer.\n\n    :param config: Configuration.\n    :type config: certbot.configuration.NamespaceConfig\n\n    :param installer: Installer object\n    :type installer: interfaces.Installer\n\n    :returns: If all the requested enhancements are supported by the installer\n    :rtype: bool\n    '
    for enh in enabled_enhancements(config):
        if not isinstance(installer, enh['class']):
            return False
    return True

def enable(lineage: Optional[interfaces.RenewableCert], domains: Iterable[str], installer: Optional[interfaces.Installer], config: configuration.NamespaceConfig) -> None:
    if False:
        return 10
    '\n    Run enable method for each requested enhancement that is supported.\n\n    :param lineage: Certificate lineage object\n    :type lineage: certbot.interfaces.RenewableCert\n\n    :param domains: List of domains in certificate to enhance\n    :type domains: str\n\n    :param installer: Installer object\n    :type installer: interfaces.Installer\n\n    :param config: Configuration.\n    :type config: certbot.configuration.NamespaceConfig\n    '
    if installer:
        for enh in enabled_enhancements(config):
            getattr(installer, enh['enable_function'])(lineage, domains)

def populate_cli(add: Callable[..., None]) -> None:
    if False:
        return 10
    '\n    Populates the command line flags for certbot._internal.cli.HelpfulParser\n\n    :param add: Add function of certbot._internal.cli.HelpfulParser\n    :type add: func\n    '
    for enh in _INDEX:
        add(enh['cli_groups'], enh['cli_flag'], action=enh['cli_action'], dest=enh['cli_dest'], default=enh['cli_flag_default'], help=enh['cli_help'])

class AutoHSTSEnhancement(object, metaclass=abc.ABCMeta):
    """
    Enhancement interface that installer plugins can implement in order to
    provide functionality that configures the software to have a
    'Strict-Transport-Security' with initially low max-age value that will
    increase over time.

    The plugins implementing new style enhancements are responsible of handling
    the saving of configuration checkpoints as well as calling possible restarts
    of managed software themselves. For update_autohsts method, the installer may
    have to call prepare() to finalize the plugin initialization.

    Methods:
        enable_autohsts is called when the header is initially installed using a
        low max-age value.

        update_autohsts is called every time when Certbot is run using 'renew'
        verb. The max-age value should be increased over time using this method.

        deploy_autohsts is called for every lineage that has had its certificate
        renewed. A long HSTS max-age value should be set here, as we should be
        confident that the user is able to automatically renew their certificates.


    """

    @abc.abstractmethod
    def update_autohsts(self, lineage: interfaces.RenewableCert, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        "\n        Gets called for each lineage every time Certbot is run with 'renew' verb.\n        Implementation of this method should increase the max-age value.\n\n        :param lineage: Certificate lineage object\n        :type lineage: certbot.interfaces.RenewableCert\n\n        .. note:: prepare() method inherited from `interfaces.Plugin` might need\n            to be called manually within implementation of this interface method\n            to finalize the plugin initialization.\n        "

    @abc.abstractmethod
    def deploy_autohsts(self, lineage: interfaces.RenewableCert, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Gets called for a lineage when its certificate is successfully renewed.\n        Long max-age value should be set in implementation of this method.\n\n        :param lineage: Certificate lineage object\n        :type lineage: certbot.interfaces.RenewableCert\n        '

    @abc.abstractmethod
    def enable_autohsts(self, lineage: Optional[interfaces.RenewableCert], domains: Iterable[str], *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        '\n        Enables the AutoHSTS enhancement, installing\n        Strict-Transport-Security header with a low initial value to be increased\n        over the subsequent runs of Certbot renew.\n\n        :param lineage: Certificate lineage object\n        :type lineage: certbot.interfaces.RenewableCert\n\n        :param domains: List of domains in certificate to enhance\n        :type domains: `list` of `str`\n        '
_INDEX: List[Dict[str, Any]] = [{'name': 'AutoHSTS', 'cli_help': 'Gradually increasing max-age value for HTTP Strict Transport ' + 'Security security header', 'cli_flag': '--auto-hsts', 'cli_flag_default': constants.CLI_DEFAULTS['auto_hsts'], 'cli_groups': ['security', 'enhance'], 'cli_dest': 'auto_hsts', 'cli_action': 'store_true', 'class': AutoHSTSEnhancement, 'updater_function': 'update_autohsts', 'deployer_function': 'deploy_autohsts', 'enable_function': 'enable_autohsts'}]