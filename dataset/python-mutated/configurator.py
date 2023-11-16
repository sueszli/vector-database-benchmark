"""Nginx Configuration"""
import atexit
from contextlib import ExitStack
import logging
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union
import OpenSSL
from acme import challenges
from acme import crypto_util as acme_crypto_util
from certbot import achallenges
from certbot import crypto_util
from certbot import errors
from certbot import util
from certbot.compat import os
from certbot.display import util as display_util
from certbot.plugins import common
from certbot_nginx._internal import constants
from certbot_nginx._internal import display_ops
from certbot_nginx._internal import http_01
from certbot_nginx._internal import nginxparser
from certbot_nginx._internal import obj
from certbot_nginx._internal import parser
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
NAME_RANK = 0
START_WILDCARD_RANK = 1
END_WILDCARD_RANK = 2
REGEX_RANK = 3
NO_SSL_MODIFIER = 4
logger = logging.getLogger(__name__)

class NginxConfigurator(common.Configurator):
    """Nginx configurator.

    .. todo:: Add proper support for comments in the config. Currently,
        config files modified by the configurator will lose all their comments.

    :ivar config: Configuration.
    :type config: certbot.configuration.NamespaceConfig

    :ivar parser: Handles low level parsing
    :type parser: :class:`~certbot_nginx._internal.parser`

    :ivar str save_notes: Human-readable config change notes

    :ivar reverter: saves and reverts checkpoints
    :type reverter: :class:`certbot.reverter.Reverter`

    :ivar tup version: version of Nginx

    """
    description = 'Nginx Web Server plugin'
    DEFAULT_LISTEN_PORT = '80'
    SSL_DIRECTIVES = ['ssl_certificate', 'ssl_certificate_key', 'ssl_dhparam']

    @classmethod
    def add_parser_arguments(cls, add: Callable[..., None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        default_server_root = _determine_default_server_root()
        add('server-root', default=constants.CLI_DEFAULTS['server_root'], help='Nginx server root directory. (default: %s)' % default_server_root)
        add('ctl', default=constants.CLI_DEFAULTS['ctl'], help="Path to the 'nginx' binary, used for 'configtest' and retrieving nginx version number.")
        add('sleep-seconds', default=constants.CLI_DEFAULTS['sleep_seconds'], type=int, help='Number of seconds to wait for nginx configuration changes to apply when reloading.')

    @property
    def nginx_conf(self) -> str:
        if False:
            i = 10
            return i + 15
        'Nginx config file path.'
        return os.path.join(self.conf('server_root'), 'nginx.conf')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize an Nginx Configurator.\n\n        :param tup version: version of Nginx as a tuple (1, 4, 7)\n            (used mostly for unittesting)\n\n        :param tup openssl_version: version of OpenSSL linked to Nginx as a tuple (1, 4, 7)\n            (used mostly for unittesting)\n\n        '
        version = kwargs.pop('version', None)
        openssl_version = kwargs.pop('openssl_version', None)
        super().__init__(*args, **kwargs)
        self.save_notes = ''
        self.new_vhost: Optional[obj.VirtualHost] = None
        self._wildcard_vhosts: Dict[str, List[obj.VirtualHost]] = {}
        self._wildcard_redirect_vhosts: Dict[str, List[obj.VirtualHost]] = {}
        self._chall_out = 0
        self.version = version
        self.openssl_version = openssl_version
        self._enhance_func = {'redirect': self._enable_redirect, 'ensure-http-header': self._set_http_header, 'staple-ocsp': self._enable_ocsp_stapling}
        self.reverter.recovery_routine()
        self.parser: parser.NginxParser

    @property
    def mod_ssl_conf_src(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Full absolute path to SSL configuration file source.'
        use_tls13 = self.version >= (1, 13, 0)
        min_openssl_version = util.parse_loose_version('1.0.2l')
        session_tix_off = self.version >= (1, 5, 9) and self.openssl_version and (util.parse_loose_version(self.openssl_version) >= min_openssl_version)
        if use_tls13:
            if session_tix_off:
                config_filename = 'options-ssl-nginx.conf'
            else:
                config_filename = 'options-ssl-nginx-tls13-session-tix-on.conf'
        elif session_tix_off:
            config_filename = 'options-ssl-nginx-tls12-only.conf'
        else:
            config_filename = 'options-ssl-nginx-old.conf'
        file_manager = ExitStack()
        atexit.register(file_manager.close)
        ref = importlib_resources.files('certbot_nginx').joinpath('_internal', 'tls_configs', config_filename)
        return str(file_manager.enter_context(importlib_resources.as_file(ref)))

    @property
    def mod_ssl_conf(self) -> str:
        if False:
            while True:
                i = 10
        'Full absolute path to SSL configuration file.'
        return os.path.join(self.config.config_dir, constants.MOD_SSL_CONF_DEST)

    @property
    def updated_mod_ssl_conf_digest(self) -> str:
        if False:
            while True:
                i = 10
        'Full absolute path to digest of updated SSL configuration file.'
        return os.path.join(self.config.config_dir, constants.UPDATED_MOD_SSL_CONF_DIGEST)

    def install_ssl_options_conf(self, options_ssl: str, options_ssl_digest: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Copy Certbot's SSL options file into the system's config dir if required."
        common.install_version_controlled_file(options_ssl, options_ssl_digest, self.mod_ssl_conf_src, constants.ALL_SSL_OPTIONS_HASHES)

    def prepare(self) -> None:
        if False:
            while True:
                i = 10
        'Prepare the authenticator/installer.\n\n        :raises .errors.NoInstallationError: If Nginx ctl cannot be found\n        :raises .errors.MisconfigurationError: If Nginx is misconfigured\n        '
        if not util.exe_exists(self.conf('ctl')):
            raise errors.NoInstallationError("Could not find a usable 'nginx' binary. Ensure nginx exists, the binary is executable, and your PATH is set correctly.")
        self.config_test()
        self.parser = parser.NginxParser(self.conf('server-root'))
        if self.version is None:
            self.version = self.get_version()
        if self.openssl_version is None:
            self.openssl_version = self._get_openssl_version()
        self.install_ssl_options_conf(self.mod_ssl_conf, self.updated_mod_ssl_conf_digest)
        self.install_ssl_dhparams()
        try:
            util.lock_dir_until_exit(self.conf('server-root'))
        except (OSError, errors.LockError):
            logger.debug('Encountered error:', exc_info=True)
            raise errors.PluginError('Unable to lock {0}'.format(self.conf('server-root')))

    def deploy_cert(self, domain: str, cert_path: str, key_path: str, chain_path: str, fullchain_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Deploys certificate to specified virtual host.\n\n        .. note:: Aborts if the vhost is missing ssl_certificate or\n            ssl_certificate_key.\n\n        .. note:: This doesn't save the config files!\n\n        :raises errors.PluginError: When unable to deploy certificate due to\n            a lack of directives or configuration\n\n        "
        if not fullchain_path:
            raise errors.PluginError('The nginx plugin currently requires --fullchain-path to install a certificate.')
        vhosts = self.choose_vhosts(domain, create_if_no_match=True)
        for vhost in vhosts:
            self._deploy_cert(vhost, cert_path, key_path, chain_path, fullchain_path)
            display_util.notify('Successfully deployed certificate for {} to {}'.format(domain, vhost.filep))

    def _deploy_cert(self, vhost: obj.VirtualHost, _cert_path: str, key_path: str, _chain_path: str, fullchain_path: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Helper function for deploy_cert() that handles the actual deployment\n        this exists because we might want to do multiple deployments per\n        domain originally passed for deploy_cert(). This is especially true\n        with wildcard certificates\n        '
        cert_directives = [['\n    ', 'ssl_certificate', ' ', fullchain_path], ['\n    ', 'ssl_certificate_key', ' ', key_path]]
        self.parser.update_or_add_server_directives(vhost, cert_directives)
        logger.info('Deploying Certificate to VirtualHost %s', vhost.filep)
        self.save_notes += 'Changed vhost at %s with addresses of %s\n' % (vhost.filep, ', '.join((str(addr) for addr in vhost.addrs)))
        self.save_notes += '\tssl_certificate %s\n' % fullchain_path
        self.save_notes += '\tssl_certificate_key %s\n' % key_path

    def _choose_vhosts_wildcard(self, domain: str, prefer_ssl: bool, no_ssl_filter_port: Optional[str]=None) -> List[obj.VirtualHost]:
        if False:
            return 10
        'Prompts user to choose vhosts to install a wildcard certificate for'
        if prefer_ssl:
            vhosts_cache = self._wildcard_vhosts

            def preference_test(x: obj.VirtualHost) -> bool:
                if False:
                    i = 10
                    return i + 15
                return x.ssl
        else:
            vhosts_cache = self._wildcard_redirect_vhosts

            def preference_test(x: obj.VirtualHost) -> bool:
                if False:
                    i = 10
                    return i + 15
                return not x.ssl
        if domain in vhosts_cache:
            return vhosts_cache[domain]
        vhosts = self.parser.get_vhosts()
        filtered_vhosts = {}
        for vhost in vhosts:
            if no_ssl_filter_port is not None:
                if not self._vhost_listening_on_port_no_ssl(vhost, no_ssl_filter_port):
                    continue
            for name in vhost.names:
                if preference_test(vhost):
                    filtered_vhosts[name] = vhost
                elif name not in filtered_vhosts:
                    filtered_vhosts[name] = vhost
        dialog_input = set(filtered_vhosts.values())
        return_vhosts = display_ops.select_vhost_multiple(list(dialog_input))
        for vhost in return_vhosts:
            if domain not in vhosts_cache:
                vhosts_cache[domain] = []
            vhosts_cache[domain].append(vhost)
        return return_vhosts

    def _choose_vhost_single(self, target_name: str) -> List[obj.VirtualHost]:
        if False:
            i = 10
            return i + 15
        matches = self._get_ranked_matches(target_name)
        vhosts = [x for x in [self._select_best_name_match(matches)] if x is not None]
        return vhosts

    def choose_vhosts(self, target_name: str, create_if_no_match: bool=False) -> List[obj.VirtualHost]:
        if False:
            print('Hello World!')
        "Chooses a virtual host based on the given domain name.\n\n        .. note:: This makes the vhost SSL-enabled if it isn't already. Follows\n            Nginx's server block selection rules preferring blocks that are\n            already SSL.\n\n        .. todo:: This should maybe return list if no obvious answer\n            is presented.\n\n        :param str target_name: domain name\n        :param bool create_if_no_match: If we should create a new vhost from default\n            when there is no match found. If we can't choose a default, raise a\n            MisconfigurationError.\n\n        :returns: ssl vhosts associated with name\n        :rtype: list of :class:`~certbot_nginx._internal.obj.VirtualHost`\n\n        "
        if util.is_wildcard_domain(target_name):
            vhosts = self._choose_vhosts_wildcard(target_name, prefer_ssl=True)
        else:
            vhosts = self._choose_vhost_single(target_name)
        if not vhosts:
            if create_if_no_match:
                vhosts = [self._vhost_from_duplicated_default(target_name, True, str(self.config.https_port))]
            else:
                raise errors.MisconfigurationError('Cannot find a VirtualHost matching domain %s. In order for Certbot to correctly perform the challenge please add a corresponding server_name directive to your nginx configuration for every domain on your certificate: https://nginx.org/en/docs/http/server_names.html' % target_name)
        for vhost in vhosts:
            if not vhost.ssl:
                self._make_server_ssl(vhost)
        return vhosts

    def ipv6_info(self, port: str) -> Tuple[bool, bool]:
        if False:
            print('Hello World!')
        'Returns tuple of booleans (ipv6_active, ipv6only_present)\n        ipv6_active is true if any server block listens ipv6 address in any port\n\n        ipv6only_present is true if ipv6only=on option exists in any server\n        block ipv6 listen directive for the specified port.\n\n        :param str port: Port to check ipv6only=on directive for\n\n        :returns: Tuple containing information if IPv6 is enabled in the global\n            configuration, and existence of ipv6only directive for specified port\n        :rtype: tuple of type (bool, bool)\n        '
        vhosts = self.parser.get_vhosts()
        ipv6_active = False
        ipv6only_present = False
        for vh in vhosts:
            for addr in vh.addrs:
                if addr.ipv6:
                    ipv6_active = True
                if addr.ipv6only and addr.get_port() == port:
                    ipv6only_present = True
        return (ipv6_active, ipv6only_present)

    def _vhost_from_duplicated_default(self, domain: str, allow_port_mismatch: bool, port: str) -> obj.VirtualHost:
        if False:
            return 10
        'if allow_port_mismatch is False, only server blocks with matching ports will be\n           used as a default server block template.\n        '
        assert self.parser is not None
        if self.new_vhost is None:
            default_vhost = self._get_default_vhost(domain, allow_port_mismatch, port)
            self.new_vhost = self.parser.duplicate_vhost(default_vhost, remove_singleton_listen_params=True)
            self.new_vhost.names = set()
        self._add_server_name_to_vhost(self.new_vhost, domain)
        return self.new_vhost

    def _add_server_name_to_vhost(self, vhost: obj.VirtualHost, domain: str) -> None:
        if False:
            while True:
                i = 10
        vhost.names.add(domain)
        name_block = [['\n    ', 'server_name']]
        for name in vhost.names:
            name_block[0].append(' ')
            name_block[0].append(name)
        self.parser.update_or_add_server_directives(vhost, name_block)

    def _get_default_vhost(self, domain: str, allow_port_mismatch: bool, port: str) -> obj.VirtualHost:
        if False:
            return 10
        'Helper method for _vhost_from_duplicated_default; see argument documentation there'
        vhost_list = self.parser.get_vhosts()
        all_default_vhosts = []
        port_matching_vhosts = []
        for vhost in vhost_list:
            for addr in vhost.addrs:
                if addr.default:
                    all_default_vhosts.append(vhost)
                    if self._port_matches(port, addr.get_port()):
                        port_matching_vhosts.append(vhost)
                    break
        if len(port_matching_vhosts) == 1:
            return port_matching_vhosts[0]
        elif len(all_default_vhosts) == 1 and allow_port_mismatch:
            return all_default_vhosts[0]
        raise errors.MisconfigurationError(f'Could not automatically find a matching server block for {domain}. Set the `server_name` directive to use the Nginx installer.')

    def _get_ranked_matches(self, target_name: str) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a ranked list of vhosts that match target_name.\n        The ranking gives preference to SSL vhosts.\n\n        :param str target_name: The name to match\n        :returns: list of dicts containing the vhost, the matching name, and\n            the numerical rank\n        :rtype: list\n\n        '
        vhost_list = self.parser.get_vhosts()
        return self._rank_matches_by_name_and_ssl(vhost_list, target_name)

    def _select_best_name_match(self, matches: Sequence[Mapping[str, Any]]) -> Optional[obj.VirtualHost]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the best name match of a ranked list of vhosts.\n\n        :param list matches: list of dicts containing the vhost, the matching name,\n            and the numerical rank\n        :returns: the most matching vhost\n        :rtype: :class:`~certbot_nginx._internal.obj.VirtualHost`\n\n        '
        if not matches:
            return None
        elif matches[0]['rank'] in [START_WILDCARD_RANK, END_WILDCARD_RANK, START_WILDCARD_RANK + NO_SSL_MODIFIER, END_WILDCARD_RANK + NO_SSL_MODIFIER]:
            rank = matches[0]['rank']
            wildcards = [x for x in matches if x['rank'] == rank]
            return max(wildcards, key=lambda x: len(x['name']))['vhost']
        return matches[0]['vhost']

    def _rank_matches_by_name(self, vhost_list: Iterable[obj.VirtualHost], target_name: str) -> List[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        'Returns a ranked list of vhosts from vhost_list that match target_name.\n        This method should always be followed by a call to _select_best_name_match.\n\n        :param list vhost_list: list of vhosts to filter and rank\n        :param str target_name: The name to match\n        :returns: list of dicts containing the vhost, the matching name, and\n            the numerical rank\n        :rtype: list\n\n        '
        matches = []
        for vhost in vhost_list:
            (name_type, name) = parser.get_best_match(target_name, vhost.names)
            if name_type == 'exact':
                matches.append({'vhost': vhost, 'name': name, 'rank': NAME_RANK})
            elif name_type == 'wildcard_start':
                matches.append({'vhost': vhost, 'name': name, 'rank': START_WILDCARD_RANK})
            elif name_type == 'wildcard_end':
                matches.append({'vhost': vhost, 'name': name, 'rank': END_WILDCARD_RANK})
            elif name_type == 'regex':
                matches.append({'vhost': vhost, 'name': name, 'rank': REGEX_RANK})
        return sorted(matches, key=lambda x: x['rank'])

    def _rank_matches_by_name_and_ssl(self, vhost_list: Iterable[obj.VirtualHost], target_name: str) -> List[Dict[str, Any]]:
        if False:
            return 10
        'Returns a ranked list of vhosts from vhost_list that match target_name.\n        The ranking gives preference to SSLishness before name match level.\n\n        :param list vhost_list: list of vhosts to filter and rank\n        :param str target_name: The name to match\n        :returns: list of dicts containing the vhost, the matching name, and\n            the numerical rank\n        :rtype: list\n\n        '
        matches = self._rank_matches_by_name(vhost_list, target_name)
        for match in matches:
            if not match['vhost'].ssl:
                match['rank'] += NO_SSL_MODIFIER
        return sorted(matches, key=lambda x: x['rank'])

    def choose_redirect_vhosts(self, target_name: str, port: str) -> List[obj.VirtualHost]:
        if False:
            while True:
                i = 10
        'Chooses a single virtual host for redirect enhancement.\n\n        Chooses the vhost most closely matching target_name that is\n        listening to port without using ssl.\n\n        .. todo:: This should maybe return list if no obvious answer\n            is presented.\n\n        .. todo:: The special name "$hostname" corresponds to the machine\'s\n            hostname. Currently we just ignore this.\n\n        :param str target_name: domain name\n        :param str port: port number\n\n        :returns: vhosts associated with name\n        :rtype: list of :class:`~certbot_nginx._internal.obj.VirtualHost`\n\n        '
        if util.is_wildcard_domain(target_name):
            vhosts = self._choose_vhosts_wildcard(target_name, prefer_ssl=False, no_ssl_filter_port=port)
        else:
            matches = self._get_redirect_ranked_matches(target_name, port)
            vhosts = [x for x in [self._select_best_name_match(matches)] if x is not None]
        return vhosts

    def choose_auth_vhosts(self, target_name: str) -> Tuple[List[obj.VirtualHost], List[obj.VirtualHost]]:
        if False:
            i = 10
            return i + 15
        'Returns a list of HTTP and HTTPS vhosts with a server_name matching target_name.\n\n        If no HTTP vhost exists, one will be cloned from the default vhost. If that fails, no HTTP\n        vhost will be returned.\n\n        :param str target_name: non-wildcard domain name\n\n        :returns: tuple of HTTP and HTTPS virtualhosts\n        :rtype: tuple of :class:`~certbot_nginx._internal.obj.VirtualHost`\n\n        '
        vhosts = [m['vhost'] for m in self._get_ranked_matches(target_name) if m and 'vhost' in m]
        http_vhosts = [vh for vh in vhosts if self._vhost_listening(vh, str(self.config.http01_port), False)]
        https_vhosts = [vh for vh in vhosts if self._vhost_listening(vh, str(self.config.https_port), True)]
        if not http_vhosts:
            try:
                http_vhosts = [self._vhost_from_duplicated_default(target_name, False, str(self.config.http01_port))]
            except errors.MisconfigurationError:
                http_vhosts = []
        return (http_vhosts, https_vhosts)

    def _port_matches(self, test_port: str, matching_port: str) -> bool:
        if False:
            return 10
        if matching_port == '' or matching_port is None:
            return test_port == self.DEFAULT_LISTEN_PORT
        return test_port == matching_port

    def _vhost_listening(self, vhost: obj.VirtualHost, port: str, ssl: bool) -> bool:
        if False:
            i = 10
            return i + 15
        'Tests whether a vhost has an address listening on a port with SSL enabled or disabled.\n\n        :param `obj.VirtualHost` vhost: The vhost whose addresses will be tested\n        :param port str: The port number as a string that the address should be bound to\n        :param bool ssl: Whether SSL should be enabled or disabled on the address\n\n        :returns: Whether the vhost has an address listening on the port and protocol.\n        :rtype: bool\n\n        '
        assert self.parser is not None
        all_addrs_are_ssl = self.parser.has_ssl_on_directive(vhost)

        def _ssl_matches(addr: obj.Addr) -> bool:
            if False:
                print('Hello World!')
            return addr.ssl or all_addrs_are_ssl if ssl else not addr.ssl and (not all_addrs_are_ssl)
        if not vhost.addrs:
            return port == self.DEFAULT_LISTEN_PORT and ssl == all_addrs_are_ssl
        return any((self._port_matches(port, addr.get_port()) and _ssl_matches(addr) for addr in vhost.addrs))

    def _vhost_listening_on_port_no_ssl(self, vhost: obj.VirtualHost, port: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._vhost_listening(vhost, port, False)

    def _get_redirect_ranked_matches(self, target_name: str, port: str) -> List[Dict[str, Any]]:
        if False:
            while True:
                i = 10
        'Gets a ranked list of plaintextish port-listening vhosts matching target_name\n\n        Filter all hosts for those listening on port without using ssl.\n        Rank by how well these match target_name.\n\n        :param str target_name: The name to match\n        :param str port: port number as a string\n        :returns: list of dicts containing the vhost, the matching name, and\n            the numerical rank\n        :rtype: list\n\n        '
        all_vhosts = self.parser.get_vhosts()

        def _vhost_matches(vhost: obj.VirtualHost, port: str) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return self._vhost_listening_on_port_no_ssl(vhost, port)
        matching_vhosts = [vhost for vhost in all_vhosts if _vhost_matches(vhost, port)]
        return self._rank_matches_by_name(matching_vhosts, target_name)

    def get_all_names(self) -> Set[str]:
        if False:
            while True:
                i = 10
        'Returns all names found in the Nginx Configuration.\n\n        :returns: All ServerNames, ServerAliases, and reverse DNS entries for\n                  virtual host addresses\n        :rtype: set\n\n        '
        all_names: Set[str] = set()
        for vhost in self.parser.get_vhosts():
            try:
                vhost.names.remove('$hostname')
                vhost.names.add(socket.gethostname())
            except KeyError:
                pass
            all_names.update(vhost.names)
            for addr in vhost.addrs:
                host = addr.get_addr()
                if common.hostname_regex.match(host):
                    all_names.add(host)
                elif not common.private_ips_regex.match(host):
                    try:
                        if addr.ipv6:
                            host = addr.get_ipv6_exploded()
                            socket.inet_pton(socket.AF_INET6, host)
                        else:
                            socket.inet_pton(socket.AF_INET, host)
                        all_names.add(socket.gethostbyaddr(host)[0])
                    except (socket.error, socket.herror, socket.timeout):
                        continue
        return util.get_filtered_names(all_names)

    def _get_snakeoil_paths(self) -> Tuple[str, str]:
        if False:
            i = 10
            return i + 15
        'Generate invalid certs that let us create ssl directives for Nginx'
        tmp_dir = os.path.join(self.config.work_dir, 'snakeoil')
        le_key = crypto_util.generate_key(key_size=1024, key_dir=tmp_dir, keyname='key.pem', strict_permissions=self.config.strict_permissions)
        assert le_key.file is not None
        key = OpenSSL.crypto.load_privatekey(OpenSSL.crypto.FILETYPE_PEM, le_key.pem)
        cert = acme_crypto_util.gen_ss_cert(key, domains=[socket.gethostname()])
        cert_pem = OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, cert)
        (cert_file, cert_path) = util.unique_file(os.path.join(tmp_dir, 'cert.pem'), mode='wb')
        with cert_file:
            cert_file.write(cert_pem)
        return (cert_path, le_key.file)

    def _make_server_ssl(self, vhost: obj.VirtualHost) -> None:
        if False:
            while True:
                i = 10
        'Make a server SSL.\n\n        Make a server SSL by adding new listen and SSL directives.\n\n        :param vhost: The vhost to add SSL to.\n        :type vhost: :class:`~certbot_nginx._internal.obj.VirtualHost`\n\n        '
        https_port = self.config.https_port
        ipv6info = self.ipv6_info(str(https_port))
        ipv6_block = ['']
        ipv4_block = ['']
        if not vhost.addrs:
            listen_block = [['\n    ', 'listen', ' ', self.DEFAULT_LISTEN_PORT]]
            self.parser.add_server_directives(vhost, listen_block)
        if vhost.ipv6_enabled():
            ipv6_block = ['\n    ', 'listen', ' ', '[::]:{0}'.format(https_port), ' ', 'ssl']
            if not ipv6info[1]:
                ipv6_block.append(' ')
                ipv6_block.append('ipv6only=on')
        if vhost.ipv4_enabled():
            ipv4_block = ['\n    ', 'listen', ' ', '{0}'.format(https_port), ' ', 'ssl']
        (snakeoil_cert, snakeoil_key) = self._get_snakeoil_paths()
        ssl_block = [ipv6_block, ipv4_block, ['\n    ', 'ssl_certificate', ' ', snakeoil_cert], ['\n    ', 'ssl_certificate_key', ' ', snakeoil_key], ['\n    ', 'include', ' ', self.mod_ssl_conf], ['\n    ', 'ssl_dhparam', ' ', self.ssl_dhparams]]
        self.parser.add_server_directives(vhost, ssl_block)

    def supported_enhancements(self) -> List[str]:
        if False:
            return 10
        'Returns currently supported enhancements.'
        return ['redirect', 'ensure-http-header', 'staple-ocsp']

    def enhance(self, domain: str, enhancement: str, options: Optional[Union[str, List[str]]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Enhance configuration.\n\n        :param str domain: domain to enhance\n        :param str enhancement: enhancement type defined in\n            :const:`~certbot.plugins.enhancements.ENHANCEMENTS`\n        :param options: options for the enhancement\n            See :const:`~certbot.plugins.enhancements.ENHANCEMENTS`\n            documentation for appropriate parameter.\n\n        '
        try:
            self._enhance_func[enhancement](domain, options)
        except (KeyError, ValueError):
            raise errors.PluginError('Unsupported enhancement: {0}'.format(enhancement))

    def _has_certbot_redirect(self, vhost: obj.VirtualHost, domain: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        test_redirect_block = _test_block_from_block(_redirect_block_for_domain(domain))
        return vhost.contains_list(test_redirect_block)

    def _set_http_header(self, domain: str, header_substring: Union[str, List[str], None]) -> None:
        if False:
            return 10
        'Enables header identified by header_substring on domain.\n\n        If the vhost is listening plaintextishly, separates out the relevant\n        directives into a new server block, and only add header directive to\n        HTTPS block.\n\n        :param str domain: the domain to enable header for.\n        :param str header_substring: String to uniquely identify a header.\n                        e.g. Strict-Transport-Security, Upgrade-Insecure-Requests\n        :returns: Success\n        :raises .errors.PluginError: If no viable HTTPS host can be created or\n            set with header header_substring.\n        '
        if not isinstance(header_substring, str):
            raise errors.NotSupportedError(f'Invalid header_substring type {type(header_substring)}, expected a str.')
        if header_substring not in constants.HEADER_ARGS:
            raise errors.NotSupportedError(f'{header_substring} is not supported by the nginx plugin.')
        vhosts = self.choose_vhosts(domain)
        if not vhosts:
            raise errors.PluginError('Unable to find corresponding HTTPS host for enhancement.')
        for vhost in vhosts:
            if vhost.has_header(header_substring):
                raise errors.PluginEnhancementAlreadyPresent('Existing %s header' % header_substring)
            if vhost.ssl and any((not addr.ssl for addr in vhost.addrs)):
                (_, vhost) = self._split_block(vhost)
            header_directives = [['\n    ', 'add_header', ' ', header_substring, ' '] + constants.HEADER_ARGS[header_substring], ['\n']]
            self.parser.add_server_directives(vhost, header_directives)

    def _add_redirect_block(self, vhost: obj.VirtualHost, domain: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add redirect directive to vhost\n        '
        redirect_block = _redirect_block_for_domain(domain)
        self.parser.add_server_directives(vhost, redirect_block, insert_at_top=True)

    def _split_block(self, vhost: obj.VirtualHost, only_directives: Optional[List[str]]=None) -> Tuple[obj.VirtualHost, obj.VirtualHost]:
        if False:
            i = 10
            return i + 15
        'Splits this "virtual host" (i.e. this nginx server block) into\n        separate HTTP and HTTPS blocks.\n\n        :param vhost: The server block to break up into two.\n        :param list only_directives: If this exists, only duplicate these directives\n            when splitting the block.\n        :type vhost: :class:`~certbot_nginx._internal.obj.VirtualHost`\n        :returns: tuple (http_vhost, https_vhost)\n        :rtype: tuple of type :class:`~certbot_nginx._internal.obj.VirtualHost`\n        '
        http_vhost = self.parser.duplicate_vhost(vhost, only_directives=only_directives)

        def _ssl_match_func(directive: str) -> bool:
            if False:
                while True:
                    i = 10
            return 'ssl' in directive

        def _ssl_config_match_func(directive: str) -> bool:
            if False:
                i = 10
                return i + 15
            return self.mod_ssl_conf in directive

        def _no_ssl_match_func(directive: str) -> bool:
            if False:
                while True:
                    i = 10
            return 'ssl' not in directive
        for directive in self.SSL_DIRECTIVES:
            self.parser.remove_server_directives(http_vhost, directive)
        self.parser.remove_server_directives(http_vhost, 'listen', match_func=_ssl_match_func)
        self.parser.remove_server_directives(http_vhost, 'include', match_func=_ssl_config_match_func)
        self.parser.remove_server_directives(vhost, 'listen', match_func=_no_ssl_match_func)
        return (http_vhost, vhost)

    def _enable_redirect(self, domain: str, unused_options: Optional[Union[str, List[str]]]) -> None:
        if False:
            return 10
        'Redirect all equivalent HTTP traffic to ssl_vhost.\n\n        If the vhost is listening plaintextishly, separate out the\n        relevant directives into a new server block and add a rewrite directive.\n\n        .. note:: This function saves the configuration\n\n        :param str domain: domain to enable redirect for\n        :param unused_options: Not currently used\n        :type unused_options: Not Available\n        '
        port = self.DEFAULT_LISTEN_PORT
        vhosts = self.choose_redirect_vhosts(domain, port)
        if not vhosts:
            logger.info('No matching insecure server blocks listening on port %s found.', self.DEFAULT_LISTEN_PORT)
            return
        for vhost in vhosts:
            self._enable_redirect_single(domain, vhost)

    def _enable_redirect_single(self, domain: str, vhost: obj.VirtualHost) -> None:
        if False:
            i = 10
            return i + 15
        'Redirect all equivalent HTTP traffic to ssl_vhost.\n\n        If the vhost is listening plaintextishly, separate out the\n        relevant directives into a new server block and add a rewrite directive.\n\n        .. note:: This function saves the configuration\n\n        :param str domain: domain to enable redirect for\n        :param `~obj.Vhost` vhost: vhost to enable redirect for\n        '
        if vhost.ssl:
            (http_vhost, _) = self._split_block(vhost, ['listen', 'server_name'])
            return_404_directive = [['\n    ', 'return', ' ', '404']]
            self.parser.add_server_directives(http_vhost, return_404_directive)
            vhost = http_vhost
        if self._has_certbot_redirect(vhost, domain):
            logger.info('Traffic on port %s already redirecting to ssl in %s', self.DEFAULT_LISTEN_PORT, vhost.filep)
        else:
            self._add_redirect_block(vhost, domain)
            logger.info('Redirecting all traffic on port %s to ssl in %s', self.DEFAULT_LISTEN_PORT, vhost.filep)

    def _enable_ocsp_stapling(self, domain: str, chain_path: Optional[Union[str, List[str]]]) -> None:
        if False:
            while True:
                i = 10
        'Include OCSP response in TLS handshake\n\n        :param str domain: domain to enable OCSP response for\n        :param chain_path: chain file path\n        :type chain_path: `str` or `None`\n\n        '
        if not isinstance(chain_path, str) and chain_path is not None:
            raise errors.NotSupportedError(f'Invalid chain_path type {type(chain_path)}, expected a str or None.')
        vhosts = self.choose_vhosts(domain)
        for vhost in vhosts:
            self._enable_ocsp_stapling_single(vhost, chain_path)

    def _enable_ocsp_stapling_single(self, vhost: obj.VirtualHost, chain_path: Optional[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Include OCSP response in TLS handshake\n\n        :param str vhost: vhost to enable OCSP response for\n        :param chain_path: chain file path\n        :type chain_path: `str` or `None`\n\n        '
        if self.version < (1, 3, 7):
            raise errors.PluginError('Version 1.3.7 or greater of nginx is needed to enable OCSP stapling')
        if chain_path is None:
            raise errors.PluginError('--chain-path is required to enable Online Certificate Status Protocol (OCSP) stapling on nginx >= 1.3.7.')
        stapling_directives = [['\n    ', 'ssl_trusted_certificate', ' ', chain_path], ['\n    ', 'ssl_stapling', ' ', 'on'], ['\n    ', 'ssl_stapling_verify', ' ', 'on'], ['\n']]
        try:
            self.parser.add_server_directives(vhost, stapling_directives)
        except errors.MisconfigurationError as error:
            logger.debug(str(error))
            raise errors.PluginError('An error occurred while enabling OCSP stapling for {0}.'.format(vhost.names))
        self.save_notes += 'OCSP Stapling was enabled on SSL Vhost: {0}.\n'.format(vhost.filep)
        self.save_notes += '\tssl_trusted_certificate {0}\n'.format(chain_path)
        self.save_notes += '\tssl_stapling on\n'
        self.save_notes += '\tssl_stapling_verify on\n'

    def restart(self) -> None:
        if False:
            while True:
                i = 10
        'Restarts nginx server.\n\n        :raises .errors.MisconfigurationError: If either the reload fails.\n\n        '
        nginx_restart(self.conf('ctl'), self.nginx_conf, self.conf('sleep-seconds'))

    def config_test(self) -> None:
        if False:
            while True:
                i = 10
        'Check the configuration of Nginx for errors.\n\n        :raises .errors.MisconfigurationError: If config_test fails\n\n        '
        try:
            util.run_script([self.conf('ctl'), '-c', self.nginx_conf, '-t'])
        except errors.SubprocessError as err:
            raise errors.MisconfigurationError(str(err))

    def _nginx_version(self) -> str:
        if False:
            while True:
                i = 10
        'Return results of nginx -V\n\n        :returns: version text\n        :rtype: str\n\n        :raises .PluginError:\n            Unable to run Nginx version command\n        '
        try:
            proc = subprocess.run([self.conf('ctl'), '-c', self.nginx_conf, '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False, env=util.env_no_snap_for_external_calls())
            text = proc.stderr
        except (OSError, ValueError) as error:
            logger.debug(str(error), exc_info=True)
            raise errors.PluginError('Unable to run %s -V' % self.conf('ctl'))
        return text

    def get_version(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        'Return version of Nginx Server.\n\n        Version is returned as tuple. (ie. 2.4.7 = (2, 4, 7))\n\n        :returns: version\n        :rtype: tuple\n\n        :raises .PluginError:\n            Unable to find Nginx version or version is unsupported\n\n        '
        text = self._nginx_version()
        version_regex = re.compile('nginx version: ([^/]+)/([0-9\\.]*)', re.IGNORECASE)
        version_matches = version_regex.findall(text)
        sni_regex = re.compile('TLS SNI support enabled', re.IGNORECASE)
        sni_matches = sni_regex.findall(text)
        ssl_regex = re.compile(' --with-http_ssl_module')
        ssl_matches = ssl_regex.findall(text)
        if not version_matches:
            raise errors.PluginError('Unable to find Nginx version')
        if not ssl_matches:
            raise errors.PluginError('Nginx build is missing SSL module (--with-http_ssl_module).')
        if not sni_matches:
            raise errors.PluginError("Nginx build doesn't support SNI")
        (product_name, product_version) = version_matches[0]
        if product_name != 'nginx':
            logger.warning('NGINX derivative %s is not officially supported by certbot', product_name)
        nginx_version = tuple((int(i) for i in product_version.split('.')))
        if nginx_version < (0, 8, 48):
            raise errors.NotSupportedError('Nginx version must be 0.8.48+')
        return nginx_version

    def _get_openssl_version(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return version of OpenSSL linked to Nginx.\n\n        Version is returned as string. If no version can be found, empty string is returned.\n\n        :returns: openssl_version\n        :rtype: str\n\n        :raises .PluginError:\n            Unable to run Nginx version command\n        '
        text = self._nginx_version()
        matches = re.findall('running with OpenSSL ([^ ]+) ', text)
        if not matches:
            matches = re.findall('built with OpenSSL ([^ ]+) ', text)
            if not matches:
                logger.warning('NGINX configured with OpenSSL alternatives is not officially supported by Certbot.')
                return ''
        return matches[0]

    def more_info(self) -> str:
        if False:
            while True:
                i = 10
        'Human-readable string to help understand the module'
        return 'Configures Nginx to authenticate and install HTTPS.{0}Server root: {root}{0}Version: {version}'.format(os.linesep, root=self.parser.config_root, version='.'.join((str(i) for i in self.version)))

    def auth_hint(self, failed_achalls: Iterable[achallenges.AnnotatedChallenge]) -> str:
        if False:
            while True:
                i = 10
        return 'The Certificate Authority failed to verify the temporary nginx configuration changes made by Certbot. Ensure the listed domains point to this nginx server and that it is accessible from the internet.'

    def save(self, title: Optional[str]=None, temporary: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Saves all changes to the configuration files.\n\n        :param str title: The title of the save. If a title is given, the\n            configuration will be saved as a new checkpoint and put in a\n            timestamped directory.\n\n        :param bool temporary: Indicates whether the changes made will\n            be quickly reversed in the future (ie. challenges)\n\n        :raises .errors.PluginError: If there was an error in\n            an attempt to save the configuration, or an error creating a\n            checkpoint\n\n        '
        save_files = set(self.parser.parsed.keys())
        self.add_to_checkpoint(save_files, self.save_notes, temporary)
        self.save_notes = ''
        self.parser.filedump(ext='')
        if title and (not temporary):
            self.finalize_checkpoint(title)

    def recovery_routine(self) -> None:
        if False:
            while True:
                i = 10
        'Revert all previously modified files.\n\n        Reverts all modified files that have not been saved as a checkpoint\n\n        :raises .errors.PluginError: If unable to recover the configuration\n\n        '
        super().recovery_routine()
        self.new_vhost = None
        self.parser.load()

    def revert_challenge_config(self) -> None:
        if False:
            while True:
                i = 10
        'Used to cleanup challenge configurations.\n\n        :raises .errors.PluginError: If unable to revert the challenge config.\n\n        '
        self.revert_temporary_config()
        self.new_vhost = None
        self.parser.load()

    def rollback_checkpoints(self, rollback: int=1) -> None:
        if False:
            i = 10
            return i + 15
        'Rollback saved checkpoints.\n\n        :param int rollback: Number of checkpoints to revert\n\n        :raises .errors.PluginError: If there is a problem with the input or\n            the function is unable to correctly revert the configuration\n\n        '
        super().rollback_checkpoints(rollback)
        self.new_vhost = None
        self.parser.load()

    def get_chall_pref(self, unused_domain: str) -> List[Type[challenges.Challenge]]:
        if False:
            print('Hello World!')
        'Return list of challenge preferences.'
        return [challenges.HTTP01]

    def perform(self, achalls: List[achallenges.AnnotatedChallenge]) -> List[challenges.ChallengeResponse]:
        if False:
            for i in range(10):
                print('nop')
        'Perform the configuration related challenge.\n\n        This function currently assumes all challenges will be fulfilled.\n        If this turns out not to be the case in the future. Cleanup and\n        outstanding challenges will have to be designed better.\n\n        '
        self._chall_out += len(achalls)
        responses: List[Optional[challenges.ChallengeResponse]] = [None] * len(achalls)
        http_doer = http_01.NginxHttp01(self)
        for (i, achall) in enumerate(achalls):
            if not isinstance(achall, achallenges.KeyAuthorizationAnnotatedChallenge):
                raise errors.Error('Challenge should be an instance of KeyAuthorizationAnnotatedChallenge')
            http_doer.add_chall(achall, i)
        http_response = http_doer.perform()
        self.restart()
        for (i, resp) in enumerate(http_response):
            responses[http_doer.indices[i]] = resp
        return [response for response in responses if response]

    def cleanup(self, achalls: List[achallenges.AnnotatedChallenge]) -> None:
        if False:
            print('Hello World!')
        'Revert all challenges.'
        self._chall_out -= len(achalls)
        if self._chall_out <= 0:
            self.revert_challenge_config()
            self.restart()

def _test_block_from_block(block: List[Any]) -> List[Any]:
    if False:
        i = 10
        return i + 15
    test_block = nginxparser.UnspacedList(block)
    parser.comment_directive(test_block, 0)
    return test_block[:-1]

def _redirect_block_for_domain(domain: str) -> List[Any]:
    if False:
        print('Hello World!')
    updated_domain = domain
    match_symbol = '='
    if util.is_wildcard_domain(domain):
        match_symbol = '~'
        updated_domain = updated_domain.replace('.', '\\.')
        updated_domain = updated_domain.replace('*', '[^.]+')
        updated_domain = '^' + updated_domain + '$'
    redirect_block = [[['\n    ', 'if', ' ', '($host', ' ', match_symbol, ' ', '%s)' % updated_domain, ' '], [['\n        ', 'return', ' ', '301', ' ', 'https://$host$request_uri'], '\n    ']], ['\n']]
    return redirect_block

def nginx_restart(nginx_ctl: str, nginx_conf: str, sleep_duration: int) -> None:
    if False:
        print('Hello World!')
    'Restarts the Nginx Server.\n\n    .. todo:: Nginx restart is fatal if the configuration references\n        non-existent SSL cert/key files. Remove references to /etc/letsencrypt\n        before restart.\n\n    :param str nginx_ctl: Path to the Nginx binary.\n    :param str nginx_conf: Path to the Nginx configuration file.\n    :param int sleep_duration: How long to sleep after sending the reload signal.\n\n    '
    try:
        reload_output: str = ''
        with tempfile.TemporaryFile() as out:
            proc = subprocess.run([nginx_ctl, '-c', nginx_conf, '-s', 'reload'], env=util.env_no_snap_for_external_calls(), stdout=out, stderr=out, check=False)
            out.seek(0)
            reload_output = out.read().decode('utf-8')
        if proc.returncode != 0:
            logger.debug('nginx reload failed:\n%s', reload_output)
            with tempfile.TemporaryFile() as out:
                nginx_proc = subprocess.run([nginx_ctl, '-c', nginx_conf], stdout=out, stderr=out, env=util.env_no_snap_for_external_calls(), check=False)
                if nginx_proc.returncode != 0:
                    out.seek(0)
                    raise errors.MisconfigurationError('nginx restart failed:\n%s' % out.read().decode('utf-8'))
    except (OSError, ValueError):
        raise errors.MisconfigurationError('nginx restart failed')
    if sleep_duration > 0:
        time.sleep(sleep_duration)

def _determine_default_server_root() -> str:
    if False:
        while True:
            i = 10
    if os.environ.get('CERTBOT_DOCS') == '1':
        default_server_root = f'{constants.LINUX_SERVER_ROOT} or {constants.FREEBSD_DARWIN_SERVER_ROOT}'
    else:
        default_server_root = constants.CLI_DEFAULTS['server_root']
    return default_server_root