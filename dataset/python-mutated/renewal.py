"""Functionality for autorenewal and associated juggling of configurations"""
import copy
import itertools
import logging
import random
import sys
import time
import traceback
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from certbot import configuration
from certbot import crypto_util
from certbot import errors
from certbot import util
from certbot._internal import cli
from certbot._internal import client
from certbot._internal import constants
from certbot._internal import hooks
from certbot._internal import storage
from certbot._internal import updater
from certbot._internal.display import obj as display_obj
from certbot._internal.plugins import disco as plugins_disco
from certbot.compat import os
from certbot.display import util as display_util
logger = logging.getLogger(__name__)
STR_CONFIG_ITEMS = ['config_dir', 'logs_dir', 'work_dir', 'user_agent', 'server', 'account', 'authenticator', 'installer', 'renew_hook', 'pre_hook', 'post_hook', 'http01_address', 'preferred_chain', 'key_type', 'elliptic_curve']
INT_CONFIG_ITEMS = ['rsa_key_size', 'http01_port']
BOOL_CONFIG_ITEMS = ['must_staple', 'allow_subset_of_names', 'reuse_key', 'autorenew']
CONFIG_ITEMS = set(itertools.chain(BOOL_CONFIG_ITEMS, INT_CONFIG_ITEMS, STR_CONFIG_ITEMS, ('pref_challs',)))

def reconstitute(config: configuration.NamespaceConfig, full_path: str) -> Optional[storage.RenewableCert]:
    if False:
        for i in range(10):
            print('nop')
    'Try to instantiate a RenewableCert, updating config with relevant items.\n\n    This is specifically for use in renewal and enforces several checks\n    and policies to ensure that we can try to proceed with the renewal\n    request. The config argument is modified by including relevant options\n    read from the renewal configuration file.\n\n    :param configuration.NamespaceConfig config: configuration for the\n        current lineage\n    :param str full_path: Absolute path to the configuration file that\n        defines this lineage\n\n    :returns: the RenewableCert object or None if a fatal error occurred\n    :rtype: `storage.RenewableCert` or NoneType\n\n    '
    try:
        renewal_candidate = storage.RenewableCert(full_path, config)
    except (errors.CertStorageError, IOError) as error:
        logger.error('Renewal configuration file %s is broken.', full_path)
        logger.error('The error was: %s\nSkipping.', str(error))
        logger.debug('Traceback was:\n%s', traceback.format_exc())
        return None
    if 'renewalparams' not in renewal_candidate.configuration:
        logger.error('Renewal configuration file %s lacks renewalparams. Skipping.', full_path)
        return None
    renewalparams = renewal_candidate.configuration['renewalparams']
    if 'authenticator' not in renewalparams:
        logger.error('Renewal configuration file %s does not specify an authenticator. Skipping.', full_path)
        return None
    renewalparams['key_type'] = renewalparams.get('key_type', 'rsa')
    renewalparams = _remove_deprecated_config_elements(renewalparams)
    try:
        restore_required_config_elements(config, renewalparams)
        _restore_plugin_configs(config, renewalparams)
    except (ValueError, errors.Error) as error:
        logger.error('An error occurred while parsing %s. The error was %s. Skipping the file.', full_path, str(error))
        logger.debug('Traceback was:\n%s', traceback.format_exc())
        return None
    try:
        config.domains = [util.enforce_domain_sanity(d) for d in renewal_candidate.names()]
    except errors.ConfigurationError as error:
        logger.error('Renewal configuration file %s references a certificate that contains an invalid domain name. The problem was: %s. Skipping.', full_path, error)
        return None
    return renewal_candidate

def _restore_webroot_config(config: configuration.NamespaceConfig, renewalparams: Mapping[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    webroot_map is, uniquely, a dict, and the general-purpose configuration\n    restoring logic is not able to correctly parse it from the serialized\n    form.\n    '
    if 'webroot_map' in renewalparams and (not config.set_by_user('webroot_map')):
        config.webroot_map = renewalparams['webroot_map']
    if 'webroot_path' in renewalparams and (not config.set_by_user('webroot_path')):
        wp = renewalparams['webroot_path']
        if isinstance(wp, str):
            wp = [wp]
        config.webroot_path = wp

def _restore_plugin_configs(config: configuration.NamespaceConfig, renewalparams: Mapping[str, Any]) -> None:
    if False:
        while True:
            i = 10
    'Sets plugin specific values in config from renewalparams\n\n    :param configuration.NamespaceConfig config: configuration for the\n        current lineage\n    :param configobj.Section renewalparams: Parameters from the renewal\n        configuration file that defines this lineage\n\n    '
    plugin_prefixes: List[str] = []
    if renewalparams['authenticator'] == 'webroot':
        _restore_webroot_config(config, renewalparams)
    else:
        plugin_prefixes.append(renewalparams['authenticator'])
    if renewalparams.get('installer') is not None:
        plugin_prefixes.append(renewalparams['installer'])
    for plugin_prefix in set(plugin_prefixes):
        plugin_prefix = plugin_prefix.replace('-', '_')
        for (config_item, config_value) in renewalparams.items():
            if config_item.startswith(plugin_prefix + '_') and (not config.set_by_user(config_item)):
                if config_value in ('None', 'True', 'False'):
                    setattr(config, config_item, eval(config_value))
                else:
                    cast = cli.argparse_type(config_item)
                    setattr(config, config_item, cast(config_value))

def restore_required_config_elements(config: configuration.NamespaceConfig, renewalparams: Mapping[str, Any]) -> None:
    if False:
        print('Hello World!')
    'Sets non-plugin specific values in config from renewalparams\n\n    :param configuration.NamespaceConfig config: configuration for the\n        current lineage\n    :param configobj.Section renewalparams: parameters from the renewal\n        configuration file that defines this lineage\n\n    '
    updated_values = {}
    required_items = itertools.chain((('pref_challs', _restore_pref_challs),), zip(BOOL_CONFIG_ITEMS, itertools.repeat(_restore_bool)), zip(INT_CONFIG_ITEMS, itertools.repeat(_restore_int)), zip(STR_CONFIG_ITEMS, itertools.repeat(_restore_str)))
    for (item_name, restore_func) in required_items:
        if item_name in renewalparams and (not config.set_by_user(item_name)):
            value = restore_func(item_name, renewalparams[item_name])
            updated_values[item_name] = value
    for (key, value) in updated_values.items():
        setattr(config, key, value)

def _remove_deprecated_config_elements(renewalparams: Mapping[str, Any]) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Removes deprecated config options from the parsed renewalparams.\n\n    :param dict renewalparams: list of parsed renewalparams\n\n    :returns: list of renewalparams with deprecated config options removed\n    :rtype: dict\n\n    '
    return {option_name: v for (option_name, v) in renewalparams.items() if option_name not in cli.DEPRECATED_OPTIONS}

def _restore_pref_challs(unused_name: str, value: Union[List[str], str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    "Restores preferred challenges from a renewal config file.\n\n    If value is a `str`, it should be a single challenge type.\n\n    :param str unused_name: option name\n    :param value: option value\n    :type value: `list` of `str` or `str`\n\n    :returns: converted option value to be stored in the runtime config\n    :rtype: `list` of `str`\n\n    :raises errors.Error: if value can't be converted to a bool\n\n    "
    value = [value] if isinstance(value, str) else value
    return cli.parse_preferred_challenges(value)

def _restore_bool(name: str, value: str) -> bool:
    if False:
        return 10
    "Restores a boolean key-value pair from a renewal config file.\n\n    :param str name: option name\n    :param str value: option value\n\n    :returns: converted option value to be stored in the runtime config\n    :rtype: bool\n\n    :raises errors.Error: if value can't be converted to a bool\n\n    "
    lowercase_value = value.lower()
    if lowercase_value not in ('true', 'false'):
        raise errors.Error(f'Expected True or False for {name} but found {value}')
    return lowercase_value == 'true'

def _restore_int(name: str, value: str) -> int:
    if False:
        return 10
    "Restores an integer key-value pair from a renewal config file.\n\n    :param str name: option name\n    :param str value: option value\n\n    :returns: converted option value to be stored in the runtime config\n    :rtype: int\n\n    :raises errors.Error: if value can't be converted to an int\n\n    "
    if name == 'http01_port' and value == 'None':
        logger.info('updating legacy http01_port value')
        return cli.flag_default('http01_port')
    try:
        return int(value)
    except ValueError:
        raise errors.Error(f'Expected a numeric value for {name}')

def _restore_str(name: str, value: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Restores a string key-value pair from a renewal config file.\n\n    :param str name: option name\n    :param str value: option value\n\n    :returns: converted option value to be stored in the runtime config\n    :rtype: str or None\n\n    '
    if name == 'server' and value == constants.V1_URI:
        logger.info('Using server %s instead of legacy %s', constants.CLI_DEFAULTS['server'], value)
        return constants.CLI_DEFAULTS['server']
    return None if value == 'None' else value

def should_renew(config: configuration.NamespaceConfig, lineage: storage.RenewableCert) -> bool:
    if False:
        i = 10
        return i + 15
    'Return true if any of the circumstances for automatic renewal apply.'
    if config.renew_by_default:
        logger.debug('Auto-renewal forced with --force-renewal...')
        return True
    if lineage.should_autorenew():
        logger.info('Certificate is due for renewal, auto-renewing...')
        return True
    if config.dry_run:
        logger.info('Certificate not due for renewal, but simulating renewal for dry run')
        return True
    display_util.notify('Certificate not yet due for renewal')
    return False

def _avoid_invalidating_lineage(config: configuration.NamespaceConfig, lineage: storage.RenewableCert, original_server: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Do not renew a valid cert with one from a staging server!'
    if util.is_staging(config.server):
        if not util.is_staging(original_server):
            if not config.break_my_certs:
                names = ', '.join(lineage.names())
                raise errors.Error(f"You've asked to renew/replace a seemingly valid certificate with a test certificate (domains: {names}). We will not do that unless you use the --break-my-certs flag!")

def _avoid_reuse_key_conflicts(config: configuration.NamespaceConfig, lineage: storage.RenewableCert) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Don't allow combining --reuse-key with any flags that would conflict\n    with key reuse (--key-type, --rsa-key-size, --elliptic-curve), unless\n    --new-key is also set.\n    "
    if config.set_by_user('reuse_key') and (not config.reuse_key):
        return
    if not lineage.reuse_key and (not config.reuse_key):
        return
    if config.new_key:
        return
    kt = config.key_type.lower()
    potential_conflicts = [('--key-type', lambda : kt != lineage.private_key_type.lower()), ('--rsa-key-size', lambda : kt == 'rsa' and config.rsa_key_size != lineage.rsa_key_size), ('--elliptic-curve', lambda : kt == 'ecdsa' and lineage.elliptic_curve and (config.elliptic_curve.lower() != lineage.elliptic_curve.lower()))]
    for conflict in potential_conflicts:
        if conflict[1]():
            raise errors.Error(f'Unable to change the {conflict[0]} of this certificate because --reuse-key is set. To stop reusing the private key, specify --no-reuse-key. To change the private key this one time and then reuse it in future, add --new-key.')

def renew_cert(config: configuration.NamespaceConfig, domains: Optional[List[str]], le_client: client.Client, lineage: storage.RenewableCert) -> None:
    if False:
        while True:
            i = 10
    'Renew a certificate lineage.'
    renewal_params = lineage.configuration['renewalparams']
    original_server = renewal_params.get('server', cli.flag_default('server'))
    _avoid_invalidating_lineage(config, lineage, original_server)
    _avoid_reuse_key_conflicts(config, lineage)
    if not domains:
        domains = lineage.names()
    if config.reuse_key and (not config.new_key):
        new_key = os.path.normpath(lineage.privkey)
        _update_renewal_params_from_key(new_key, config)
    else:
        new_key = None
    (new_cert, new_chain, new_key, _) = le_client.obtain_certificate(domains, new_key)
    if config.dry_run:
        logger.debug('Dry run: skipping updating lineage at %s', os.path.dirname(lineage.cert))
    else:
        prior_version = lineage.latest_common_version()
        lineage.save_successor(prior_version, new_cert, new_key.pem, new_chain, config)
        lineage.update_all_links_to(lineage.latest_common_version())
        lineage.truncate()
    hooks.renew_hook(config, domains, lineage.live_dir)

def report(msgs: Iterable[str], category: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Format a results report for a category of renewal outcomes'
    lines = ('%s (%s)' % (m, category) for m in msgs)
    return '  ' + '\n  '.join(lines)

def _renew_describe_results(config: configuration.NamespaceConfig, renew_successes: List[str], renew_failures: List[str], renew_skipped: List[str], parse_failures: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Print a report to the terminal about the results of the renewal process.\n\n    :param configuration.NamespaceConfiguration config: Configuration\n    :param list renew_successes: list of fullchain paths which were renewed\n    :param list renew_failures: list of fullchain paths which failed to be renewed\n    :param list renew_skipped: list of messages to print about skipped certificates\n    :param list parse_failures: list of renewal parameter paths which had errors\n    '
    notify = display_util.notify
    notify_error = logger.error
    notify(f'\n{display_obj.SIDE_FRAME}')
    renewal_noun = 'simulated renewal' if config.dry_run else 'renewal'
    if renew_skipped:
        notify('The following certificates are not due for renewal yet:')
        notify(report(renew_skipped, 'skipped'))
    if not renew_successes and (not renew_failures):
        notify(f'No {renewal_noun}s were attempted.')
        if config.pre_hook is not None or config.renew_hook is not None or config.post_hook is not None:
            notify('No hooks were run.')
    elif renew_successes and (not renew_failures):
        notify(f'Congratulations, all {renewal_noun}s succeeded: ')
        notify(report(renew_successes, 'success'))
    elif renew_failures and (not renew_successes):
        notify_error('All %ss failed. The following certificates could not be renewed:', renewal_noun)
        notify_error(report(renew_failures, 'failure'))
    elif renew_failures and renew_successes:
        notify(f'The following {renewal_noun}s succeeded:')
        notify(report(renew_successes, 'success') + '\n')
        notify_error('The following %ss failed:', renewal_noun)
        notify_error(report(renew_failures, 'failure'))
    if parse_failures:
        notify('\nAdditionally, the following renewal configurations were invalid: ')
        notify(report(parse_failures, 'parsefail'))
    notify(display_obj.SIDE_FRAME)

def handle_renewal_request(config: configuration.NamespaceConfig) -> Tuple[list, list]:
    if False:
        for i in range(10):
            print('nop')
    'Examine each lineage; renew if due and report results'
    if any((domain not in config.webroot_map for domain in config.domains)):
        raise errors.Error('Currently, the renew verb is capable of either renewing all installed certificates that are due to be renewed or renewing a single certificate specified by its name. If you would like to renew specific certificates by their domains, use the certonly command instead. The renew verb may provide other options for selecting certificates to renew in the future.')
    if config.certname:
        conf_files = [storage.renewal_file_for_certname(config, config.certname)]
    else:
        conf_files = storage.renewal_conf_files(config)
    renew_successes = []
    renew_failures = []
    renew_skipped = []
    parse_failures = []
    renewed_domains = []
    failed_domains = []
    apply_random_sleep = not sys.stdin.isatty() and config.random_sleep_on_renew
    for renewal_file in conf_files:
        display_util.notification('Processing ' + renewal_file, pause=False)
        lineage_config = copy.deepcopy(config)
        lineagename = storage.lineagename_for_filename(renewal_file)
        try:
            renewal_candidate = reconstitute(lineage_config, renewal_file)
        except Exception as e:
            logger.error('Renewal configuration file %s (cert: %s) produced an unexpected error: %s. Skipping.', renewal_file, lineagename, e)
            logger.debug('Traceback was:\n%s', traceback.format_exc())
            parse_failures.append(renewal_file)
            continue
        try:
            if not renewal_candidate:
                parse_failures.append(renewal_file)
            else:
                renewal_candidate.ensure_deployed()
                from certbot._internal import main
                plugins = plugins_disco.PluginsRegistry.find_all()
                if should_renew(lineage_config, renewal_candidate):
                    if apply_random_sleep:
                        sleep_time = random.uniform(1, 60 * 8)
                        logger.info('Non-interactive renewal: random delay of %s seconds', sleep_time)
                        time.sleep(sleep_time)
                        apply_random_sleep = False
                    main.renew_cert(lineage_config, plugins, renewal_candidate)
                    renew_successes.append(renewal_candidate.fullchain)
                    renewed_domains.extend(renewal_candidate.names())
                else:
                    expiry = crypto_util.notAfter(renewal_candidate.version('cert', renewal_candidate.latest_common_version()))
                    renew_skipped.append('%s expires on %s' % (renewal_candidate.fullchain, expiry.strftime('%Y-%m-%d')))
                updater.run_generic_updaters(lineage_config, renewal_candidate, plugins)
        except Exception as e:
            logger.error('Failed to renew certificate %s with error: %s', lineagename, e)
            logger.debug('Traceback was:\n%s', traceback.format_exc())
            if renewal_candidate:
                renew_failures.append(renewal_candidate.fullchain)
                failed_domains.extend(renewal_candidate.names())
    _renew_describe_results(config, renew_successes, renew_failures, renew_skipped, parse_failures)
    if renew_failures or parse_failures:
        raise errors.Error(f'{len(renew_failures)} renew failure(s), {len(parse_failures)} parse failure(s)')
    logger.debug('no renewal failures')
    return (renewed_domains, failed_domains)

def _update_renewal_params_from_key(key_path: str, config: configuration.NamespaceConfig) -> None:
    if False:
        for i in range(10):
            print('nop')
    with open(key_path, 'rb') as file_h:
        key = load_pem_private_key(file_h.read(), password=None, backend=default_backend())
    if isinstance(key, rsa.RSAPrivateKey):
        config.key_type = 'rsa'
        config.rsa_key_size = key.key_size
    elif isinstance(key, ec.EllipticCurvePrivateKey):
        config.key_type = 'ecdsa'
        config.elliptic_curve = key.curve.name
    else:
        raise errors.Error(f'Key at {key_path} is of an unsupported type: {type(key)}.')