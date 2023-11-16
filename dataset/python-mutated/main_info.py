"""CLI implementation for `conda info`.

Display information about current conda installation.
"""
from __future__ import annotations
import json
import os
import re
import sys
from argparse import SUPPRESS, ArgumentParser, Namespace, _StoreTrueAction, _SubParsersAction
from logging import getLogger
from os.path import exists, expanduser, isfile, join
from textwrap import wrap
from typing import Iterable
from ..deprecations import deprecated
log = getLogger(__name__)

def configure_parser(sub_parsers: _SubParsersAction, **kwargs) -> ArgumentParser:
    if False:
        for i in range(10):
            print('nop')
    from ..common.constants import NULL
    from .helpers import add_parser_json
    summary = 'Display information about current conda install.'
    description = summary
    epilog = ''
    p = sub_parsers.add_parser('info', help=summary, description=description, epilog=epilog, **kwargs)
    add_parser_json(p)
    p.add_argument('--offline', action='store_true', default=NULL, help=SUPPRESS)
    p.add_argument('-a', '--all', dest='verbosity', action=deprecated.action('24.3', '24.9', _StoreTrueAction, addendum='Use `--verbose` instead.'))
    p.add_argument('--base', action='store_true', help='Display base environment path.')
    p.add_argument('-e', '--envs', action='store_true', help='List all known conda environments.')
    p.add_argument('-l', '--license', action='store_true', help=SUPPRESS)
    p.add_argument('-s', '--system', action='store_true', help='List environment variables.')
    p.add_argument('--root', action='store_true', help=SUPPRESS, dest='base')
    p.add_argument('--unsafe-channels', action='store_true', help='Display list of channels with tokens exposed.')
    p.add_argument('packages', action='store', nargs='*', help=SUPPRESS)
    p.set_defaults(func='conda.cli.main_info.execute')
    return p

def get_user_site():
    if False:
        print('Hello World!')
    from ..common.compat import on_win
    site_dirs = []
    try:
        if not on_win:
            if exists(expanduser('~/.local/lib')):
                python_re = re.compile('python\\d\\.\\d')
                for path in os.listdir(expanduser('~/.local/lib/')):
                    if python_re.match(path):
                        site_dirs.append('~/.local/lib/%s' % path)
        else:
            if 'APPDATA' not in os.environ:
                return site_dirs
            APPDATA = os.environ['APPDATA']
            if exists(join(APPDATA, 'Python')):
                site_dirs = [join(APPDATA, 'Python', i) for i in os.listdir(join(APPDATA, 'PYTHON'))]
    except OSError as e:
        log.debug('Error accessing user site directory.\n%r', e)
    return site_dirs
IGNORE_FIELDS = {'files', 'auth', 'preferred_env', 'priority'}
SKIP_FIELDS = IGNORE_FIELDS | {'name', 'version', 'build', 'build_number', 'channel', 'schannel', 'size', 'fn', 'depends'}

def dump_record(pkg):
    if False:
        i = 10
        return i + 15
    return {k: v for (k, v) in pkg.dump().items() if k not in IGNORE_FIELDS}

def pretty_package(prec):
    if False:
        print('Hello World!')
    from ..utils import human_bytes
    pkg = dump_record(prec)
    d = {'file name': prec.fn, 'name': pkg['name'], 'version': pkg['version'], 'build string': pkg['build'], 'build number': pkg['build_number'], 'channel': str(prec.channel), 'size': human_bytes(pkg['size'])}
    for key in sorted(set(pkg.keys()) - SKIP_FIELDS):
        d[key] = pkg[key]
    print()
    header = '{} {} {}'.format(d['name'], d['version'], d['build string'])
    print(header)
    print('-' * len(header))
    for key in d:
        print('%-12s: %s' % (key, d[key]))
    print('dependencies:')
    for dep in pkg['depends']:
        print('    %s' % dep)

def print_package_info(packages):
    if False:
        print('Hello World!')
    from ..base.context import context
    from ..core.subdir_data import SubdirData
    from ..deprecations import deprecated
    from ..models.match_spec import MatchSpec
    from .common import stdout_json
    results = {}
    for package in packages:
        spec = MatchSpec(package)
        results[package] = tuple(SubdirData.query_all(spec))
    if context.json:
        stdout_json({package: results[package] for package in packages})
    else:
        for result in results.values():
            for prec in result:
                pretty_package(prec)
    deprecated.topic('23.9', '24.3', topic='`conda info package_name`', addendum='Use `conda search package_name --info` instead.')

def get_info_dict(system=False):
    if False:
        return 10
    from .. import CONDA_PACKAGE_ROOT
    from .. import __version__ as conda_version
    from ..base.context import DEFAULT_SOLVER, context, env_name, sys_rc_path, user_rc_path
    from ..common.compat import on_win
    from ..common.url import mask_anaconda_token
    from ..core.index import _supplement_index_with_system
    from ..models.channel import all_channel_urls, offline_keep
    try:
        from conda_build import __version__ as conda_build_version
    except ImportError as err:
        log.debug('Unable to import conda-build: %s', err)
        conda_build_version = 'not installed'
    except Exception as err:
        log.error('Error importing conda-build: %s', err)
        conda_build_version = 'error'
    virtual_pkg_index = {}
    _supplement_index_with_system(virtual_pkg_index)
    virtual_pkgs = [[p.name, p.version, p.build] for p in virtual_pkg_index.values()]
    channels = list(all_channel_urls(context.channels))
    if not context.json:
        channels = [c + ('' if offline_keep(c) else '  (offline)') for c in channels]
    channels = [mask_anaconda_token(c) for c in channels]
    netrc_file = os.environ.get('NETRC')
    if not netrc_file:
        user_netrc = expanduser('~/.netrc')
        if isfile(user_netrc):
            netrc_file = user_netrc
    active_prefix_name = env_name(context.active_prefix)
    solver = {'name': context.solver, 'user_agent': context.solver_user_agent(), 'default': context.solver == DEFAULT_SOLVER}
    info_dict = dict(platform=context.subdir, conda_version=conda_version, conda_env_version=conda_version, conda_build_version=conda_build_version, root_prefix=context.root_prefix, conda_prefix=context.conda_prefix, av_data_dir=context.av_data_dir, av_metadata_url_base=context.signing_metadata_url_base, root_writable=context.root_writable, pkgs_dirs=context.pkgs_dirs, envs_dirs=context.envs_dirs, default_prefix=context.default_prefix, active_prefix=context.active_prefix, active_prefix_name=active_prefix_name, conda_shlvl=context.shlvl, channels=channels, user_rc_path=user_rc_path, rc_path=user_rc_path, sys_rc_path=sys_rc_path, offline=context.offline, envs=[], python_version='.'.join(map(str, sys.version_info)), requests_version=context.requests_version, user_agent=context.user_agent, conda_location=CONDA_PACKAGE_ROOT, config_files=context.config_files, netrc_file=netrc_file, virtual_pkgs=virtual_pkgs, solver=solver)
    if on_win:
        from ..common._os.windows import is_admin_on_windows
        info_dict['is_windows_admin'] = is_admin_on_windows()
    else:
        info_dict['UID'] = os.geteuid()
        info_dict['GID'] = os.getegid()
    env_var_keys = {'CIO_TEST', 'CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'LD_PRELOAD'}
    env_var_keys.update((v for v in os.environ if v.upper().startswith('CONDA')))
    env_var_keys.update((v for v in os.environ if v.upper().startswith('PYTHON')))
    env_var_keys.update((v for v in os.environ if v.upper().endswith('PATH')))
    env_var_keys.update((v for v in os.environ if v.upper().startswith('SUDO')))
    env_vars = {ev: os.getenv(ev, os.getenv(ev.lower(), '<not set>')) for ev in env_var_keys}
    proxy_keys = (v for v in os.environ if v.upper().endswith('PROXY'))
    env_vars.update({ev: '<set>' for ev in proxy_keys})
    info_dict.update({'sys.version': sys.version, 'sys.prefix': sys.prefix, 'sys.executable': sys.executable, 'site_dirs': get_user_site(), 'env_vars': env_vars})
    return info_dict

def get_env_vars_str(info_dict):
    if False:
        return 10
    builder = []
    builder.append('%23s:' % 'environment variables')
    env_vars = info_dict.get('env_vars', {})
    for key in sorted(env_vars):
        value = wrap(env_vars[key])
        first_line = value[0] if len(value) else ''
        other_lines = value[1:] if len(value) > 1 else ()
        builder.append('%25s=%s' % (key, first_line))
        for val in other_lines:
            builder.append(' ' * 26 + val)
    return '\n'.join(builder)

def get_main_info_str(info_dict):
    if False:
        while True:
            i = 10
    from ..common.compat import on_win

    def flatten(lines: Iterable[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ('\n' + 26 * ' ').join(map(str, lines))

    def builder():
        if False:
            for i in range(10):
                print('nop')
        if info_dict['active_prefix_name']:
            yield ('active environment', info_dict['active_prefix_name'])
            yield ('active env location', info_dict['active_prefix'])
        else:
            yield ('active environment', info_dict['active_prefix'])
        if info_dict['conda_shlvl'] >= 0:
            yield ('shell level', info_dict['conda_shlvl'])
        yield ('user config file', info_dict['user_rc_path'])
        yield ('populated config files', flatten(info_dict['config_files']))
        yield ('conda version', info_dict['conda_version'])
        yield ('conda-build version', info_dict['conda_build_version'])
        yield ('python version', info_dict['python_version'])
        yield ('solver', f"{info_dict['solver']['name']}{(' (default)' if info_dict['solver']['default'] else '')}")
        yield ('virtual packages', flatten(('='.join(pkg) for pkg in info_dict['virtual_pkgs'])))
        writable = 'writable' if info_dict['root_writable'] else 'read only'
        yield ('base environment', f"{info_dict['root_prefix']}  ({writable})")
        yield ('conda av data dir', info_dict['av_data_dir'])
        yield ('conda av metadata url', info_dict['av_metadata_url_base'])
        yield ('channel URLs', flatten(info_dict['channels']))
        yield ('package cache', flatten(info_dict['pkgs_dirs']))
        yield ('envs directories', flatten(info_dict['envs_dirs']))
        yield ('platform', info_dict['platform'])
        yield ('user-agent', info_dict['user_agent'])
        if on_win:
            yield ('administrator', info_dict['is_windows_admin'])
        else:
            yield ('UID:GID', f"{info_dict['UID']}:{info_dict['GID']}")
        yield ('netrc file', info_dict['netrc_file'])
        yield ('offline mode', info_dict['offline'])
    return '\n'.join(('', *(f'{key:>23} : {value}' for (key, value) in builder()), ''))

def execute(args: Namespace, parser: ArgumentParser) -> int:
    if False:
        for i in range(10):
            print('nop')
    from ..base.context import context
    from .common import print_envs_list, stdout_json
    if args.base:
        if context.json:
            stdout_json({'root_prefix': context.root_prefix})
        else:
            print(f'{context.root_prefix}')
        return 0
    if args.packages:
        from ..resolve import ResolvePackageNotFound
        try:
            print_package_info(args.packages)
            return 0
        except ResolvePackageNotFound as e:
            from ..exceptions import PackagesNotFoundError
            raise PackagesNotFoundError(e.bad_deps)
    if args.unsafe_channels:
        if not context.json:
            print('\n'.join(context.channels))
        else:
            print(json.dumps({'channels': context.channels}))
        return 0
    options = ('envs', 'system')
    if context.verbose or context.json:
        for option in options:
            setattr(args, option, True)
    info_dict = get_info_dict(args.system)
    if (context.verbose or all((not getattr(args, opt) for opt in options))) and (not context.json):
        print(get_main_info_str(info_dict) + '\n')
    if args.envs:
        from ..core.envs_manager import list_all_known_prefixes
        info_dict['envs'] = list_all_known_prefixes()
        print_envs_list(info_dict['envs'], not context.json)
    if args.system:
        if not context.json:
            from .find_commands import find_commands, find_executable
            print('sys.version: %s...' % sys.version[:40])
            print('sys.prefix: %s' % sys.prefix)
            print('sys.executable: %s' % sys.executable)
            print('conda location: %s' % info_dict['conda_location'])
            for cmd in sorted(set(find_commands() + ('build',))):
                print('conda-{}: {}'.format(cmd, find_executable('conda-' + cmd)))
            print('user site dirs: ', end='')
            site_dirs = info_dict['site_dirs']
            if site_dirs:
                print(site_dirs[0])
            else:
                print()
            for site_dir in site_dirs[1:]:
                print('                %s' % site_dir)
            print()
            for (name, value) in sorted(info_dict['env_vars'].items()):
                print(f'{name}: {value}')
            print()
    if context.json:
        stdout_json(info_dict)
    return 0