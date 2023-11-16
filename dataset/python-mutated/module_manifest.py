from __future__ import annotations
import base64
import errno
import json
import os
import pkgutil
import random
import re
from importlib import import_module
from ansible.module_utils.compat.version import LooseVersion
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.loader import ps_module_utils_loader
from ansible.utils.collection_loader import resource_from_fqcr

class PSModuleDepFinder(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.ps_modules = dict()
        self.exec_scripts = dict()
        self.cs_utils_wrapper = dict()
        self.cs_utils_module = dict()
        self.ps_version = None
        self.os_version = None
        self.become = False
        self._re_cs_module = [re.compile(to_bytes('(?i)^using\\s((Ansible\\..+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+));\\s*$'))]
        self._re_cs_in_ps_module = [re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-csharputil\\s+((Ansible\\.[\\w\\.]+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+)|(\\.[\\w\\.]+))(?P<optional>\\s+-Optional){0,1}'))]
        self._re_ps_module = [re.compile(to_bytes('(?i)^#\\s*requires\\s+\\-module(?:s?)\\s*(Ansible\\.ModuleUtils\\..+)')), re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-powershell\\s+((Ansible\\.ModuleUtils\\.[\\w\\.]+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+)|(\\.[\\w\\.]+))(?P<optional>\\s+-Optional){0,1}'))]
        self._re_wrapper = re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-wrapper\\s+(\\w*)'))
        self._re_ps_version = re.compile(to_bytes('(?i)^#requires\\s+\\-version\\s+([0-9]+(\\.[0-9]+){0,3})$'))
        self._re_os_version = re.compile(to_bytes('(?i)^#ansiblerequires\\s+\\-osversion\\s+([0-9]+(\\.[0-9]+){0,3})$'))
        self._re_become = re.compile(to_bytes('(?i)^#ansiblerequires\\s+\\-become$'))

    def scan_module(self, module_data, fqn=None, wrapper=False, powershell=True):
        if False:
            print('Hello World!')
        lines = module_data.split(b'\n')
        module_utils = set()
        if wrapper:
            cs_utils = self.cs_utils_wrapper
        else:
            cs_utils = self.cs_utils_module
        if powershell:
            checks = [(self._re_ps_module, self.ps_modules, '.psm1'), (self._re_cs_in_ps_module, cs_utils, '.cs')]
        else:
            checks = [(self._re_cs_module, cs_utils, '.cs')]
        for line in lines:
            for check in checks:
                for pattern in check[0]:
                    match = pattern.match(line)
                    if match:
                        module_util_name = to_text(match.group(1).rstrip())
                        match_dict = match.groupdict()
                        optional = match_dict.get('optional', None) is not None
                        if module_util_name not in check[1].keys():
                            module_utils.add((module_util_name, check[2], fqn, optional))
                        break
            if powershell:
                ps_version_match = self._re_ps_version.match(line)
                if ps_version_match:
                    self._parse_version_match(ps_version_match, 'ps_version')
                os_version_match = self._re_os_version.match(line)
                if os_version_match:
                    self._parse_version_match(os_version_match, 'os_version')
                if not self.become:
                    become_match = self._re_become.match(line)
                    if become_match:
                        self.become = True
            if wrapper:
                wrapper_match = self._re_wrapper.match(line)
                if wrapper_match:
                    self.scan_exec_script(wrapper_match.group(1).rstrip())
        for m in set(module_utils):
            self._add_module(*m, wrapper=wrapper)

    def scan_exec_script(self, name):
        if False:
            print('Hello World!')
        name = to_text(name)
        if name in self.exec_scripts.keys():
            return
        data = pkgutil.get_data('ansible.executor.powershell', to_native(name + '.ps1'))
        if data is None:
            raise AnsibleError("Could not find executor powershell script for '%s'" % name)
        b_data = to_bytes(data)
        if C.DEFAULT_DEBUG:
            exec_script = b_data
        else:
            exec_script = _strip_comments(b_data)
        self.exec_scripts[name] = to_bytes(exec_script)
        self.scan_module(b_data, wrapper=True, powershell=True)

    def _add_module(self, name, ext, fqn, optional, wrapper=False):
        if False:
            print('Hello World!')
        m = to_text(name)
        util_fqn = None
        if m.startswith('Ansible.'):
            mu_path = ps_module_utils_loader.find_plugin(m, ext)
            if not mu_path:
                if optional:
                    return
                raise AnsibleError("Could not find imported module support code for '%s'" % m)
            module_util_data = to_bytes(_slurp(mu_path))
        else:
            submodules = m.split('.')
            if m.startswith('.'):
                fqn_submodules = fqn.split('.')
                for submodule in submodules:
                    if submodule:
                        break
                    del fqn_submodules[-1]
                submodules = fqn_submodules + [s for s in submodules if s]
            n_package_name = to_native('.'.join(submodules[:-1]), errors='surrogate_or_strict')
            n_resource_name = to_native(submodules[-1] + ext, errors='surrogate_or_strict')
            try:
                module_util = import_module(n_package_name)
                pkg_data = pkgutil.get_data(n_package_name, n_resource_name)
                if pkg_data is None:
                    raise ImportError('No package data found')
                module_util_data = to_bytes(pkg_data, errors='surrogate_or_strict')
                util_fqn = to_text('%s.%s ' % (n_package_name, submodules[-1]), errors='surrogate_or_strict')
                resource_paths = list(module_util.__path__)
                if len(resource_paths) != 1:
                    raise AnsibleError("Internal error: Referenced module_util package '%s' contains 0 or multiple import locations when we only expect 1." % n_package_name)
                mu_path = os.path.join(resource_paths[0], n_resource_name)
            except (ImportError, OSError) as err:
                if getattr(err, 'errno', errno.ENOENT) == errno.ENOENT:
                    if optional:
                        return
                    raise AnsibleError("Could not find collection imported module support code for '%s'" % to_native(m))
                else:
                    raise
        util_info = {'data': module_util_data, 'path': to_text(mu_path)}
        if ext == '.psm1':
            self.ps_modules[m] = util_info
        elif wrapper:
            self.cs_utils_wrapper[m] = util_info
        else:
            self.cs_utils_module[m] = util_info
        self.scan_module(module_util_data, fqn=util_fqn, wrapper=wrapper, powershell=ext == '.psm1')

    def _parse_version_match(self, match, attribute):
        if False:
            return 10
        new_version = to_text(match.group(1)).rstrip()
        if match.group(2) is None:
            new_version = '%s.0' % new_version
        existing_version = getattr(self, attribute, None)
        if existing_version is None:
            setattr(self, attribute, new_version)
        elif LooseVersion(new_version) > LooseVersion(existing_version):
            setattr(self, attribute, new_version)

def _slurp(path):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(path):
        raise AnsibleError('imported module support code does not exist at %s' % os.path.abspath(path))
    fd = open(path, 'rb')
    data = fd.read()
    fd.close()
    return data

def _strip_comments(source):
    if False:
        i = 10
        return i + 15
    buf = []
    start_block = False
    for line in source.splitlines():
        l = line.strip()
        if start_block and l.endswith(b'#>'):
            start_block = False
            continue
        elif start_block:
            continue
        elif l.startswith(b'<#'):
            start_block = True
            continue
        elif not l or l.startswith(b'#'):
            continue
        buf.append(line)
    return b'\n'.join(buf)

def _create_powershell_wrapper(b_module_data, module_path, module_args, environment, async_timeout, become, become_method, become_user, become_password, become_flags, substyle, task_vars, module_fqn):
    if False:
        return 10
    finder = PSModuleDepFinder()
    if substyle != 'script':
        finder.scan_module(b_module_data, fqn=module_fqn, powershell=substyle == 'powershell')
    module_wrapper = 'module_%s_wrapper' % substyle
    exec_manifest = dict(module_entry=to_text(base64.b64encode(b_module_data)), powershell_modules=dict(), csharp_utils=dict(), csharp_utils_module=list(), module_args=module_args, actions=[module_wrapper], environment=environment, encoded_output=False)
    finder.scan_exec_script(module_wrapper)
    if async_timeout > 0:
        finder.scan_exec_script('exec_wrapper')
        finder.scan_exec_script('async_watchdog')
        finder.scan_exec_script('async_wrapper')
        exec_manifest['actions'].insert(0, 'async_watchdog')
        exec_manifest['actions'].insert(0, 'async_wrapper')
        exec_manifest['async_jid'] = f'j{random.randint(0, 999999999999)}'
        exec_manifest['async_timeout_sec'] = async_timeout
        exec_manifest['async_startup_timeout'] = C.config.get_config_value('WIN_ASYNC_STARTUP_TIMEOUT', variables=task_vars)
    if become and resource_from_fqcr(become_method) == 'runas':
        finder.scan_exec_script('exec_wrapper')
        finder.scan_exec_script('become_wrapper')
        exec_manifest['actions'].insert(0, 'become_wrapper')
        exec_manifest['become_user'] = become_user
        exec_manifest['become_password'] = become_password
        exec_manifest['become_flags'] = become_flags
    exec_manifest['min_ps_version'] = finder.ps_version
    exec_manifest['min_os_version'] = finder.os_version
    if finder.become and 'become_wrapper' not in exec_manifest['actions']:
        finder.scan_exec_script('exec_wrapper')
        finder.scan_exec_script('become_wrapper')
        exec_manifest['actions'].insert(0, 'become_wrapper')
        exec_manifest['become_user'] = 'SYSTEM'
        exec_manifest['become_password'] = None
        exec_manifest['become_flags'] = None
    coverage_manifest = dict(module_path=module_path, module_util_paths=dict(), output=None)
    coverage_output = C.config.get_config_value('COVERAGE_REMOTE_OUTPUT', variables=task_vars)
    if coverage_output and substyle == 'powershell':
        finder.scan_exec_script('coverage_wrapper')
        coverage_manifest['output'] = coverage_output
        coverage_enabled = C.config.get_config_value('COVERAGE_REMOTE_PATHS', variables=task_vars)
        coverage_manifest['path_filter'] = coverage_enabled
    if len(finder.cs_utils_wrapper) > 0 or len(finder.cs_utils_module) > 0:
        finder._add_module(b'Ansible.ModuleUtils.AddType', '.psm1', None, False, wrapper=False)
    exec_required = 'exec_wrapper' in finder.exec_scripts.keys()
    finder.scan_exec_script('exec_wrapper')
    finder.exec_scripts['exec_wrapper'] += b'\n\n'
    exec_wrapper = finder.exec_scripts['exec_wrapper']
    if not exec_required:
        finder.exec_scripts.pop('exec_wrapper')
    for (name, data) in finder.exec_scripts.items():
        b64_data = to_text(base64.b64encode(data))
        exec_manifest[name] = b64_data
    for (name, data) in finder.ps_modules.items():
        b64_data = to_text(base64.b64encode(data['data']))
        exec_manifest['powershell_modules'][name] = b64_data
        coverage_manifest['module_util_paths'][name] = data['path']
    cs_utils = {}
    for cs_util in [finder.cs_utils_wrapper, finder.cs_utils_module]:
        for (name, data) in cs_util.items():
            cs_utils[name] = data['data']
    for (name, data) in cs_utils.items():
        b64_data = to_text(base64.b64encode(data))
        exec_manifest['csharp_utils'][name] = b64_data
    exec_manifest['csharp_utils_module'] = list(finder.cs_utils_module.keys())
    if 'coverage_wrapper' in exec_manifest:
        exec_manifest['coverage'] = coverage_manifest
    b_json = to_bytes(json.dumps(exec_manifest))
    b_data = exec_wrapper + b'\x00\x00\x00\x00' + b_json
    return b_data