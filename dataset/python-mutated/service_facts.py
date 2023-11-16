from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: service_facts\nshort_description: Return service state information as fact data\ndescription:\n     - Return service state information as fact data for various service management utilities.\nversion_added: "2.5"\nrequirements: ["Any of the following supported init systems: systemd, sysv, upstart, openrc, AIX SRC"]\nextends_documentation_fragment:\n  -  action_common_attributes\n  -  action_common_attributes.facts\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    facts:\n        support: full\n    platform:\n        platforms: posix\nnotes:\n  - When accessing the RV(ansible_facts.services) facts collected by this module,\n    it is recommended to not use "dot notation" because services can have a C(-)\n    character in their name which would result in invalid "dot notation", such as\n    C(ansible_facts.services.zuul-gateway). It is instead recommended to\n    using the string value of the service name as the key in order to obtain\n    the fact data value like C(ansible_facts.services[\'zuul-gateway\'])\n  - AIX SRC was added in version 2.11.\nauthor:\n  - Adam Miller (@maxamillion)\n'
EXAMPLES = '\n- name: Populate service facts\n  ansible.builtin.service_facts:\n\n- name: Print service facts\n  ansible.builtin.debug:\n    var: ansible_facts.services\n'
RETURN = "\nansible_facts:\n  description: Facts to add to ansible_facts about the services on the system\n  returned: always\n  type: complex\n  contains:\n    services:\n      description: States of the services with service name as key.\n      returned: always\n      type: list\n      elements: dict\n      contains:\n        source:\n          description:\n          - Init system of the service.\n          - One of V(rcctl), V(systemd), V(sysv), V(upstart), V(src).\n          returned: always\n          type: str\n          sample: sysv\n        state:\n          description:\n          - State of the service.\n          - 'This commonly includes (but is not limited to) the following: V(failed), V(running), V(stopped) or V(unknown).'\n          - Depending on the used init system additional states might be returned.\n          returned: always\n          type: str\n          sample: running\n        status:\n          description:\n          - State of the service.\n          - Either V(enabled), V(disabled), V(static), V(indirect) or V(unknown).\n          returned: systemd systems or RedHat/SUSE flavored sysvinit/upstart or OpenBSD\n          type: str\n          sample: enabled\n        name:\n          description: Name of the service.\n          returned: always\n          type: str\n          sample: arp-ethers.service\n"
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.service import is_systemd_managed

class BaseService(object):

    def __init__(self, module):
        if False:
            i = 10
            return i + 15
        self.module = module

class ServiceScanService(BaseService):

    def _list_sysvinit(self, services):
        if False:
            for i in range(10):
                print('nop')
        (rc, stdout, stderr) = self.module.run_command('%s --status-all' % self.service_path)
        if rc == 4 and (not os.path.exists('/etc/init.d')):
            return
        if rc != 0:
            self.module.warn("Unable to query 'service' tool (%s): %s" % (rc, stderr))
        p = re.compile('^\\s*\\[ (?P<state>\\+|\\-) \\]\\s+(?P<name>.+)$', flags=re.M)
        for match in p.finditer(stdout):
            service_name = match.group('name')
            if match.group('state') == '+':
                service_state = 'running'
            else:
                service_state = 'stopped'
            services[service_name] = {'name': service_name, 'state': service_state, 'source': 'sysv'}

    def _list_upstart(self, services):
        if False:
            i = 10
            return i + 15
        p = re.compile('^\\s?(?P<name>.*)\\s(?P<goal>\\w+)\\/(?P<state>\\w+)(\\,\\sprocess\\s(?P<pid>[0-9]+))?\\s*$')
        (rc, stdout, stderr) = self.module.run_command('%s list' % self.initctl_path)
        if rc != 0:
            self.module.warn('Unable to query upstart for service data: %s' % stderr)
        else:
            real_stdout = stdout.replace('\r', '')
            for line in real_stdout.split('\n'):
                m = p.match(line)
                if not m:
                    continue
                service_name = m.group('name')
                service_goal = m.group('goal')
                service_state = m.group('state')
                if m.group('pid'):
                    pid = m.group('pid')
                else:
                    pid = None
                payload = {'name': service_name, 'state': service_state, 'goal': service_goal, 'source': 'upstart'}
                services[service_name] = payload

    def _list_rh(self, services):
        if False:
            print('Hello World!')
        p = re.compile('(?P<service>.*?)\\s+[0-9]:(?P<rl0>on|off)\\s+[0-9]:(?P<rl1>on|off)\\s+[0-9]:(?P<rl2>on|off)\\s+[0-9]:(?P<rl3>on|off)\\s+[0-9]:(?P<rl4>on|off)\\s+[0-9]:(?P<rl5>on|off)\\s+[0-9]:(?P<rl6>on|off)')
        (rc, stdout, stderr) = self.module.run_command('%s' % self.chkconfig_path, use_unsafe_shell=True)
        match_any = False
        for line in stdout.split('\n'):
            if p.match(line):
                match_any = True
        if not match_any:
            p_simple = re.compile('(?P<service>.*?)\\s+(?P<rl0>on|off)')
            match_any = False
            for line in stdout.split('\n'):
                if p_simple.match(line):
                    match_any = True
            if match_any:
                (rc, stdout, stderr) = self.module.run_command('%s -l --allservices' % self.chkconfig_path, use_unsafe_shell=True)
            elif '--list' in stderr:
                (rc, stdout, stderr) = self.module.run_command('%s --list' % self.chkconfig_path, use_unsafe_shell=True)
        for line in stdout.split('\n'):
            m = p.match(line)
            if m:
                service_name = m.group('service')
                service_state = 'stopped'
                service_status = 'disabled'
                if m.group('rl3') == 'on':
                    service_status = 'enabled'
                (rc, stdout, stderr) = self.module.run_command('%s %s status' % (self.service_path, service_name), use_unsafe_shell=True)
                service_state = rc
                if rc in (0,):
                    service_state = 'running'
                else:
                    output = stderr.lower()
                    for x in ('root', 'permission', 'not in sudoers'):
                        if x in output:
                            self.module.warn('Insufficient permissions to query sysV service "%s" and their states' % service_name)
                            break
                    else:
                        service_state = 'stopped'
                service_data = {'name': service_name, 'state': service_state, 'status': service_status, 'source': 'sysv'}
                services[service_name] = service_data

    def _list_openrc(self, services):
        if False:
            return 10
        all_services_runlevels = {}
        (rc, stdout, stderr) = self.module.run_command("%s -a -s -m 2>&1 | grep '^ ' | tr -d '[]'" % self.rc_status_path, use_unsafe_shell=True)
        (rc_u, stdout_u, stderr_u) = self.module.run_command("%s show -v 2>&1 | grep '|'" % self.rc_update_path, use_unsafe_shell=True)
        for line in stdout_u.split('\n'):
            line_data = line.split('|')
            if len(line_data) < 2:
                continue
            service_name = line_data[0].strip()
            runlevels = line_data[1].strip()
            if not runlevels:
                all_services_runlevels[service_name] = None
            else:
                all_services_runlevels[service_name] = runlevels.split()
        for line in stdout.split('\n'):
            line_data = line.split()
            if len(line_data) < 2:
                continue
            service_name = line_data[0]
            service_state = line_data[1]
            service_runlevels = all_services_runlevels[service_name]
            service_data = {'name': service_name, 'runlevels': service_runlevels, 'state': service_state, 'source': 'openrc'}
            services[service_name] = service_data

    def gather_services(self):
        if False:
            return 10
        services = {}
        self.service_path = self.module.get_bin_path('service')
        self.chkconfig_path = self.module.get_bin_path('chkconfig')
        self.initctl_path = self.module.get_bin_path('initctl')
        self.rc_status_path = self.module.get_bin_path('rc-status')
        self.rc_update_path = self.module.get_bin_path('rc-update')
        if self.service_path and self.chkconfig_path is None and (self.rc_status_path is None):
            self._list_sysvinit(services)
        if self.initctl_path and self.chkconfig_path is None:
            self._list_upstart(services)
        elif self.chkconfig_path:
            self._list_rh(services)
        elif self.rc_status_path is not None and self.rc_update_path is not None:
            self._list_openrc(services)
        return services

class SystemctlScanService(BaseService):
    BAD_STATES = frozenset(['not-found', 'masked', 'failed'])

    def systemd_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return is_systemd_managed(self.module)

    def _list_from_units(self, systemctl_path, services):
        if False:
            return 10
        (rc, stdout, stderr) = self.module.run_command('%s list-units --no-pager --type service --all' % systemctl_path, use_unsafe_shell=True)
        if rc != 0:
            self.module.warn('Could not list units from systemd: %s' % stderr)
        else:
            for line in [svc_line for svc_line in stdout.split('\n') if '.service' in svc_line]:
                state_val = 'stopped'
                status_val = 'unknown'
                fields = line.split()
                for bad in self.BAD_STATES:
                    if bad in fields:
                        status_val = bad
                        fields = fields[1:]
                        break
                else:
                    status_val = fields[2]
                service_name = fields[0]
                if fields[3] == 'running':
                    state_val = 'running'
                services[service_name] = {'name': service_name, 'state': state_val, 'status': status_val, 'source': 'systemd'}

    def _list_from_unit_files(self, systemctl_path, services):
        if False:
            i = 10
            return i + 15
        (rc, stdout, stderr) = self.module.run_command('%s list-unit-files --no-pager --type service --all' % systemctl_path, use_unsafe_shell=True)
        if rc != 0:
            self.module.warn('Could not get unit files data from systemd: %s' % stderr)
        else:
            for line in [svc_line for svc_line in stdout.split('\n') if '.service' in svc_line]:
                try:
                    (service_name, status_val) = line.split()[:2]
                except IndexError:
                    self.module.fail_json(msg='Malformed output discovered from systemd list-unit-files: {0}'.format(line))
                if service_name not in services:
                    (rc, stdout, stderr) = self.module.run_command('%s show %s --property=ActiveState' % (systemctl_path, service_name), use_unsafe_shell=True)
                    state = 'unknown'
                    if not rc and stdout != '':
                        state = stdout.replace('ActiveState=', '').rstrip()
                    services[service_name] = {'name': service_name, 'state': state, 'status': status_val, 'source': 'systemd'}
                elif services[service_name]['status'] not in self.BAD_STATES:
                    services[service_name]['status'] = status_val

    def gather_services(self):
        if False:
            print('Hello World!')
        services = {}
        if self.systemd_enabled():
            systemctl_path = self.module.get_bin_path('systemctl', opt_dirs=['/usr/bin', '/usr/local/bin'])
            if systemctl_path:
                self._list_from_units(systemctl_path, services)
                self._list_from_unit_files(systemctl_path, services)
        return services

class AIXScanService(BaseService):

    def gather_services(self):
        if False:
            return 10
        services = {}
        if platform.system() == 'AIX':
            lssrc_path = self.module.get_bin_path('lssrc')
            if lssrc_path:
                (rc, stdout, stderr) = self.module.run_command('%s -a' % lssrc_path)
                if rc != 0:
                    self.module.warn('lssrc could not retrieve service data (%s): %s' % (rc, stderr))
                else:
                    for line in stdout.split('\n'):
                        line_data = line.split()
                        if len(line_data) < 2:
                            continue
                        if line_data[0] == 'Subsystem':
                            continue
                        service_name = line_data[0]
                        if line_data[-1] == 'active':
                            service_state = 'running'
                        elif line_data[-1] == 'inoperative':
                            service_state = 'stopped'
                        else:
                            service_state = 'unknown'
                        services[service_name] = {'name': service_name, 'state': service_state, 'source': 'src'}
        return services

class OpenBSDScanService(BaseService):

    def query_rcctl(self, cmd):
        if False:
            return 10
        svcs = []
        (rc, stdout, stderr) = self.module.run_command('%s ls %s' % (self.rcctl_path, cmd))
        if 'needs root privileges' in stderr.lower():
            self.module.warn('rcctl requires root privileges')
        else:
            for svc in stdout.split('\n'):
                if svc == '':
                    continue
                else:
                    svcs.append(svc)
        return svcs

    def get_info(self, name):
        if False:
            i = 10
            return i + 15
        info = {}
        (rc, stdout, stderr) = self.module.run_command('%s get %s' % (self.rcctl_path, name))
        if 'needs root privileges' in stderr.lower():
            self.module.warn('rcctl requires root privileges')
        else:
            undy = '%s_' % name
            for variable in stdout.split('\n'):
                if variable == '' or '=' not in variable:
                    continue
                else:
                    (k, v) = variable.replace(undy, '', 1).split('=')
                    info[k] = v
        return info

    def gather_services(self):
        if False:
            while True:
                i = 10
        services = {}
        self.rcctl_path = self.module.get_bin_path('rcctl')
        if self.rcctl_path:
            for svc in self.query_rcctl('all'):
                services[svc] = {'name': svc, 'source': 'rcctl', 'rogue': False}
                services[svc].update(self.get_info(svc))
            for svc in self.query_rcctl('on'):
                services[svc].update({'status': 'enabled'})
            for svc in self.query_rcctl('started'):
                services[svc].update({'state': 'running'})
            for svc in self.query_rcctl('failed'):
                services[svc].update({'state': 'failed'})
            for svc in services.keys():
                if services[svc].get('status') is None:
                    services[svc].update({'status': 'disabled'})
                if services[svc].get('state') is None:
                    services[svc].update({'state': 'stopped'})
            for svc in self.query_rcctl('rogue'):
                services[svc]['rogue'] = True
        return services

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(), supports_check_mode=True)
    locale = get_best_parsable_locale(module)
    module.run_command_environ_update = dict(LANG=locale, LC_ALL=locale)
    service_modules = (ServiceScanService, SystemctlScanService, AIXScanService, OpenBSDScanService)
    all_services = {}
    for svc_module in service_modules:
        svcmod = svc_module(module)
        svc = svcmod.gather_services()
        if svc:
            all_services.update(svc)
    if len(all_services) == 0:
        results = dict(skipped=True, msg='Failed to find any services. This can be due to privileges or some other configuration issue.')
    else:
        results = dict(ansible_facts=dict(services=all_services))
    module.exit_json(**results)
if __name__ == '__main__':
    main()