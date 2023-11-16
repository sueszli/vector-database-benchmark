from __future__ import annotations
DOCUMENTATION = '\nmodule: systemd_service\nauthor:\n    - Ansible Core Team\nversion_added: "2.2"\nshort_description:  Manage systemd units\ndescription:\n    - Controls systemd units (services, timers, and so on) on remote hosts.\n    - M(ansible.builtin.systemd) is renamed to M(ansible.builtin.systemd_service) to better reflect the scope of the module.\n      M(ansible.builtin.systemd) is kept as an alias for backward compatibility.\noptions:\n    name:\n        description:\n            - Name of the unit. This parameter takes the name of exactly one unit to work with.\n            - When no extension is given, it is implied to a C(.service) as systemd.\n            - When using in a chroot environment you always need to specify the name of the unit with the extension. For example, C(crond.service).\n        type: str\n        aliases: [ service, unit ]\n    state:\n        description:\n            - V(started)/V(stopped) are idempotent actions that will not run commands unless necessary.\n              V(restarted) will always bounce the unit.\n              V(reloaded) will always reload and if the service is not running at the moment of the reload, it is started.\n        type: str\n        choices: [ reloaded, restarted, started, stopped ]\n    enabled:\n        description:\n            - Whether the unit should start on boot. B(At least one of the states and enabled are required.)\n        type: bool\n    force:\n        description:\n            - Whether to override existing symlinks.\n        type: bool\n        version_added: 2.6\n    masked:\n        description:\n            - Whether the unit should be masked or not, a masked unit is impossible to start.\n        type: bool\n    daemon_reload:\n        description:\n            - Run daemon-reload before doing any other operations, to make sure systemd has read any changes.\n            - When set to V(true), runs daemon-reload even if the module does not start or stop anything.\n        type: bool\n        default: no\n        aliases: [ daemon-reload ]\n    daemon_reexec:\n        description:\n            - Run daemon_reexec command before doing any other operations, the systemd manager will serialize the manager state.\n        type: bool\n        default: no\n        aliases: [ daemon-reexec ]\n        version_added: "2.8"\n    scope:\n        description:\n            - Run systemctl within a given service manager scope, either as the default system scope V(system),\n              the current user\'s scope V(user), or the scope of all users V(global).\n            - "For systemd to work with \'user\', the executing user must have its own instance of dbus started and accessible (systemd requirement)."\n            - "The user dbus process is normally started during normal login, but not during the run of Ansible tasks.\n              Otherwise you will probably get a \'Failed to connect to bus: no such file or directory\' error."\n            - The user must have access, normally given via setting the C(XDG_RUNTIME_DIR) variable, see the example below.\n\n        type: str\n        choices: [ system, user, global ]\n        default: system\n        version_added: "2.7"\n    no_block:\n        description:\n            - Do not synchronously wait for the requested operation to finish.\n              Enqueued job will continue without Ansible blocking on its completion.\n        type: bool\n        default: no\n        version_added: "2.3"\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    platform:\n        platforms: posix\nnotes:\n    - Since 2.4, one of the following options is required O(state), O(enabled), O(masked), O(daemon_reload), (O(daemon_reexec) since 2.8),\n      and all except O(daemon_reload) and (O(daemon_reexec) since 2.8) also require O(name).\n    - Before 2.4 you always required O(name).\n    - Globs are not supported in name, in other words, C(postgres*.service).\n    - The service names might vary by specific OS/distribution\n    - The order of execution when having multiple properties is to first enable/disable, then mask/unmask and then deal with the service state.\n      It has been reported that systemctl can behave differently depending on the order of operations if you do the same manually.\nrequirements:\n    - A system managed by systemd.\n'
EXAMPLES = '\n- name: Make sure a service unit is running\n  ansible.builtin.systemd_service:\n    state: started\n    name: httpd\n\n- name: Stop service cron on debian, if running\n  ansible.builtin.systemd_service:\n    name: cron\n    state: stopped\n\n- name: Restart service cron on centos, in all cases, also issue daemon-reload to pick up config changes\n  ansible.builtin.systemd_service:\n    state: restarted\n    daemon_reload: true\n    name: crond\n\n- name: Reload service httpd, in all cases\n  ansible.builtin.systemd_service:\n    name: httpd.service\n    state: reloaded\n\n- name: Enable service httpd and ensure it is not masked\n  ansible.builtin.systemd_service:\n    name: httpd\n    enabled: true\n    masked: no\n\n- name: Enable a timer unit for dnf-automatic\n  ansible.builtin.systemd_service:\n    name: dnf-automatic.timer\n    state: started\n    enabled: true\n\n- name: Just force systemd to reread configs (2.4 and above)\n  ansible.builtin.systemd_service:\n    daemon_reload: true\n\n- name: Just force systemd to re-execute itself (2.8 and above)\n  ansible.builtin.systemd_service:\n    daemon_reexec: true\n\n- name: Run a user service when XDG_RUNTIME_DIR is not set on remote login\n  ansible.builtin.systemd_service:\n    name: myservice\n    state: started\n    scope: user\n  environment:\n    XDG_RUNTIME_DIR: "/run/user/{{ myuid }}"\n'
RETURN = '\nstatus:\n    description: A dictionary with the key=value pairs returned from C(systemctl show).\n    returned: success\n    type: dict\n    sample: {\n            "ActiveEnterTimestamp": "Sun 2016-05-15 18:28:49 EDT",\n            "ActiveEnterTimestampMonotonic": "8135942",\n            "ActiveExitTimestampMonotonic": "0",\n            "ActiveState": "active",\n            "After": "auditd.service systemd-user-sessions.service time-sync.target systemd-journald.socket basic.target system.slice",\n            "AllowIsolate": "no",\n            "Before": "shutdown.target multi-user.target",\n            "BlockIOAccounting": "no",\n            "BlockIOWeight": "1000",\n            "CPUAccounting": "no",\n            "CPUSchedulingPolicy": "0",\n            "CPUSchedulingPriority": "0",\n            "CPUSchedulingResetOnFork": "no",\n            "CPUShares": "1024",\n            "CanIsolate": "no",\n            "CanReload": "yes",\n            "CanStart": "yes",\n            "CanStop": "yes",\n            "CapabilityBoundingSet": "18446744073709551615",\n            "ConditionResult": "yes",\n            "ConditionTimestamp": "Sun 2016-05-15 18:28:49 EDT",\n            "ConditionTimestampMonotonic": "7902742",\n            "Conflicts": "shutdown.target",\n            "ControlGroup": "/system.slice/crond.service",\n            "ControlPID": "0",\n            "DefaultDependencies": "yes",\n            "Delegate": "no",\n            "Description": "Command Scheduler",\n            "DevicePolicy": "auto",\n            "EnvironmentFile": "/etc/sysconfig/crond (ignore_errors=no)",\n            "ExecMainCode": "0",\n            "ExecMainExitTimestampMonotonic": "0",\n            "ExecMainPID": "595",\n            "ExecMainStartTimestamp": "Sun 2016-05-15 18:28:49 EDT",\n            "ExecMainStartTimestampMonotonic": "8134990",\n            "ExecMainStatus": "0",\n            "ExecReload": "{ path=/bin/kill ; argv[]=/bin/kill -HUP $MAINPID ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }",\n            "ExecStart": "{ path=/usr/sbin/crond ; argv[]=/usr/sbin/crond -n $CRONDARGS ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }",\n            "FragmentPath": "/usr/lib/systemd/system/crond.service",\n            "GuessMainPID": "yes",\n            "IOScheduling": "0",\n            "Id": "crond.service",\n            "IgnoreOnIsolate": "no",\n            "IgnoreOnSnapshot": "no",\n            "IgnoreSIGPIPE": "yes",\n            "InactiveEnterTimestampMonotonic": "0",\n            "InactiveExitTimestamp": "Sun 2016-05-15 18:28:49 EDT",\n            "InactiveExitTimestampMonotonic": "8135942",\n            "JobTimeoutUSec": "0",\n            "KillMode": "process",\n            "KillSignal": "15",\n            "LimitAS": "18446744073709551615",\n            "LimitCORE": "18446744073709551615",\n            "LimitCPU": "18446744073709551615",\n            "LimitDATA": "18446744073709551615",\n            "LimitFSIZE": "18446744073709551615",\n            "LimitLOCKS": "18446744073709551615",\n            "LimitMEMLOCK": "65536",\n            "LimitMSGQUEUE": "819200",\n            "LimitNICE": "0",\n            "LimitNOFILE": "4096",\n            "LimitNPROC": "3902",\n            "LimitRSS": "18446744073709551615",\n            "LimitRTPRIO": "0",\n            "LimitRTTIME": "18446744073709551615",\n            "LimitSIGPENDING": "3902",\n            "LimitSTACK": "18446744073709551615",\n            "LoadState": "loaded",\n            "MainPID": "595",\n            "MemoryAccounting": "no",\n            "MemoryLimit": "18446744073709551615",\n            "MountFlags": "0",\n            "Names": "crond.service",\n            "NeedDaemonReload": "no",\n            "Nice": "0",\n            "NoNewPrivileges": "no",\n            "NonBlocking": "no",\n            "NotifyAccess": "none",\n            "OOMScoreAdjust": "0",\n            "OnFailureIsolate": "no",\n            "PermissionsStartOnly": "no",\n            "PrivateNetwork": "no",\n            "PrivateTmp": "no",\n            "RefuseManualStart": "no",\n            "RefuseManualStop": "no",\n            "RemainAfterExit": "no",\n            "Requires": "basic.target",\n            "Restart": "no",\n            "RestartUSec": "100ms",\n            "Result": "success",\n            "RootDirectoryStartOnly": "no",\n            "SameProcessGroup": "no",\n            "SecureBits": "0",\n            "SendSIGHUP": "no",\n            "SendSIGKILL": "yes",\n            "Slice": "system.slice",\n            "StandardError": "inherit",\n            "StandardInput": "null",\n            "StandardOutput": "journal",\n            "StartLimitAction": "none",\n            "StartLimitBurst": "5",\n            "StartLimitInterval": "10000000",\n            "StatusErrno": "0",\n            "StopWhenUnneeded": "no",\n            "SubState": "running",\n            "SyslogLevelPrefix": "yes",\n            "SyslogPriority": "30",\n            "TTYReset": "no",\n            "TTYVHangup": "no",\n            "TTYVTDisallocate": "no",\n            "TimeoutStartUSec": "1min 30s",\n            "TimeoutStopUSec": "1min 30s",\n            "TimerSlackNSec": "50000",\n            "Transient": "no",\n            "Type": "simple",\n            "UMask": "0022",\n            "UnitFileState": "enabled",\n            "WantedBy": "multi-user.target",\n            "Wants": "system.slice",\n            "WatchdogTimestampMonotonic": "0",\n            "WatchdogUSec": "0",\n        }\n'
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.facts.system.chroot import is_chroot
from ansible.module_utils.service import sysv_exists, sysv_is_enabled, fail_if_missing
from ansible.module_utils.common.text.converters import to_native

def is_running_service(service_status):
    if False:
        for i in range(10):
            print('nop')
    return service_status['ActiveState'] in set(['active', 'activating'])

def is_deactivating_service(service_status):
    if False:
        i = 10
        return i + 15
    return service_status['ActiveState'] in set(['deactivating'])

def request_was_ignored(out):
    if False:
        i = 10
        return i + 15
    return '=' not in out and ('ignoring request' in out or 'ignoring command' in out)

def parse_systemctl_show(lines):
    if False:
        while True:
            i = 10
    parsed = {}
    multival = []
    k = None
    for line in lines:
        if k is None:
            if '=' in line:
                (k, v) = line.split('=', 1)
                if k.startswith('Exec') and v.lstrip().startswith('{'):
                    if not v.rstrip().endswith('}'):
                        multival.append(v)
                        continue
                parsed[k] = v.strip()
                k = None
        else:
            multival.append(line)
            if line.rstrip().endswith('}'):
                parsed[k] = '\n'.join(multival).strip()
                multival = []
                k = None
    return parsed

def main():
    if False:
        for i in range(10):
            print('nop')
    module = AnsibleModule(argument_spec=dict(name=dict(type='str', aliases=['service', 'unit']), state=dict(type='str', choices=['reloaded', 'restarted', 'started', 'stopped']), enabled=dict(type='bool'), force=dict(type='bool'), masked=dict(type='bool'), daemon_reload=dict(type='bool', default=False, aliases=['daemon-reload']), daemon_reexec=dict(type='bool', default=False, aliases=['daemon-reexec']), scope=dict(type='str', default='system', choices=['system', 'user', 'global']), no_block=dict(type='bool', default=False)), supports_check_mode=True, required_one_of=[['state', 'enabled', 'masked', 'daemon_reload', 'daemon_reexec']], required_by=dict(state=('name',), enabled=('name',), masked=('name',)))
    unit = module.params['name']
    if unit is not None:
        for globpattern in ('*', '?', '['):
            if globpattern in unit:
                module.fail_json(msg="This module does not currently support using glob patterns, found '%s' in service name: %s" % (globpattern, unit))
    systemctl = module.get_bin_path('systemctl', True)
    if os.getenv('XDG_RUNTIME_DIR') is None:
        os.environ['XDG_RUNTIME_DIR'] = '/run/user/%s' % os.geteuid()
    if module.params['scope'] != 'system':
        systemctl += ' --%s' % module.params['scope']
    if module.params['no_block']:
        systemctl += ' --no-block'
    if module.params['force']:
        systemctl += ' --force'
    rc = 0
    out = err = ''
    result = dict(name=unit, changed=False, status=dict())
    if module.params['daemon_reload'] and (not module.check_mode):
        (rc, out, err) = module.run_command('%s daemon-reload' % systemctl)
        if rc != 0:
            if is_chroot(module) or os.environ.get('SYSTEMD_OFFLINE') == '1':
                module.warn('daemon-reload failed, but target is a chroot or systemd is offline. Continuing. Error was: %d / %s' % (rc, err))
            else:
                module.fail_json(msg='failure %d during daemon-reload: %s' % (rc, err))
    if module.params['daemon_reexec'] and (not module.check_mode):
        (rc, out, err) = module.run_command('%s daemon-reexec' % systemctl)
        if rc != 0:
            if is_chroot(module) or os.environ.get('SYSTEMD_OFFLINE') == '1':
                module.warn('daemon-reexec failed, but target is a chroot or systemd is offline. Continuing. Error was: %d / %s' % (rc, err))
            else:
                module.fail_json(msg='failure %d during daemon-reexec: %s' % (rc, err))
    if unit:
        found = False
        is_initd = sysv_exists(unit)
        is_systemd = False
        (rc, out, err) = module.run_command("%s show '%s'" % (systemctl, unit))
        if rc == 0 and (not (request_was_ignored(out) or request_was_ignored(err))):
            if out:
                result['status'] = parse_systemctl_show(to_native(out).split('\n'))
                is_systemd = 'LoadState' in result['status'] and result['status']['LoadState'] != 'not-found'
                is_masked = 'LoadState' in result['status'] and result['status']['LoadState'] == 'masked'
                if is_systemd and (not is_masked) and ('LoadError' in result['status']):
                    module.fail_json(msg="Error loading unit file '%s': %s" % (unit, result['status']['LoadError']))
        elif err and rc == 1 and ('Failed to parse bus message' in err):
            result['status'] = parse_systemctl_show(to_native(out).split('\n'))
            (unit_base, sep, suffix) = unit.partition('@')
            unit_search = '{unit_base}{sep}'.format(unit_base=unit_base, sep=sep)
            (rc, out, err) = module.run_command("{systemctl} list-unit-files '{unit_search}*'".format(systemctl=systemctl, unit_search=unit_search))
            is_systemd = unit_search in out
            (rc, out, err) = module.run_command("{systemctl} is-active '{unit}'".format(systemctl=systemctl, unit=unit))
            result['status']['ActiveState'] = out.rstrip('\n')
        else:
            valid_enabled_states = ['enabled', 'enabled-runtime', 'linked', 'linked-runtime', 'masked', 'masked-runtime', 'static', 'indirect', 'disabled', 'generated', 'transient']
            (rc, out, err) = module.run_command("%s is-enabled '%s'" % (systemctl, unit))
            if out.strip() in valid_enabled_states:
                is_systemd = True
            else:
                (rc, out, err) = module.run_command("%s list-unit-files '%s'" % (systemctl, unit))
                if rc == 0:
                    is_systemd = True
                else:
                    module.run_command(systemctl, check_rc=True)
        found = is_systemd or is_initd
        if is_initd and (not is_systemd):
            module.warn('The service (%s) is actually an init script but the system is managed by systemd' % unit)
        if module.params['masked'] is not None:
            (rc, out, err) = module.run_command("%s is-enabled '%s'" % (systemctl, unit))
            masked = out.strip() == 'masked'
            if masked != module.params['masked']:
                result['changed'] = True
                if module.params['masked']:
                    action = 'mask'
                else:
                    action = 'unmask'
                if not module.check_mode:
                    (rc, out, err) = module.run_command("%s %s '%s'" % (systemctl, action, unit))
                    if rc != 0:
                        fail_if_missing(module, found, unit, msg='host')
        if module.params['enabled'] is not None:
            if module.params['enabled']:
                action = 'enable'
            else:
                action = 'disable'
            fail_if_missing(module, found, unit, msg='host')
            enabled = False
            (rc, out, err) = module.run_command("%s is-enabled '%s' -l" % (systemctl, unit))
            if rc == 0:
                enabled = True
                if out.splitlines() == ['indirect'] or out.splitlines() == ['alias']:
                    enabled = False
            elif rc == 1:
                if module.params['scope'] == 'system' and is_initd and (not out.strip().endswith('disabled')) and sysv_is_enabled(unit):
                    enabled = True
            result['enabled'] = enabled
            if enabled != module.params['enabled']:
                result['changed'] = True
                if not module.check_mode:
                    (rc, out, err) = module.run_command("%s %s '%s'" % (systemctl, action, unit))
                    if rc != 0:
                        module.fail_json(msg='Unable to %s service %s: %s' % (action, unit, out + err))
                result['enabled'] = not enabled
        if module.params['state'] is not None:
            fail_if_missing(module, found, unit, msg='host')
            result['state'] = module.params['state']
            if 'ActiveState' in result['status']:
                action = None
                if module.params['state'] == 'started':
                    if not is_running_service(result['status']):
                        action = 'start'
                elif module.params['state'] == 'stopped':
                    if is_running_service(result['status']) or is_deactivating_service(result['status']):
                        action = 'stop'
                else:
                    if not is_running_service(result['status']):
                        action = 'start'
                    else:
                        action = module.params['state'][:-2]
                    result['state'] = 'started'
                if action:
                    result['changed'] = True
                    if not module.check_mode:
                        (rc, out, err) = module.run_command("%s %s '%s'" % (systemctl, action, unit))
                        if rc != 0:
                            module.fail_json(msg='Unable to %s service %s: %s' % (action, unit, err))
            elif is_chroot(module) or os.environ.get('SYSTEMD_OFFLINE') == '1':
                module.warn('Target is a chroot or systemd is offline. This can lead to false positives or prevent the init system tools from working.')
            else:
                module.fail_json(msg='Service is in unknown state', status=result['status'])
    module.exit_json(**result)
if __name__ == '__main__':
    main()