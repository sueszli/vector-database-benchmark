from __future__ import annotations
DOCUMENTATION = '\nmodule: sysvinit\nauthor:\n    - "Ansible Core Team"\nversion_added: "2.6"\nshort_description:  Manage SysV services.\ndescription:\n    - Controls services on target hosts that use the SysV init system.\noptions:\n    name:\n        required: true\n        description:\n            - Name of the service.\n        type: str\n        aliases: [\'service\']\n    state:\n        choices: [ \'started\', \'stopped\', \'restarted\', \'reloaded\' ]\n        description:\n            - V(started)/V(stopped) are idempotent actions that will not run commands unless necessary.\n              Not all init scripts support V(restarted) nor V(reloaded) natively, so these will both trigger a stop and start as needed.\n        type: str\n    enabled:\n        type: bool\n        description:\n            - Whether the service should start on boot. B(At least one of state and enabled are required.)\n    sleep:\n        default: 1\n        description:\n            - If the service is being V(restarted) or V(reloaded) then sleep this many seconds between the stop and start command.\n              This helps to workaround badly behaving services.\n        type: int\n    pattern:\n        description:\n            - A substring to look for as would be found in the output of the I(ps) command as a stand-in for a status result.\n            - If the string is found, the service will be assumed to be running.\n            - "This option is mainly for use with init scripts that don\'t support the \'status\' option."\n        type: str\n    runlevels:\n        description:\n            - The runlevels this script should be enabled/disabled from.\n            - Use this to override the defaults set by the package or init script itself.\n        type: list\n        elements: str\n    arguments:\n        description:\n            - Additional arguments provided on the command line that some init scripts accept.\n        type: str\n        aliases: [ \'args\' ]\n    daemonize:\n        type: bool\n        description:\n            - Have the module daemonize as the service itself might not do so properly.\n            - This is useful with badly written init scripts or daemons, which\n              commonly manifests as the task hanging as it is still holding the\n              tty or the service dying when the task is over as the connection\n              closes the session.\n        default: no\nextends_documentation_fragment: action_common_attributes\nattributes:\n    check_mode:\n        support: full\n    diff_mode:\n        support: none\n    platform:\n        platforms: posix\nnotes:\n    - One option other than name is required.\n    - The service names might vary by specific OS/distribution\nrequirements:\n    - That the service managed has a corresponding init script.\n'
EXAMPLES = '\n- name: Make sure apache2 is started\n  ansible.builtin.sysvinit:\n      name: apache2\n      state: started\n      enabled: yes\n\n- name: Sleep for 5 seconds between stop and start command of badly behaving service\n  ansible.builtin.sysvinit:\n    name: apache2\n    state: restarted\n    sleep: 5\n\n- name: Make sure apache2 is started on runlevels 3 and 5\n  ansible.builtin.sysvinit:\n      name: apache2\n      state: started\n      enabled: yes\n      runlevels:\n        - 3\n        - 5\n'
RETURN = '\nresults:\n    description: results from actions taken\n    returned: always\n    type: complex\n    contains:\n      name:\n        description: Name of the service\n        type: str\n        returned: always\n        sample: "apache2"\n      status:\n        description: Status of the service\n        type: dict\n        returned: changed\n        sample: {\n          "enabled": {\n             "changed": true,\n             "rc": 0,\n             "stderr": "",\n             "stdout": ""\n          },\n          "stopped": {\n             "changed": true,\n             "rc": 0,\n             "stderr": "",\n             "stdout": "Stopping web server: apache2.\\n"\n          }\n        }\n'
import re
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import sysv_is_enabled, get_sysv_script, sysv_exists, fail_if_missing, get_ps, daemonize

def main():
    if False:
        print('Hello World!')
    module = AnsibleModule(argument_spec=dict(name=dict(required=True, type='str', aliases=['service']), state=dict(choices=['started', 'stopped', 'restarted', 'reloaded'], type='str'), enabled=dict(type='bool'), sleep=dict(type='int', default=1), pattern=dict(type='str'), arguments=dict(type='str', aliases=['args']), runlevels=dict(type='list', elements='str'), daemonize=dict(type='bool', default=False)), supports_check_mode=True, required_one_of=[['state', 'enabled']])
    name = module.params['name']
    action = module.params['state']
    enabled = module.params['enabled']
    runlevels = module.params['runlevels']
    pattern = module.params['pattern']
    sleep_for = module.params['sleep']
    rc = 0
    out = err = ''
    result = {'name': name, 'changed': False, 'status': {}}
    fail_if_missing(module, sysv_exists(name), name)
    script = get_sysv_script(name)
    paths = ['/sbin', '/usr/sbin', '/bin', '/usr/bin']
    binaries = ['chkconfig', 'update-rc.d', 'insserv', 'service']
    runlevel_status = {}
    location = {}
    for binary in binaries:
        location[binary] = module.get_bin_path(binary, opt_dirs=paths)
    if runlevels:
        for rl in runlevels:
            runlevel_status.setdefault(rl, {})
            runlevel_status[rl]['enabled'] = sysv_is_enabled(name, runlevel=rl)
    else:
        runlevel_status['enabled'] = sysv_is_enabled(name)
    is_started = False
    worked = False
    if pattern:
        worked = is_started = get_ps(module, pattern)
    else:
        if location.get('service'):
            cmd = '%s %s status' % (location['service'], name)
        elif script:
            cmd = '%s status' % script
        else:
            module.fail_json(msg='Unable to determine service status')
        (rc, out, err) = module.run_command(cmd)
        if not rc == -1:
            if name == 'iptables' and 'ACCEPT' in out:
                worked = True
                is_started = True
            if not worked and out.count('\n') <= 1:
                cleanout = out.lower().replace(name.lower(), '')
                for stopped in ['stop', 'is dead ', 'dead but ', 'could not access pid file', 'inactive']:
                    if stopped in cleanout:
                        worked = True
                        break
                if not worked:
                    for started_status in ['run', 'start', 'active']:
                        if started_status in cleanout and 'not ' not in cleanout:
                            is_started = True
                            worked = True
                            break
            if not worked and rc in [1, 2, 3, 4, 69]:
                worked = True
        if not worked:
            if rc == 0:
                is_started = True
                worked = True
            elif get_ps(module, name):
                is_started = True
                worked = True
                module.warn('Used ps output to match service name and determine it is up, this is very unreliable')
    if not worked:
        module.warn('Unable to determine if service is up, assuming it is down')
    result['status'].setdefault('enabled', {})
    result['status']['enabled']['changed'] = False
    result['status']['enabled']['rc'] = None
    result['status']['enabled']['stdout'] = None
    result['status']['enabled']['stderr'] = None
    if runlevels:
        result['status']['enabled']['runlevels'] = runlevels
        for rl in runlevels:
            if enabled != runlevel_status[rl]['enabled']:
                result['changed'] = True
                result['status']['enabled']['changed'] = True
        if not module.check_mode and result['changed']:
            if enabled:
                if location.get('update-rc.d'):
                    (rc, out, err) = module.run_command('%s %s enable %s' % (location['update-rc.d'], name, ' '.join(runlevels)))
                elif location.get('chkconfig'):
                    (rc, out, err) = module.run_command('%s --level %s %s on' % (location['chkconfig'], ''.join(runlevels), name))
            elif location.get('update-rc.d'):
                (rc, out, err) = module.run_command('%s %s disable %s' % (location['update-rc.d'], name, ' '.join(runlevels)))
            elif location.get('chkconfig'):
                (rc, out, err) = module.run_command('%s --level %s %s off' % (location['chkconfig'], ''.join(runlevels), name))
    else:
        if enabled is not None and enabled != runlevel_status['enabled']:
            result['changed'] = True
            result['status']['enabled']['changed'] = True
        if not module.check_mode and result['changed']:
            if enabled:
                if location.get('update-rc.d'):
                    (rc, out, err) = module.run_command('%s %s defaults' % (location['update-rc.d'], name))
                elif location.get('chkconfig'):
                    (rc, out, err) = module.run_command('%s %s on' % (location['chkconfig'], name))
            elif location.get('update-rc.d'):
                (rc, out, err) = module.run_command('%s %s disable' % (location['update-rc.d'], name))
            elif location.get('chkconfig'):
                (rc, out, err) = module.run_command('%s %s off' % (location['chkconfig'], name))
    if not module.check_mode and result['status']['enabled']['changed']:
        result['status']['enabled']['rc'] = rc
        result['status']['enabled']['stdout'] = out
        result['status']['enabled']['stderr'] = err
        (rc, out, err) = (None, None, None)
        if 'illegal runlevel specified' in result['status']['enabled']['stderr']:
            module.fail_json(msg='Illegal runlevel specified for enable operation on service %s' % name, **result)
    result['status'].setdefault(module.params['state'], {})
    result['status'][module.params['state']]['changed'] = False
    result['status'][module.params['state']]['rc'] = None
    result['status'][module.params['state']]['stdout'] = None
    result['status'][module.params['state']]['stderr'] = None
    if action:
        action = re.sub('p?ed$', '', action.lower())

        def runme(doit):
            if False:
                return 10
            args = module.params['arguments']
            cmd = '%s %s %s' % (script, doit, '' if args is None else args)
            if module.params['daemonize']:
                (rc, out, err) = daemonize(module, cmd)
            else:
                (rc, out, err) = module.run_command(cmd)
            if rc != 0:
                module.fail_json(msg='Failed to %s service: %s' % (action, name), rc=rc, stdout=out, stderr=err)
            return (rc, out, err)
        if action == 'restart':
            result['changed'] = True
            result['status'][module.params['state']]['changed'] = True
            if not module.check_mode:
                for dothis in ['stop', 'start']:
                    (rc, out, err) = runme(dothis)
                    if sleep_for:
                        sleep(sleep_for)
        elif is_started != (action == 'start'):
            result['changed'] = True
            result['status'][module.params['state']]['changed'] = True
            if not module.check_mode:
                (rc, out, err) = runme(action)
        elif is_started == (action == 'stop'):
            result['changed'] = True
            result['status'][module.params['state']]['changed'] = True
            if not module.check_mode:
                (rc, out, err) = runme(action)
        if not module.check_mode and result['status'][module.params['state']]['changed']:
            result['status'][module.params['state']]['rc'] = rc
            result['status'][module.params['state']]['stdout'] = out
            result['status'][module.params['state']]['stderr'] = err
            (rc, out, err) = (None, None, None)
    module.exit_json(**result)
if __name__ == '__main__':
    main()