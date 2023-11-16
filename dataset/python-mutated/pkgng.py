from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = '\n---\nmodule: pkgng\nshort_description: Package manager for FreeBSD >= 9.0\ndescription:\n    - Manage binary packages for FreeBSD using \'pkgng\' which is available in versions after 9.0.\nversion_added: "1.2"\noptions:\n    name:\n        description:\n            - Name or list of names of packages to install/remove.\n        required: true\n    state:\n        description:\n            - State of the package.\n            - \'Note: "latest" added in 2.7\'\n        choices: [ \'present\', \'latest\', \'absent\' ]\n        required: false\n        default: present\n    cached:\n        description:\n            - Use local package base instead of fetching an updated one.\n        type: bool\n        required: false\n        default: no\n    annotation:\n        description:\n            - A comma-separated list of keyvalue-pairs of the form\n              C(<+/-/:><key>[=<value>]). A C(+) denotes adding an annotation, a\n              C(-) denotes removing an annotation, and C(:) denotes modifying an\n              annotation.\n              If setting or modifying annotations, a value must be provided.\n        required: false\n        version_added: "1.6"\n    pkgsite:\n        description:\n            - For pkgng versions before 1.1.4, specify packagesite to use\n              for downloading packages. If not specified, use settings from\n              C(/usr/local/etc/pkg.conf).\n            - For newer pkgng versions, specify a the name of a repository\n              configured in C(/usr/local/etc/pkg/repos).\n        required: false\n    rootdir:\n        description:\n            - For pkgng versions 1.5 and later, pkg will install all packages\n              within the specified root directory.\n            - Can not be used together with I(chroot) or I(jail) options.\n        required: false\n    chroot:\n        version_added: "2.1"\n        description:\n            - Pkg will chroot in the specified environment.\n            - Can not be used together with I(rootdir) or I(jail) options.\n        required: false\n    jail:\n        version_added: "2.4"\n        description:\n            - Pkg will execute in the given jail name or id.\n            - Can not be used together with I(chroot) or I(rootdir) options.\n    autoremove:\n        version_added: "2.2"\n        description:\n            - Remove automatically installed packages which are no longer needed.\n        required: false\n        type: bool\n        default: no\nauthor: "bleader (@bleader)"\nnotes:\n  - When using pkgsite, be careful that already in cache packages won\'t be downloaded again.\n  - When used with a `loop:` each package will be processed individually,\n    it is much more efficient to pass the list directly to the `name` option.\n'
EXAMPLES = '\n- name: Install package foo\n  pkgng:\n    name: foo\n    state: present\n\n- name: Annotate package foo and bar\n  pkgng:\n    name: foo,bar\n    annotation: \'+test1=baz,-test2,:test3=foobar\'\n\n- name: Remove packages foo and bar\n  pkgng:\n    name: foo,bar\n    state: absent\n\n# "latest" support added in 2.7\n- name: Upgrade package baz\n  pkgng:\n    name: baz\n    state: latest\n'
import re
from ansible.module_utils.basic import AnsibleModule

def query_package(module, pkgng_path, name, dir_arg):
    if False:
        while True:
            i = 10
    (rc, out, err) = module.run_command('%s %s info -g -e %s' % (pkgng_path, dir_arg, name))
    if rc == 0:
        return True
    return False

def query_update(module, pkgng_path, name, dir_arg, old_pkgng, pkgsite):
    if False:
        return 10
    if old_pkgng:
        (rc, out, err) = module.run_command('%s %s upgrade -g -n %s' % (pkgsite, pkgng_path, name))
    else:
        (rc, out, err) = module.run_command('%s %s upgrade %s -g -n %s' % (pkgng_path, dir_arg, pkgsite, name))
    if rc == 1:
        return True
    return False

def pkgng_older_than(module, pkgng_path, compare_version):
    if False:
        for i in range(10):
            print('nop')
    (rc, out, err) = module.run_command('%s -v' % pkgng_path)
    version = [int(x) for x in re.split('[\\._]', out)]
    i = 0
    new_pkgng = True
    while compare_version[i] == version[i]:
        i += 1
        if i == min(len(compare_version), len(version)):
            break
    else:
        if compare_version[i] > version[i]:
            new_pkgng = False
    return not new_pkgng

def remove_packages(module, pkgng_path, packages, dir_arg):
    if False:
        i = 10
        return i + 15
    remove_c = 0
    for package in packages:
        if not query_package(module, pkgng_path, package, dir_arg):
            continue
        if not module.check_mode:
            (rc, out, err) = module.run_command('%s %s delete -y %s' % (pkgng_path, dir_arg, package))
        if not module.check_mode and query_package(module, pkgng_path, package, dir_arg):
            module.fail_json(msg='failed to remove %s: %s' % (package, out))
        remove_c += 1
    if remove_c > 0:
        return (True, 'removed %s package(s)' % remove_c)
    return (False, 'package(s) already absent')

def install_packages(module, pkgng_path, packages, cached, pkgsite, dir_arg, state):
    if False:
        for i in range(10):
            print('nop')
    install_c = 0
    old_pkgng = pkgng_older_than(module, pkgng_path, [1, 1, 4])
    if pkgsite != '':
        if old_pkgng:
            pkgsite = 'PACKAGESITE=%s' % pkgsite
        else:
            pkgsite = '-r %s' % pkgsite
    batch_var = 'env BATCH=yes'
    if not module.check_mode and (not cached):
        if old_pkgng:
            (rc, out, err) = module.run_command('%s %s update' % (pkgsite, pkgng_path))
        else:
            (rc, out, err) = module.run_command('%s %s update' % (pkgng_path, dir_arg))
        if rc != 0:
            module.fail_json(msg='Could not update catalogue [%d]: %s %s' % (rc, out, err))
    for package in packages:
        already_installed = query_package(module, pkgng_path, package, dir_arg)
        if already_installed and state == 'present':
            continue
        update_available = query_update(module, pkgng_path, package, dir_arg, old_pkgng, pkgsite)
        if not update_available and already_installed and (state == 'latest'):
            continue
        if not module.check_mode:
            if already_installed:
                action = 'upgrade'
            else:
                action = 'install'
            if old_pkgng:
                (rc, out, err) = module.run_command('%s %s %s %s -g -U -y %s' % (batch_var, pkgsite, pkgng_path, action, package))
            else:
                (rc, out, err) = module.run_command('%s %s %s %s %s -g -U -y %s' % (batch_var, pkgng_path, dir_arg, action, pkgsite, package))
        if not module.check_mode and (not query_package(module, pkgng_path, package, dir_arg)):
            module.fail_json(msg='failed to %s %s: %s' % (action, package, out), stderr=err)
        install_c += 1
    if install_c > 0:
        return (True, 'added %s package(s)' % install_c)
    return (False, 'package(s) already %s' % state)

def annotation_query(module, pkgng_path, package, tag, dir_arg):
    if False:
        for i in range(10):
            print('nop')
    (rc, out, err) = module.run_command('%s %s info -g -A %s' % (pkgng_path, dir_arg, package))
    match = re.search('^\\s*(?P<tag>%s)\\s*:\\s*(?P<value>\\w+)' % tag, out, flags=re.MULTILINE)
    if match:
        return match.group('value')
    return False

def annotation_add(module, pkgng_path, package, tag, value, dir_arg):
    if False:
        return 10
    _value = annotation_query(module, pkgng_path, package, tag, dir_arg)
    if not _value:
        (rc, out, err) = module.run_command('%s %s annotate -y -A %s %s "%s"' % (pkgng_path, dir_arg, package, tag, value))
        if rc != 0:
            module.fail_json(msg='could not annotate %s: %s' % (package, out), stderr=err)
        return True
    elif _value != value:
        module.fail_json(mgs='failed to annotate %s, because %s is already set to %s, but should be set to %s' % (package, tag, _value, value))
        return False
    else:
        return False

def annotation_delete(module, pkgng_path, package, tag, value, dir_arg):
    if False:
        while True:
            i = 10
    _value = annotation_query(module, pkgng_path, package, tag, dir_arg)
    if _value:
        (rc, out, err) = module.run_command('%s %s annotate -y -D %s %s' % (pkgng_path, dir_arg, package, tag))
        if rc != 0:
            module.fail_json(msg='could not delete annotation to %s: %s' % (package, out), stderr=err)
        return True
    return False

def annotation_modify(module, pkgng_path, package, tag, value, dir_arg):
    if False:
        print('Hello World!')
    _value = annotation_query(module, pkgng_path, package, tag, dir_arg)
    if not value:
        module.fail_json(msg='could not change annotation to %s: tag %s does not exist' % (package, tag))
    elif _value == value:
        return False
    else:
        (rc, out, err) = module.run_command('%s %s annotate -y -M %s %s "%s"' % (pkgng_path, dir_arg, package, tag, value))
        if rc != 0:
            module.fail_json(msg='could not change annotation annotation to %s: %s' % (package, out), stderr=err)
        return True

def annotate_packages(module, pkgng_path, packages, annotation, dir_arg):
    if False:
        while True:
            i = 10
    annotate_c = 0
    annotations = map(lambda _annotation: re.match('(?P<operation>[\\+-:])(?P<tag>\\w+)(=(?P<value>\\w+))?', _annotation).groupdict(), re.split(',', annotation))
    operation = {'+': annotation_add, '-': annotation_delete, ':': annotation_modify}
    for package in packages:
        for _annotation in annotations:
            if operation[_annotation['operation']](module, pkgng_path, package, _annotation['tag'], _annotation['value']):
                annotate_c += 1
    if annotate_c > 0:
        return (True, 'added %s annotations.' % annotate_c)
    return (False, 'changed no annotations')

def autoremove_packages(module, pkgng_path, dir_arg):
    if False:
        i = 10
        return i + 15
    (rc, out, err) = module.run_command('%s %s autoremove -n' % (pkgng_path, dir_arg))
    autoremove_c = 0
    match = re.search('^Deinstallation has been requested for the following ([0-9]+) packages', out, re.MULTILINE)
    if match:
        autoremove_c = int(match.group(1))
    if autoremove_c == 0:
        return (False, 'no package(s) to autoremove')
    if not module.check_mode:
        (rc, out, err) = module.run_command('%s %s autoremove -y' % (pkgng_path, dir_arg))
    return (True, 'autoremoved %d package(s)' % autoremove_c)

def main():
    if False:
        return 10
    module = AnsibleModule(argument_spec=dict(state=dict(default='present', choices=['present', 'latest', 'absent'], required=False), name=dict(aliases=['pkg'], required=True, type='list'), cached=dict(default=False, type='bool'), annotation=dict(default='', required=False), pkgsite=dict(default='', required=False), rootdir=dict(default='', required=False, type='path'), chroot=dict(default='', required=False, type='path'), jail=dict(default='', required=False, type='str'), autoremove=dict(default=False, type='bool')), supports_check_mode=True, mutually_exclusive=[['rootdir', 'chroot', 'jail']])
    pkgng_path = module.get_bin_path('pkg', True)
    p = module.params
    pkgs = p['name']
    changed = False
    msgs = []
    dir_arg = ''
    if p['rootdir'] != '':
        old_pkgng = pkgng_older_than(module, pkgng_path, [1, 5, 0])
        if old_pkgng:
            module.fail_json(msg="To use option 'rootdir' pkg version must be 1.5 or greater")
        else:
            dir_arg = '--rootdir %s' % p['rootdir']
    if p['chroot'] != '':
        dir_arg = '--chroot %s' % p['chroot']
    if p['jail'] != '':
        dir_arg = '--jail %s' % p['jail']
    if p['state'] in ('present', 'latest'):
        (_changed, _msg) = install_packages(module, pkgng_path, pkgs, p['cached'], p['pkgsite'], dir_arg, p['state'])
        changed = changed or _changed
        msgs.append(_msg)
    elif p['state'] == 'absent':
        (_changed, _msg) = remove_packages(module, pkgng_path, pkgs, dir_arg)
        changed = changed or _changed
        msgs.append(_msg)
    if p['autoremove']:
        (_changed, _msg) = autoremove_packages(module, pkgng_path, dir_arg)
        changed = changed or _changed
        msgs.append(_msg)
    if p['annotation']:
        (_changed, _msg) = annotate_packages(module, pkgng_path, pkgs, p['annotation'], dir_arg)
        changed = changed or _changed
        msgs.append(_msg)
    module.exit_json(changed=changed, msg=', '.join(msgs))
if __name__ == '__main__':
    main()