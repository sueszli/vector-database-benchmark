from __future__ import annotations
from collections.abc import Mapping
from ansible.errors import AnsibleActionFail
from ansible.module_utils.six import string_types
from ansible.plugins.action import ActionBase
from ansible.parsing.utils.addresses import parse_address
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
display = Display()

class ActionModule(ActionBase):
    """ Create inventory hosts and groups in the memory inventory"""
    BYPASS_HOST_LOOP = True
    _requires_connection = False
    _supports_check_mode = True

    def run(self, tmp=None, task_vars=None):
        if False:
            while True:
                i = 10
        result = super(ActionModule, self).run(tmp, task_vars)
        del tmp
        args = self._task.args
        raw = args.pop('_raw_params', {})
        if isinstance(raw, Mapping):
            args = combine_vars(raw, args)
        else:
            raise AnsibleActionFail('Invalid raw parameters passed, requires a dictionary/mapping got a  %s' % type(raw))
        new_name = args.get('name', args.get('hostname', args.get('host', None)))
        if new_name is None:
            raise AnsibleActionFail('name, host or hostname needs to be provided')
        display.vv("creating host via 'add_host': hostname=%s" % new_name)
        try:
            (name, port) = parse_address(new_name, allow_ranges=False)
        except Exception:
            name = new_name
            port = None
        if port:
            args['ansible_ssh_port'] = port
        groups = args.get('groupname', args.get('groups', args.get('group', '')))
        new_groups = []
        if groups:
            if isinstance(groups, list):
                group_list = groups
            elif isinstance(groups, string_types):
                group_list = groups.split(',')
            else:
                raise AnsibleActionFail('Groups must be specified as a list.', obj=self._task)
            for group_name in group_list:
                if group_name not in new_groups:
                    new_groups.append(group_name.strip())
        host_vars = dict()
        special_args = frozenset(('name', 'hostname', 'groupname', 'groups'))
        for k in args.keys():
            if k not in special_args:
                host_vars[k] = args[k]
        result['changed'] = False
        result['add_host'] = dict(host_name=name, groups=new_groups, host_vars=host_vars)
        return result