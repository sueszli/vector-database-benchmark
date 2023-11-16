from __future__ import absolute_import
from oslo_config import cfg
from st2common import config
from st2common.services import coordination
'\nTool which lists all the services registered in the service registry and their capabilities.\n'

def main(group_id=None):
    if False:
        while True:
            i = 10
    coordinator = coordination.get_coordinator()
    if not group_id:
        group_ids = list(coordinator.get_groups().get())
        group_ids = [item.decode('utf-8') for item in group_ids]
        print('Available groups (%s):' % len(group_ids))
        for group_id in group_ids:
            print(' - %s' % group_id)
        print('')
    else:
        group_ids = [group_id]
    for group_id in group_ids:
        member_ids = list(coordinator.get_members(group_id).get())
        member_ids = [member_id.decode('utf-8') for member_id in member_ids]
        print('Members in group "%s" (%s):' % (group_id, len(member_ids)))
        for member_id in member_ids:
            capabilities = coordinator.get_member_capabilities(group_id, member_id).get()
            print(' - %s (capabilities=%s)' % (member_id, str(capabilities)))

def do_register_cli_opts(opts, ignore_errors=False):
    if False:
        for i in range(10):
            print('nop')
    for opt in opts:
        try:
            cfg.CONF.register_cli_opt(opt)
        except:
            if not ignore_errors:
                raise
if __name__ == '__main__':
    cli_opts = [cfg.StrOpt('group-id', default=None, help='If provided, only list members for that group.')]
    do_register_cli_opts(cli_opts)
    config.parse_args()
    main(group_id=cfg.CONF.group_id)