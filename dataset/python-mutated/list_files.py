"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import os
import volatility.plugins.mac.common as common
import volatility.plugins.mac.mount as mac_mount
import volatility.obj as obj

class mac_list_files(common.AbstractMacCommand):
    """ Lists files in the file cache """

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        common.AbstractMacCommand.__init__(self, config, *args, **kwargs)
        self._config.add_option('SHOW_ORPHANS', short_option='s', default=False, help='Show orphans (vnodes without a parent)', action='store_true')

    @staticmethod
    def walk_vnodelist(listhead, loop_vnodes):
        if False:
            while True:
                i = 10
        seen = set()
        vnode = listhead.tqh_first.dereference()
        while vnode:
            if vnode in seen:
                break
            seen.add(vnode)
            loop_vnodes.add(vnode)
            vnode = vnode.v_mntvnodes.tqe_next.dereference()
        return loop_vnodes

    @staticmethod
    def list_files(config):
        if False:
            i = 10
            return i + 15
        plugin = mac_mount.mac_mount(config)
        mounts = plugin.calculate()
        vnodes = {}
        parent_vnodes = {}
        loop_vnodes = set()
        seen = set()
        for mount in mounts:
            loop_vnodes = mac_list_files.walk_vnodelist(mount.mnt_vnodelist, loop_vnodes)
            loop_vnodes = mac_list_files.walk_vnodelist(mount.mnt_workerqueue, loop_vnodes)
            loop_vnodes = mac_list_files.walk_vnodelist(mount.mnt_newvnodes, loop_vnodes)
            loop_vnodes.add(mount.mnt_vnodecovered)
            loop_vnodes.add(mount.mnt_realrootvp)
            loop_vnodes.add(mount.mnt_devvp)
        for vnode in loop_vnodes:
            while vnode:
                if vnode.obj_offset in vnodes:
                    break
                if int(vnode.v_flag) & 1:
                    name = vnode.full_path()
                    entry = [name, None, vnode]
                    vnodes[vnode.obj_offset] = entry
                else:
                    name = vnode.v_name.dereference()
                    parent = vnode.v_parent.dereference()
                    if parent:
                        par_offset = parent.obj_offset
                    elif config.SHOW_ORPHANS:
                        par_offset = None
                    else:
                        vnode = vnode.v_mntvnodes.tqe_next.dereference()
                        vnodes[vnode.obj_offset] = [None, None, vnode]
                        continue
                    entry = [name, par_offset, vnode]
                    vnodes[vnode.obj_offset] = entry
                vnode = vnode.v_mntvnodes.tqe_next.dereference()
        for (key, val) in vnodes.items():
            (name, parent, vnode) = val
            if not name or not parent:
                continue
            parent = obj.Object('vnode', offset=parent, vm=vnode.obj_vm)
            while parent:
                if parent.obj_offset in vnodes:
                    break
                name = parent.v_name.dereference()
                next_parent = parent.v_parent.dereference()
                if next_parent:
                    par_offset = next_parent.obj_offset
                else:
                    par_offset = None
                entry = [str(name), par_offset, parent]
                vnodes[parent.obj_offset] = entry
                parent = next_parent
        for (key, val) in vnodes.items():
            (name, parent, vnode) = val
            if not name:
                continue
            if not vnode.is_dir():
                continue
            name = str(name)
            if parent in parent_vnodes:
                full_path = parent_vnodes[parent] + '/' + name
            else:
                paths = [name]
                seen_subs = set()
                while parent and parent not in seen_subs:
                    seen_subs.add(parent)
                    entry = vnodes.get(parent)
                    if not entry:
                        break
                    (name, parent, _vnode) = entry
                    if not name:
                        break
                    paths.append(str(name))
                full_path = '/'.join(reversed(paths))
            parent_vnodes[key] = full_path
        for val in vnodes.values():
            (name, parent, vnode) = val
            if not name:
                continue
            name = str(name)
            entry = parent_vnodes.get(parent)
            if not entry:
                yield (vnode, name)
            else:
                full_path = entry + '/' + name
                if full_path[0] != '/':
                    full_path = '/' + full_path
                elif full_path[0:2] == '//':
                    full_path = full_path[1:]
                yield (vnode, full_path)

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        config = self._config
        for result in mac_list_files.list_files(config):
            yield result

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Offset (V)', '[addrpad]'), ('File Path', '')])
        for (vnode, path) in data:
            self.table_row(outfd, vnode.obj_offset, path)