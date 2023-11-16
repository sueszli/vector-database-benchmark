"""
@author:       Georg Wicherski
@license:      GNU General Public License 2.0
@contact:      georg@crowdstrike.com
@organization: CrowdStrike, Inc.
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist

class linux_plthook(linux_pslist.linux_pslist):
    """Scan ELF binaries' PLT for hooks to non-NEEDED images"""

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        linux_pslist.linux_pslist.__init__(self, config, *args, **kwargs)
        self._config.add_option('ALL', short_option='a', default=False, help='Display all PLT slots (incl. not hooked)', action='store_true')
        self._config.add_option('IGNORE', default=[], help='Ignore mappings backed by this path, ' + ' useful for bad -l compiles (i.e. apache2 modules)', action='append')

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        self.table_header(outfd, [('Task', '10'), ('ELF Start', '[addrpad]'), ('ELF Name', '24'), ('Symbol', '24'), ('Resolved Address', '[addrpad]'), ('H', '1'), ('Target Info', '')])
        ignore = frozenset(self._config.IGNORE)
        for task in data:
            for (soname, elf, elf_start, elf_end, addr, symbol_name, hookdesc, hooked) in task.plt_hook_info():
                if not hooked and (not self._config.ALL):
                    continue
                if hookdesc in ignore:
                    continue
                if hookdesc == '[RTLD_LAZY]' and (not self._config.ALL):
                    continue
                self.table_row(outfd, task.pid, elf_start, soname if soname else '[main]', symbol_name, addr, '!' if hooked else ' ', hookdesc)