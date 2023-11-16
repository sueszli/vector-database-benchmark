"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import os.path
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.proc_maps as linux_proc_maps

class linux_dump_map(linux_proc_maps.linux_proc_maps):
    """ Writes selected memory mappings to disk """

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        linux_proc_maps.linux_proc_maps.__init__(self, config, *args, **kwargs)
        self._config.add_option('VMA', short_option='s', default=None, help='Filter by VMA starting address', action='store', type='long')
        self._config.add_option('DUMP-DIR', short_option='D', default=None, help='Output directory', action='store', type='str')

    def read_addr_range(self, task, start, end):
        if False:
            for i in range(10):
                print('nop')
        pagesize = 4096
        proc_as = task.get_process_address_space()
        while start < end:
            page = proc_as.zread(start, pagesize)
            yield page
            start = start + pagesize

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        if not self._config.DUMP_DIR or not os.path.isdir(self._config.DUMP_DIR):
            debug.error('Please specify an existing output dir (--dump-dir)')
        self.table_header(outfd, [('Task', '10'), ('VM Start', '[addrpad]'), ('VM End', '[addrpad]'), ('Length', '[addr]'), ('Path', '')])
        for (task, vma) in data:
            if not self._config.VMA or vma.vm_start == self._config.VMA:
                file_name = 'task.{0}.{1:#x}.vma'.format(task.pid, vma.vm_start)
                file_path = os.path.join(self._config.DUMP_DIR, file_name)
                outfile = open(file_path, 'wb+')
                for page in self.read_addr_range(task, vma.vm_start, vma.vm_end):
                    outfile.write(page)
                outfile.close()
                self.table_row(outfd, task.pid, vma.vm_start, vma.vm_end, vma.vm_end - vma.vm_start, file_path)