"""
@author:       Joe Sylve
@license:      GNU General Public License 2.0
@contact:      joe.sylve@gmail.com
@organization: Digital Forensics Solutions
"""
import volatility.plugins.linux.common as linux_common
from volatility.plugins.linux.slab_info import linux_slabinfo
import volatility.plugins.linux.pslist as linux_pslist

class linux_pslist_cache(linux_pslist.linux_pslist):
    """Gather tasks from the kmem_cache"""

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        linux_pslist.linux_pslist.__init__(self, config, *args, **kwargs)
        self._config.add_option('UNALLOCATED', short_option='u', default=False, help='Show unallocated', action='store_true')

    def calculate(self):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        pidlist = self._config.PID
        if pidlist:
            pidlist = [int(p) for p in self._config.PID.split(',')]
        cache = linux_slabinfo(self._config).get_kmem_cache('task_struct', self._config.UNALLOCATED)
        for task in cache:
            if not pidlist or task.pid in pidlist:
                yield task