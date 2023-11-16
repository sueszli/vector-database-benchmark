"""
@author:       Jamie Levy (gleeda)
@license:      GNU General Public License 2.0
@contact:      jamie@memoryanalysis.net
@organization: Volatility Foundation
"""
import volatility.plugins.taskmods as taskmods
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class JobLinks(taskmods.DllList):
    """ Print process job link information"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        config.add_option('PHYSICAL-OFFSET', short_option='P', default=False, cache_invalidator=False, help='Display physical offsets instead of virtual', action='store_true')

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        return TreeGrid([('Offset{0}'.format(offsettype), Address), ('Name', str), ('PID', int), ('PPID', int), ('Sess', int), ('JobSess', int), ('Wow64', int), ('Total', int), ('Active', int), ('Term', int), ('Joblink', str), ('Process', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            job = task.Job.dereference()
            if job:
                if not self._config.PHYSICAL_OFFSET:
                    offset = task.obj_offset
                else:
                    offset = task.obj_vm.vtop(task.obj_offset)
                yield (0, [Address(offset), str(task.ImageFileName), int(task.UniqueProcessId), int(task.InheritedFromUniqueProcessId), int(task.SessionId), int(job.SessionId), int(task.IsWow64), int(job.TotalProcesses), int(job.ActiveProcesses), int(job.TotalTerminatedProcesses), '-', '(Original Process)'])
                for item in job.ProcessListHead.list_of_type('_EPROCESS', 'JobLinks'):
                    if not self._config.PHYSICAL_OFFSET:
                        offset = item.obj_offset
                    else:
                        offset = item.obj_vm.vtop(item.obj_offset)
                    path = str(item.ImageFileName)
                    if item.Peb:
                        path = str(item.Peb.ProcessParameters.ImagePathName.v().encode('utf8', 'ignore'))
                    yield (0, [Address(offset), str(item.ImageFileName), int(item.UniqueProcessId), int(item.InheritedFromUniqueProcessId), int(item.SessionId), 0, int(item.IsWow64), 0, 0, 0, 'Yes', path])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        header = '*' * 107
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        self.table_header(outfd, [('Offset{0}'.format(offsettype), '[addrpad]'), ('Name', '20s'), ('PID', '>6'), ('PPID', '>6'), ('Sess', '>6'), ('JobSess', '>7'), ('Wow64', '>6'), ('Total', '>6'), ('Active', '>6'), ('Term', '>6'), ('JobLink', '>8'), ('Process', '')])
        for task in data:
            job = task.Job.dereference()
            if job:
                if not self._config.PHYSICAL_OFFSET:
                    offset = task.obj_offset
                else:
                    offset = task.obj_vm.vtop(task.obj_offset)
                self.table_row(outfd, offset, task.ImageFileName, task.UniqueProcessId, task.InheritedFromUniqueProcessId, task.SessionId, job.SessionId, task.IsWow64, job.TotalProcesses, job.ActiveProcesses, job.TotalTerminatedProcesses, '-', '(Original Process)')
                for item in job.ProcessListHead.list_of_type('_EPROCESS', 'JobLinks'):
                    if not self._config.PHYSICAL_OFFSET:
                        offset = item.obj_offset
                    else:
                        offset = item.obj_vm.vtop(item.obj_offset)
                    self.table_row(outfd, offset, item.ImageFileName, item.UniqueProcessId, item.InheritedFromUniqueProcessId, item.SessionId, '-', item.IsWow64, '-', '-', '-', 'Yes', item.Peb.ProcessParameters.ImagePathName.v().encode('utf8', 'ignore'))
                outfd.write('{0}\n'.format(header))