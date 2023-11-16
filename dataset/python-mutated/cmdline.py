import volatility.plugins.taskmods as taskmods
from volatility.renderers import TreeGrid

class Cmdline(taskmods.DllList):
    """Display process command-line arguments"""

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        config.add_option('VERBOSE', short_option='v', default=False, cache_invalidator=False, help='Display full path of executable', action='store_true')

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Process', str), ('PID', int), ('CommandLine', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            cmdline = ''
            name = str(task.ImageFileName)
            try:
                if self._config.VERBOSE and task.SeAuditProcessCreationInfo.ImageFileName.Name != None:
                    name = str(task.SeAuditProcessCreationInfo.ImageFileName.Name)
            except AttributeError:
                pass
            if task.Peb:
                cmdline = '{0}'.format(str(task.Peb.ProcessParameters.CommandLine or '')).strip()
            yield (0, [name, int(task.UniqueProcessId), str(cmdline)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for task in data:
            pid = task.UniqueProcessId
            name = str(task.ImageFileName)
            try:
                if self._config.VERBOSE and task.SeAuditProcessCreationInfo.ImageFileName.Name != None:
                    name = str(task.SeAuditProcessCreationInfo.ImageFileName.Name)
            except AttributeError:
                pass
            outfd.write('*' * 72 + '\n')
            outfd.write('{0} pid: {1:6}\n'.format(name, pid))
            if task.Peb:
                outfd.write('Command line : {0}\n'.format(str(task.Peb.ProcessParameters.CommandLine or '')))