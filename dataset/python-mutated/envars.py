import volatility.plugins.taskmods as taskmods
import volatility.plugins.registry.registryapi as registryapi
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class Envars(taskmods.DllList):
    """Display process environment variables"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        config.add_option('SILENT', short_option='s', default=False, help='Suppress common and non-persistent variables', action='store_true')

    def _get_silent_vars(self):
        if False:
            i = 10
            return i + 15
        'Enumerate persistent & common variables.\n        \n        This function collects the global (all users) and \n        user-specific environment variables from the \n        registry. Any variables in a process env block that\n        does not exist in the persistent list was explicitly\n        set with the SetEnvironmentVariable() API.\n        '
        values = []
        regapi = registryapi.RegistryApi(self._config)
        ccs = regapi.reg_get_currentcontrolset()
        for (value, _) in regapi.reg_yield_values(hive_name='system', key='{0}\\Control\\Session Manager\\Environment'.format(ccs)):
            values.append(value)
        regapi.reset_current()
        for (value, _) in regapi.reg_yield_values(hive_name='ntuser.dat', key='Environment'):
            values.append(value)
        for (value, _) in regapi.reg_yield_values(hive_name='ntuser.dat', key='Volatile Environment'):
            values.append(value)
        values.extend(['ProgramFiles', 'CommonProgramFiles', 'SystemDrive', 'SystemRoot', 'ProgramData', 'PUBLIC', 'ALLUSERSPROFILE', 'COMPUTERNAME', 'SESSIONNAME', 'USERNAME', 'USERPROFILE', 'PROMPT', 'USERDOMAIN', 'AppData', 'CommonFiles', 'CommonDesktop', 'CommonProgramGroups', 'CommonStartMenu', 'CommonStartUp', 'Cookies', 'DesktopDirectory', 'Favorites', 'History', 'NetHood', 'PersonalDocuments', 'RecycleBin', 'StartMenu', 'Templates', 'AltStartup', 'CommonFavorites', 'ConnectionWizard', 'DocAndSettingRoot', 'InternetCache', 'windir', 'Path', 'HOMEDRIVE', 'PROCESSOR_ARCHITECTURE', 'NUMBER_OF_PROCESSORS', 'ProgramFiles(x86)', 'CommonProgramFiles(x86)', 'CommonProgramW6432', 'PSModulePath', 'PROCESSOR_IDENTIFIER', 'FP_NO_HOST_CHECK', 'LOCALAPPDATA', 'TMP', 'ProgramW6432'])
        return values

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Pid', int), ('Process', str), ('Block', Address), ('Variable', str), ('Value', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        if self._config.SILENT:
            silent_vars = self._get_silent_vars()
        for task in data:
            for (var, val) in task.environment_variables():
                if self._config.SILENT:
                    if var in silent_vars:
                        continue
                yield (0, [int(task.UniqueProcessId), str(task.ImageFileName), Address(task.Peb.ProcessParameters.Environment), str(var), str(val)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Pid', '8'), ('Process', '20'), ('Block', '[addrpad]'), ('Variable', '30'), ('Value', '')])
        if self._config.SILENT:
            silent_vars = self._get_silent_vars()
        for task in data:
            for (var, val) in task.environment_variables():
                if self._config.SILENT:
                    if var in silent_vars:
                        continue
                self.table_row(outfd, task.UniqueProcessId, task.ImageFileName, task.Peb.ProcessParameters.Environment, var, val)