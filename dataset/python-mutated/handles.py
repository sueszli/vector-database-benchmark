import volatility.plugins.taskmods as taskmods
from volatility import renderers
from volatility.renderers.basic import Address, Hex

class Handles(taskmods.DllList):
    """Print list of open handles for each process"""

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        taskmods.DllList.__init__(self, config, *args, **kwargs)
        config.add_option('PHYSICAL-OFFSET', short_option='P', default=False, help='Physical Offset', action='store_true')
        config.add_option('OBJECT-TYPE', short_option='t', default=None, help='Show these object types (comma-separated)', action='store', type='str')
        config.add_option('SILENT', short_option='s', default=False, action='store_true', help='Suppress less meaningful results')

    def generator(self, data):
        if False:
            print('Hello World!')
        if self._config.OBJECT_TYPE:
            object_list = [s.lower() for s in self._config.OBJECT_TYPE.split(',')]
        else:
            object_list = []
        for (pid, handle, object_type, name) in data:
            if object_list and object_type.lower() not in object_list:
                continue
            if self._config.SILENT:
                if len(name.replace("'", '')) == 0:
                    continue
            if not self._config.PHYSICAL_OFFSET:
                offset = handle.Body.obj_offset
            else:
                offset = handle.obj_vm.vtop(handle.Body.obj_offset)
            yield (0, [Address(offset), int(pid), Hex(handle.HandleValue), Hex(handle.GrantedAccess), str(object_type), str(name)])

    def unified_output(self, data):
        if False:
            print('Hello World!')
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        tg = renderers.TreeGrid([('Offset{0}'.format(offsettype), Address), ('Pid', int), ('Handle', Hex), ('Access', Hex), ('Type', str), ('Details', str)], self.generator(data))
        return tg

    def render_text(self, outfd, data):
        if False:
            return 10
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        self.table_header(outfd, [('Offset{0}'.format(offsettype), '[addrpad]'), ('Pid', '>6'), ('Handle', '[addr]'), ('Access', '[addr]'), ('Type', '16'), ('Details', '')])
        if self._config.OBJECT_TYPE:
            object_list = [s.lower() for s in self._config.OBJECT_TYPE.split(',')]
        else:
            object_list = []
        for (pid, handle, object_type, name) in data:
            if object_list and object_type.lower() not in object_list:
                continue
            if self._config.SILENT:
                if len(name.replace("'", '')) == 0:
                    continue
            if not self._config.PHYSICAL_OFFSET:
                offset = handle.Body.obj_offset
            else:
                offset = handle.obj_vm.vtop(handle.Body.obj_offset)
            self.table_row(outfd, offset, pid, handle.HandleValue, handle.GrantedAccess, object_type, name)

    def calculate(self):
        if False:
            print('Hello World!')
        for task in taskmods.DllList.calculate(self):
            pid = task.UniqueProcessId
            if task.ObjectTable.HandleTableList:
                for handle in task.ObjectTable.handles():
                    if not handle.is_valid():
                        continue
                    name = ''
                    object_type = handle.get_object_type()
                    if object_type == 'File':
                        file_obj = handle.dereference_as('_FILE_OBJECT')
                        name = str(file_obj.file_name_with_device())
                    elif object_type == 'Key':
                        key_obj = handle.dereference_as('_CM_KEY_BODY')
                        name = key_obj.full_key_name()
                    elif object_type == 'Process':
                        proc_obj = handle.dereference_as('_EPROCESS')
                        name = '{0}({1})'.format(proc_obj.ImageFileName, proc_obj.UniqueProcessId)
                    elif object_type == 'Thread':
                        thrd_obj = handle.dereference_as('_ETHREAD')
                        name = 'TID {0} PID {1}'.format(thrd_obj.Cid.UniqueThread, thrd_obj.Cid.UniqueProcess)
                    elif handle.NameInfo.Name == None:
                        name = ''
                    else:
                        name = str(handle.NameInfo.Name)
                    yield (pid, handle, object_type, name)