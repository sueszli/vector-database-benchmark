"""
@author:       Jamie Levy (gleeda)
@license:      GNU General Public License 2.0
@contact:      jamie@memoryanalysis.net
@organization: Volatility Foundation
"""
import volatility.plugins.registry.registryapi as registryapi
from volatility.renderers import TreeGrid
import volatility.plugins.common as common
import volatility.addrspace as addrspace
import volatility.obj as obj
import volatility.debug as debug
import volatility.utils as utils
import datetime
import struct
fileitems = {'0': 'Product Name', '1': 'Company Name', '2': 'File version number only', '3': 'Language code', '4': 'SwitchBackContext', '5': 'File Version', '6': 'File Size', '7': 'SizeOfImage', '8': 'Hash of PE Header', '9': 'Checksum', 'a': 'UNKNOWN', 'b': 'UNKNOWN', 'c': 'File Description', 'd': 'UNKNOWN', 'f': 'CompileTime', '10': 'UNKNOWN', '11': 'LastModified', '12': 'Created', '15': 'Path', '16': 'UNKNOWN', '17': 'LastModified', '100': 'ProgramID', '101': 'SHA1 of file'}
programsitems = {'0': 'Program Name', '1': 'Program Version', '2': 'Publisher', '3': 'Languge Code', '4': 'UNKNOWN', '5': 'UNKNOWN', '6': 'Entry Type', '7': 'Registry Uninstall Key', '8': 'UNKNOWN', '9': 'UNKNOWN', 'a': 'Install Date', 'b': 'UNKNOWN', 'c': 'UNKNOWN', 'd': 'List of File Paths', 'f': 'Product Code', '10': 'Package Code', '11': 'MSI Product Code', '12': 'MSI Package Code', '13': 'UNKNOWN', 'Files': 'List of Files in this package'}

class AmCache(common.AbstractWindowsCommand):
    """Print AmCache information"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('HIVE-OFFSET', short_option='o', help='Hive offset (virtual)', type='int')
        self.regapi = None

    def calculate(self):
        if False:
            print('Hello World!')
        addr_space = utils.load_as(self._config)
        self.regapi = registryapi.RegistryApi(self._config)
        filekey = 'root\\file'
        progkey = 'root\\programs'
        if not self._config.HIVE_OFFSET:
            self.regapi.set_current('Amcache.hve')
        else:
            name = obj.Object('_CMHIVE', vm=addr_space, offset=self._config.HIVE_OFFSET).get_name()
            self.regapi.all_offsets[self._config.HIVE_OFFSET] = name
            self.regapi.current_offsets[self._config.HIVE_OFFSET] = name
        for (key, name) in self.regapi.reg_yield_key(None, filekey):
            for guidkey in self.regapi.reg_get_all_subkeys(None, None, given_root=key):
                result = {}
                for thefile in self.regapi.reg_get_all_subkeys(None, None, given_root=guidkey):
                    result['hive'] = name
                    for (vname, value) in self.regapi.reg_yield_values(None, None, thetype=None, given_root=thefile):
                        result['valuename'] = vname
                        result['value'] = value
                        result['key'] = thefile
                        result['description'] = fileitems.get(str(vname), 'UNKNOWN')
                        result['timestamp'] = ''
                        if str(vname) in ['11', '12', '17']:
                            try:
                                bufferas = addrspace.BufferAddressSpace(self._config, data=struct.pack('<Q', value))
                                result['timestamp'] = obj.Object('WinTimeStamp', vm=bufferas, offset=0, is_utc=True)
                            except struct.error:
                                result['timestamp'] = ''
                        yield result
        for (key, name) in self.regapi.reg_yield_key(None, progkey):
            for guidkey in self.regapi.reg_get_all_subkeys(None, None, given_root=key):
                result = {}
                result['hive'] = name
                for (vname, value) in self.regapi.reg_yield_values(None, None, thetype=None, given_root=guidkey):
                    result['valuename'] = vname
                    result['value'] = value
                    result['key'] = guidkey
                    result['description'] = programsitems.get(str(vname), 'UNKNOWN')
                    result['timestamp'] = ''
                    if str(vname) == 'a':
                        try:
                            bufferas = addrspace.BufferAddressSpace(self._config, data=struct.pack('<I', value))
                            result['timestamp'] = obj.Object('UnixTimeStamp', vm=bufferas, offset=0, is_utc=True)
                        except struct.error:
                            pass
                    yield result

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Registry', str), ('KeyPath', str), ('LastWrite', str), ('ValueName', str), ('Description', str), ('Value', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for result in data:
            if result['key']:
                yield (0, [str(result['hive']), str(self.regapi.reg_get_key_path(result['key'])), str(result['key'].LastWriteTime), str(result['valuename']), str(result['description']), str(result['timestamp'] if result['timestamp'] else result['value'])])

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        keyfound = False
        for result in data:
            if result['key']:
                keyfound = True
                outfd.write('Registry: {0}\n'.format(result['hive']))
                outfd.write('Key Path: {0}\n'.format(self.regapi.reg_get_key_path(result['key'])))
                outfd.write('Key Last updated: {0}\n'.format(result['key'].LastWriteTime))
                outfd.write('Value Name: {0}\n'.format(result['valuename']))
                outfd.write('Description: {0}\n'.format(result['description']))
                outfd.write('Value: {0}\n\n'.format(result['timestamp'] if result['timestamp'] else result['value']))
        if not keyfound:
            outfd.write('The requested key could not be found in the hive(s) searched\n')