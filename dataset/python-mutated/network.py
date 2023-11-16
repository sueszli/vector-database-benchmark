"""
@author:       AAron Walters
@license:      GNU General Public License 2.0
@contact:      awalters@4tphi.net
@organization: Volatility Foundation
"""
import volatility.win32 as win32
import volatility.obj as obj
module_versions_xp = {'MP': {'TCBTableOff': [301032], 'SizeOff': [260040], 'AddrObjTableOffset': [296800], 'AddrObjTableSizeOffset': [296804]}, 'UP': {'TCBTableOff': [300520], 'SizeOff': [259516], 'AddrObjTableOffset': [296288], 'AddrObjTableSizeOffset': [296292]}, '2180': {'TCBTableOff': [300008], 'SizeOff': [258992], 'AddrObjTableOffset': [295776], 'AddrObjTableSizeOffset': [295780]}, '3244': {'TCBTableOff': [300776], 'SizeOff': [259772], 'AddrObjTableOffset': [296544], 'AddrObjTableSizeOffset': [296548]}, '3394': {'TCBTableOff': [300904], 'SizeOff': [259900], 'AddrObjTableOffset': [296672], 'AddrObjTableSizeOffset': [296676]}, '5625': {'TCBTableOff': [301800], 'SizeOff': [260808], 'AddrObjTableOffset': [297568], 'AddrObjTableSizeOffset': [297572]}, '2111': {'TCBTableOff': [301672], 'SizeOff': [260680], 'AddrObjTableOffset': [297440], 'AddrObjTableSizeOffset': [297444]}}
module_versions_2003 = {'3790': {'TCBTableOff': [313032], 'SizeOff': [274732], 'AddrObjTableOffset': [310176], 'AddrObjTableSizeOffset': [310180]}, '1830': {'TCBTableOff': [320552], 'SizeOff': [278848], 'AddrObjTableOffset': [316644], 'AddrObjTableSizeOffset': [316648]}, '3959': {'TCBTableOff': [509256], 'SizeOff': [328456], 'AddrObjTableOffset': [372132], 'AddrObjTableSizeOffset': [372136]}, '4573': {'TCBTableOff': [520364], 'SizeOff': [336680], 'AddrObjTableOffset': [380676], 'AddrObjTableSizeOffset': [380680]}, '3959_x64': {'TCBTableOff': [822576], 'SizeOff': [636064], 'AddrObjTableOffset': [673920], 'AddrObjTableSizeOffset': [673928]}, '1830_x64': {'TCBTableOff': [586448], 'SizeOff': [549324], 'AddrObjTableOffset': [574656], 'AddrObjTableSizeOffset': [574664]}, 'unk_1_x64': {'TCBTableOff': [840408], 'SizeOff': [648352], 'AddrObjTableOffset': [686304], 'AddrObjTableSizeOffset': [686312]}}
MAX_SOCKETS = 2000000

def determine_connections(addr_space):
    if False:
        return 10
    'Determines all connections for each module'
    all_modules = win32.modules.lsmod(addr_space)
    version = (addr_space.profile.metadata.get('major', 0), addr_space.profile.metadata.get('minor', 0))
    if version <= (5, 1):
        module_versions = module_versions_xp
    else:
        module_versions = module_versions_2003
    for m in all_modules:
        if str(m.BaseDllName).lower() == 'tcpip.sys':
            for attempt in module_versions:
                table_size = obj.Object('long', offset=m.DllBase + module_versions[attempt]['SizeOff'][0], vm=addr_space)
                table_addr = obj.Object('address', offset=m.DllBase + module_versions[attempt]['TCBTableOff'][0], vm=addr_space)
                if table_size > 0:
                    table = obj.Object('Array', offset=table_addr, vm=addr_space, count=table_size, target=obj.Curry(obj.Pointer, '_TCPT_OBJECT'))
                    if table:
                        for entry in table:
                            conn = entry.dereference()
                            seen = set()
                            while conn.is_valid() and conn.obj_offset not in seen:
                                yield conn
                                seen.add(conn.obj_offset)
                                conn = conn.Next.dereference()

def determine_sockets(addr_space):
    if False:
        return 10
    'Determines all sockets for each module'
    all_modules = win32.modules.lsmod(addr_space)
    if addr_space.profile.metadata.get('major', 0) <= 5.1 and addr_space.profile.metadata.get('minor', 0) == 1:
        module_versions = module_versions_xp
    else:
        module_versions = module_versions_2003
    for m in all_modules:
        if str(m.BaseDllName).lower() == 'tcpip.sys':
            for attempt in module_versions:
                table_size = obj.Object('unsigned long', offset=m.DllBase + module_versions[attempt]['AddrObjTableSizeOffset'][0], vm=addr_space)
                table_addr = obj.Object('address', offset=m.DllBase + module_versions[attempt]['AddrObjTableOffset'][0], vm=addr_space)
                if int(table_size) > 0 and int(table_size) < MAX_SOCKETS:
                    table = obj.Object('Array', offset=table_addr, vm=addr_space, count=table_size, target=obj.Curry(obj.Pointer, '_ADDRESS_OBJECT'))
                    if table:
                        for entry in table:
                            sock = entry.dereference()
                            seen = set()
                            while sock.is_valid() and sock.obj_offset not in seen:
                                yield sock
                                seen.add(sock.obj_offset)
                                sock = sock.Next.dereference()