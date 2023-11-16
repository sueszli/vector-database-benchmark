from volatility import renderers
import volatility.plugins.common as common
from volatility.renderers.basic import Hex, Address
import volatility.utils as utils
import volatility.poolscan as poolscan
import volatility.obj as obj

class ObjectTypeScanner(poolscan.PoolScanner):
    """Pool scanner for object type objects"""

    def __init__(self, address_space, **kwargs):
        if False:
            while True:
                i = 10
        poolscan.PoolScanner.__init__(self, address_space, **kwargs)
        self.struct_name = '_OBJECT_TYPE'
        self.object_type = 'Type'
        self.pooltag = obj.VolMagic(address_space).ObjectTypePoolTag.v()
        size = 200
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= size)), ('CheckPoolType', dict(paged=False, non_paged=True, free=True))]

class ObjectTypeKeyModification(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.merge_overlay({'_OBJECT_TYPE': [None, {'Key': [None, ['String', dict(length=4)]]}]})

class ObjTypeScan(common.AbstractScanCommand):
    """Scan for Windows object type objects"""
    scanners = [ObjectTypeScanner]

    def unified_output(self, data):
        if False:
            return 10

        def generator(data):
            if False:
                return 10
            for object_type in data:
                yield (0, [Address(object_type.obj_offset), Hex(object_type.TotalNumberOfObjects), Hex(object_type.TotalNumberOfHandles), str(object_type.Key), str(object_type.Name or ''), str(object_type.TypeInfo.PoolType)])
        return renderers.TreeGrid([('Offset', Address), ('nObjects', Hex), ('nHandles', Hex), ('Key', str), ('Name', str), ('PoolType', str)], generator(data))

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Offset', '[addrpad]'), ('nObjects', '[addr]'), ('nHandles', '[addr]'), ('Key', '8'), ('Name', '30'), ('PoolType', '20')])
        for object_type in data:
            self.table_row(outfd, object_type.obj_offset, object_type.TotalNumberOfObjects, object_type.TotalNumberOfHandles, str(object_type.Key), str(object_type.Name or ''), object_type.TypeInfo.PoolType)