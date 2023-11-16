"""
@author:       Nir Izraeli
@license:      GNU General Public License 2.0
@contact:      nirizr@gmail.com

This Address Space for Volatility is based on Nir's vmsnparser:
http://code.google.com/p/vmsnparser. It was converted by MHL. 
"""
import volatility.addrspace as addrspace
import volatility.obj as obj

class _VMWARE_HEADER(obj.CType):
    """A class for VMware VMSS/VMSN files"""

    @property
    def Version(self):
        if False:
            for i in range(10):
                print('nop')
        'The vmss/vmsn storage format version'
        return self.Magic & 15

class _VMWARE_GROUP(obj.CType):
    """A class for VMware Groups"""

    def _get_header(self):
        if False:
            return 10
        'Lookup the parent VMware header object'
        parent = self.obj_parent
        while parent.obj_name != '_VMWARE_HEADER':
            parent = parent.obj_parent
        return parent

    @property
    def Tags(self):
        if False:
            i = 10
            return i + 15
        'Generator for tags objects'
        tag = obj.Object('_VMWARE_TAG', offset=self.TagsOffset, vm=self.obj_vm, parent=self._get_header())
        while not (tag.Flags == 0 and tag.NameLength == 0):
            yield tag
            tag = obj.Object('_VMWARE_TAG', vm=self.obj_vm, parent=self._get_header(), offset=tag.RealDataOffset + tag.DataDiskSize)

class _VMWARE_TAG(obj.CType):
    """A class for VMware Tags"""

    def _size_type(self):
        if False:
            while True:
                i = 10
        "Depending on the version, the 'real' data size field is \n        either 4 or 8 bytes"
        if self.obj_parent.Version == 0:
            obj_type = 'unsigned int'
        else:
            obj_type = 'unsigned long long'
        return obj_type

    @property
    def OriginalDataOffset(self):
        if False:
            while True:
                i = 10
        "Determine the offset to this tag's data"
        return self.Name.obj_offset + self.NameLength + self.TagIndices.count * self.obj_vm.profile.get_obj_size('unsigned int')

    @property
    def RealDataOffset(self):
        if False:
            while True:
                i = 10
        "Determine the real offset to this tag's data"
        if self.OriginalDataSize in (62, 63):
            offset = self.OriginalDataOffset + self.obj_vm.profile.get_obj_size(self._size_type()) * 2
            padlen = obj.Object('unsigned short', offset=offset, vm=self.obj_vm)
            return offset + 2 + padlen
        else:
            return self.OriginalDataOffset

    @property
    def OriginalDataSize(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Flags & 63

    @property
    def DataDiskSize(self):
        if False:
            while True:
                i = 10
        "Get the tag's data size on disk"
        if self.OriginalDataSize in (62, 63):
            return obj.Object(self._size_type(), offset=self.OriginalDataOffset, vm=self.obj_vm)
        else:
            return self.OriginalDataSize

    @property
    def DataMemSize(self):
        if False:
            i = 10
            return i + 15
        "Get the tag's data size in memory"
        if self.OriginalDataSize in (62, 63):
            return obj.Object(self._size_type(), offset=self.OriginalDataOffset + self.obj_vm.profile.get_obj_size(self._size_type()), vm=self.obj_vm)
        else:
            return self.OriginalDataSize

    def cast_as(self, cast_type):
        if False:
            while True:
                i = 10
        'Cast the data in a tag as a specific type'
        return obj.Object(cast_type, offset=self.RealDataOffset, vm=self.obj_vm)

class VMwareVTypesModification(obj.ProfileModification):
    """Apply the necessary VTypes for parsing VMware headers"""

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update({'_VMWARE_HEADER': [12, {'Magic': [0, ['unsigned int']], 'GroupCount': [8, ['unsigned int']], 'Groups': [12, ['array', lambda x: x.GroupCount, ['_VMWARE_GROUP']]]}], '_VMWARE_GROUP': [80, {'Name': [0, ['String', dict(length=64, encoding='utf8')]], 'TagsOffset': [64, ['unsigned long long']]}], '_VMWARE_TAG': [None, {'Flags': [0, ['unsigned char']], 'NameLength': [1, ['unsigned char']], 'Name': [2, ['String', dict(length=lambda x: x.NameLength, encoding='utf8')]], 'TagIndices': [lambda x: x.obj_offset + 2 + x.NameLength, ['array', lambda x: x.Flags >> 6 & 3, ['unsigned int']]]}]})
        profile.object_classes.update({'_VMWARE_HEADER': _VMWARE_HEADER, '_VMWARE_GROUP': _VMWARE_GROUP, '_VMWARE_TAG': _VMWARE_TAG})

class VMWareAddressSpace(addrspace.AbstractRunBasedMemory):
    """ This AS supports VMware snapshot (VMSS) and saved state (VMSS) files """
    order = 30
    PAGE_SIZE = 4096

    def __init__(self, base, config, **kwargs):
        if False:
            while True:
                i = 10
        self.as_assert(base, 'No base Address Space')
        addrspace.BaseAddressSpace.__init__(self, base, config, **kwargs)
        self.runs = []
        self.header = obj.Object('_VMWARE_HEADER', offset=0, vm=base)
        self.as_assert(self.header.Magic in [3201482448, 3134307025, 3201482450, 3201547987], 'Invalid VMware signature: {0:#x}'.format(self.header.Magic))
        region_count = self.get_tag(self.header, grp_name='memory', tag_name='regionsCount', data_type='unsigned int')
        if not region_count.is_valid() or region_count == 0:
            memory_tag = self.get_tag(self.header, grp_name='memory', tag_name='Memory')
            self.as_assert(memory_tag != None, 'Cannot find the single-region Memory tag')
            self.runs.append((0, memory_tag.RealDataOffset, memory_tag.DataDiskSize))
        else:
            for i in range(region_count):
                memory_tag = self.get_tag(self.header, grp_name='memory', tag_name='Memory', indices=[0, 0])
                self.as_assert(memory_tag != None, 'Cannot find the Memory tag')
                memory_offset = self.get_tag(self.header, grp_name='memory', tag_name='regionPPN', indices=[i], data_type='unsigned int') * self.PAGE_SIZE
                file_offset = self.get_tag(self.header, grp_name='memory', tag_name='regionPageNum', indices=[i], data_type='unsigned int') * self.PAGE_SIZE + memory_tag.RealDataOffset
                length = self.get_tag(self.header, grp_name='memory', tag_name='regionSize', indices=[i], data_type='unsigned int') * self.PAGE_SIZE
                self.runs.append((memory_offset, file_offset, length))
        self.as_assert(len(self.runs) > 0, 'Cannot find any memory run information')

    @staticmethod
    def get_tag(header, grp_name, tag_name, indices=None, data_type=None):
        if False:
            while True:
                i = 10
        'Get a tag from the VMware headers\n        \n        @param grp_name: the group name (from _VMWARE_GROUP.Name)\n        \n        @param tag_name: the tag name (from _VMWARE_TAG.Name)\n        \n        @param indices: a group can contain multiple tags of the same name, \n        and tags can also contain meta-tags. this parameter lets you specify \n        which tag or meta-tag exactly to operate on. for example the 3rd CR \n        register (CR3) of the first CPU would use [0][3] indices. If this \n        parameter is None, then you just match on grp_name and tag_name. \n        \n        @param data_type: the type of data depends on the purpose of the tag. \n        If you supply this parameter, the function returns an object of the \n        specified type (for example an int or long). If not supplied, you just \n        get back the _VMWARE_TAG object itself. \n        '
        for group in header.Groups:
            if str(group.Name) != grp_name:
                continue
            for tag in group.Tags:
                if str(tag.Name) != tag_name:
                    continue
                if indices and tag.TagIndices != indices:
                    continue
                if data_type:
                    return tag.cast_as(data_type)
                else:
                    return tag
        return obj.NoneObject('Cannot find [{0}][{1}]'.format(grp_name, tag_name))