from struct import unpack, pack
from impacket.structure import Structure
RECALC_ACE_SIZE = True

class LDAP_SID_IDENTIFIER_AUTHORITY(Structure):
    structure = (('Value', '6s'),)

class LDAP_SID(Structure):
    structure = (('Revision', '<B'), ('SubAuthorityCount', '<B'), ('IdentifierAuthority', ':', LDAP_SID_IDENTIFIER_AUTHORITY), ('SubLen', '_-SubAuthority', 'self["SubAuthorityCount"]*4'), ('SubAuthority', ':'))

    def formatCanonical(self):
        if False:
            while True:
                i = 10
        ans = 'S-%d-%d' % (self['Revision'], ord(self['IdentifierAuthority']['Value'][5:6]))
        for i in range(self['SubAuthorityCount']):
            ans += '-%d' % unpack('<L', self['SubAuthority'][i * 4:i * 4 + 4])[0]
        return ans

    def fromCanonical(self, canonical):
        if False:
            while True:
                i = 10
        items = canonical.split('-')
        self['Revision'] = int(items[1])
        self['IdentifierAuthority'] = LDAP_SID_IDENTIFIER_AUTHORITY()
        self['IdentifierAuthority']['Value'] = b'\x00\x00\x00\x00\x00' + pack('B', int(items[2]))
        self['SubAuthorityCount'] = len(items) - 3
        self['SubAuthority'] = b''
        for i in range(self['SubAuthorityCount']):
            self['SubAuthority'] += pack('<L', int(items[i + 3]))
'\nSelf-relative security descriptor as described in 2.4.6\nhttps://msdn.microsoft.com/en-us/library/cc230366.aspx\n'

class SR_SECURITY_DESCRIPTOR(Structure):
    structure = (('Revision', 'c'), ('Sbz1', 'c'), ('Control', '<H'), ('OffsetOwner', '<L'), ('OffsetGroup', '<L'), ('OffsetSacl', '<L'), ('OffsetDacl', '<L'), ('Sacl', ':'), ('Dacl', ':'), ('OwnerSid', ':'), ('GroupSid', ':'))

    def fromString(self, data):
        if False:
            print('Hello World!')
        Structure.fromString(self, data)
        if self['OffsetOwner'] != 0:
            self['OwnerSid'] = LDAP_SID(data=data[self['OffsetOwner']:])
        else:
            self['OwnerSid'] = b''
        if self['OffsetGroup'] != 0:
            self['GroupSid'] = LDAP_SID(data=data[self['OffsetGroup']:])
        else:
            self['GroupSid'] = b''
        if self['OffsetSacl'] != 0:
            self['Sacl'] = ACL(data=data[self['OffsetSacl']:])
        else:
            self['Sacl'] = b''
        if self['OffsetDacl'] != 0:
            self['Dacl'] = ACL(data=data[self['OffsetDacl']:])
        else:
            self['Sacl'] = b''

    def getData(self):
        if False:
            print('Hello World!')
        headerlen = 20
        datalen = 0
        if self['Sacl'] != b'':
            self['OffsetSacl'] = headerlen + datalen
            datalen += len(self['Sacl'].getData())
        else:
            self['OffsetSacl'] = 0
        if self['Dacl'] != b'':
            self['OffsetDacl'] = headerlen + datalen
            datalen += len(self['Dacl'].getData())
        else:
            self['OffsetDacl'] = 0
        if self['OwnerSid'] != b'':
            self['OffsetOwner'] = headerlen + datalen
            datalen += len(self['OwnerSid'].getData())
        else:
            self['OffsetOwner'] = 0
        if self['GroupSid'] != b'':
            self['OffsetGroup'] = headerlen + datalen
            datalen += len(self['GroupSid'].getData())
        else:
            self['OffsetGroup'] = 0
        return Structure.getData(self)
'\nACE as described in 2.4.4\nhttps://msdn.microsoft.com/en-us/library/cc230295.aspx\n'

class ACE(Structure):
    CONTAINER_INHERIT_ACE = 2
    FAILED_ACCESS_ACE_FLAG = 128
    INHERIT_ONLY_ACE = 8
    INHERITED_ACE = 16
    NO_PROPAGATE_INHERIT_ACE = 4
    OBJECT_INHERIT_ACE = 1
    SUCCESSFUL_ACCESS_ACE_FLAG = 64
    structure = (('AceType', 'B'), ('AceFlags', 'B'), ('AceSize', '<H'), ('AceLen', '_-Ace', 'self["AceSize"]-4'), ('Ace', ':'))

    def fromString(self, data):
        if False:
            print('Hello World!')
        Structure.fromString(self, data)
        self['TypeName'] = ACE_TYPE_MAP[self['AceType']].__name__
        self['Ace'] = ACE_TYPE_MAP[self['AceType']](data=self['Ace'])

    def getData(self):
        if False:
            return 10
        if RECALC_ACE_SIZE or 'AceSize' not in self.fields:
            self['AceSize'] = len(self['Ace'].getData()) + 4
        if self['AceSize'] % 4 != 0:
            self['AceSize'] += self['AceSize'] % 4
        data = Structure.getData(self)
        if len(data) < self['AceSize']:
            data += '\x00' * (self['AceSize'] - len(data))
        return data

    def hasFlag(self, flag):
        if False:
            print('Hello World!')
        return self['AceFlags'] & flag == flag
'\nACCESS_MASK as described in 2.4.3\nhttps://msdn.microsoft.com/en-us/library/cc230294.aspx\n'

class ACCESS_MASK(Structure):
    GENERIC_READ = 2147483648
    GENERIC_WRITE = 1073741824
    GENERIC_EXECUTE = 536870912
    GENERIC_ALL = 268435456
    MAXIMUM_ALLOWED = 33554432
    ACCESS_SYSTEM_SECURITY = 16777216
    SYNCHRONIZE = 1048576
    WRITE_OWNER = 524288
    WRITE_DACL = 262144
    READ_CONTROL = 131072
    DELETE = 65536
    structure = (('Mask', '<L'),)

    def hasPriv(self, priv):
        if False:
            i = 10
            return i + 15
        return self['Mask'] & priv == priv

    def setPriv(self, priv):
        if False:
            for i in range(10):
                print('nop')
        self['Mask'] |= priv

    def removePriv(self, priv):
        if False:
            while True:
                i = 10
        self['Mask'] ^= priv
'\nACCESS_ALLOWED_ACE as described in 2.4.4.2\nhttps://msdn.microsoft.com/en-us/library/cc230286.aspx\n'

class ACCESS_ALLOWED_ACE(Structure):
    ACE_TYPE = 0
    structure = (('Mask', ':', ACCESS_MASK), ('Sid', ':', LDAP_SID))
'\nACCESS_ALLOWED_OBJECT_ACE as described in 2.4.4.3\nhttps://msdn.microsoft.com/en-us/library/cc230289.aspx\n'

class ACCESS_ALLOWED_OBJECT_ACE(Structure):
    ACE_TYPE = 5
    ACE_OBJECT_TYPE_PRESENT = 1
    ACE_INHERITED_OBJECT_TYPE_PRESENT = 2
    ADS_RIGHT_DS_CONTROL_ACCESS = 256
    ADS_RIGHT_DS_CREATE_CHILD = 1
    ADS_RIGHT_DS_DELETE_CHILD = 2
    ADS_RIGHT_DS_READ_PROP = 16
    ADS_RIGHT_DS_WRITE_PROP = 32
    ADS_RIGHT_DS_SELF = 8
    structure = (('Mask', ':', ACCESS_MASK), ('Flags', '<L'), ('ObjectTypeLen', '_-ObjectType', 'self.checkObjectType(self["Flags"])'), ('ObjectType', ':=""'), ('InheritedObjectTypeLen', '_-InheritedObjectType', 'self.checkInheritedObjectType(self["Flags"])'), ('InheritedObjectType', ':=""'), ('Sid', ':', LDAP_SID))

    @staticmethod
    def checkInheritedObjectType(flags):
        if False:
            i = 10
            return i + 15
        if flags & ACCESS_ALLOWED_OBJECT_ACE.ACE_INHERITED_OBJECT_TYPE_PRESENT:
            return 16
        return 0

    @staticmethod
    def checkObjectType(flags):
        if False:
            i = 10
            return i + 15
        if flags & ACCESS_ALLOWED_OBJECT_ACE.ACE_OBJECT_TYPE_PRESENT:
            return 16
        return 0

    def getData(self):
        if False:
            return 10
        if self['ObjectType'] != b'':
            self['Flags'] |= self.ACE_OBJECT_TYPE_PRESENT
        if self['InheritedObjectType'] != b'':
            self['Flags'] |= self.ACE_INHERITED_OBJECT_TYPE_PRESENT
        return Structure.getData(self)

    def hasFlag(self, flag):
        if False:
            for i in range(10):
                print('nop')
        return self['Flags'] & flag == flag
'\nACCESS_DENIED_ACE as described in 2.4.4.4\nhttps://msdn.microsoft.com/en-us/library/cc230291.aspx\nStructure is identical to ACCESS_ALLOWED_ACE\n'

class ACCESS_DENIED_ACE(ACCESS_ALLOWED_ACE):
    ACE_TYPE = 1
'\nACCESS_DENIED_OBJECT_ACE as described in 2.4.4.5\nhttps://msdn.microsoft.com/en-us/library/gg750297.aspx\nStructure is identical to ACCESS_ALLOWED_OBJECT_ACE\n'

class ACCESS_DENIED_OBJECT_ACE(ACCESS_ALLOWED_OBJECT_ACE):
    ACE_TYPE = 6
'\nACCESS_ALLOWED_CALLBACK_ACE as described in 2.4.4.6\nhttps://msdn.microsoft.com/en-us/library/cc230287.aspx\n'

class ACCESS_ALLOWED_CALLBACK_ACE(Structure):
    ACE_TYPE = 9
    structure = (('Mask', ':', ACCESS_MASK), ('Sid', ':', LDAP_SID), ('ApplicationData', ':'))
'\nACCESS_DENIED_OBJECT_ACE as described in 2.4.4.7\nhttps://msdn.microsoft.com/en-us/library/cc230292.aspx\nStructure is identical to ACCESS_ALLOWED_CALLBACK_ACE\n'

class ACCESS_DENIED_CALLBACK_ACE(ACCESS_ALLOWED_CALLBACK_ACE):
    ACE_TYPE = 10
'\nACCESS_ALLOWED_CALLBACK_OBJECT_ACE as described in 2.4.4.8\nhttps://msdn.microsoft.com/en-us/library/cc230288.aspx\n'

class ACCESS_ALLOWED_CALLBACK_OBJECT_ACE(ACCESS_ALLOWED_OBJECT_ACE):
    ACE_TYPE = 11
    structure = (('Mask', ':', ACCESS_MASK), ('Flags', '<L'), ('ObjectTypeLen', '_-ObjectType', 'self.checkObjectType(self["Flags"])'), ('ObjectType', ':=""'), ('InheritedObjectTypeLen', '_-InheritedObjectType', 'self.checkInheritedObjectType(self["Flags"])'), ('InheritedObjectType', ':=""'), ('Sid', ':', LDAP_SID), ('ApplicationData', ':'))
'\nACCESS_DENIED_CALLBACK_OBJECT_ACE as described in 2.4.4.7\nhttps://msdn.microsoft.com/en-us/library/cc230292.aspx\nStructure is identical to ACCESS_ALLOWED_OBJECT_OBJECT_ACE\n'

class ACCESS_DENIED_CALLBACK_OBJECT_ACE(ACCESS_ALLOWED_CALLBACK_OBJECT_ACE):
    ACE_TYPE = 12
'\nSYSTEM_AUDIT_ACE as described in 2.4.4.10\nhttps://msdn.microsoft.com/en-us/library/cc230376.aspx\nStructure is identical to ACCESS_ALLOWED_ACE\n'

class SYSTEM_AUDIT_ACE(ACCESS_ALLOWED_ACE):
    ACE_TYPE = 2
'\nSYSTEM_AUDIT_OBJECT_ACE as described in 2.4.4.11\nhttps://msdn.microsoft.com/en-us/library/gg750298.aspx\nStructure is identical to ACCESS_ALLOWED_CALLBACK_OBJECT_ACE\n'

class SYSTEM_AUDIT_OBJECT_ACE(ACCESS_ALLOWED_CALLBACK_OBJECT_ACE):
    ACE_TYPE = 7
'\nSYSTEM_AUDIT_CALLBACK_ACE as described in 2.4.4.12\nhttps://msdn.microsoft.com/en-us/library/cc230377.aspx\nStructure is identical to ACCESS_ALLOWED_CALLBACK_ACE\n'

class SYSTEM_AUDIT_CALLBACK_ACE(ACCESS_ALLOWED_CALLBACK_ACE):
    ACE_TYPE = 13
'\nSYSTEM_AUDIT_CALLBACK_ACE as described in 2.4.4.13\nhttps://msdn.microsoft.com/en-us/library/cc230379.aspx\nStructure is identical to ACCESS_ALLOWED_ACE, but with custom masks and meanings.\nLets keep it separate for now\n'

class SYSTEM_MANDATORY_LABEL_ACE(Structure):
    ACE_TYPE = 17
    structure = (('Mask', ':', ACCESS_MASK), ('Sid', ':', LDAP_SID))
'\nSYSTEM_AUDIT_CALLBACK_ACE as described in 2.4.4.14\nhttps://msdn.microsoft.com/en-us/library/cc230378.aspx\nStructure is identical to ACCESS_ALLOWED_CALLBACK_OBJECT_ACE\n'

class SYSTEM_AUDIT_CALLBACK_OBJECT_ACE(ACCESS_ALLOWED_CALLBACK_OBJECT_ACE):
    ACE_TYPE = 15
'\nSYSTEM_RESOURCE_ATTRIBUTE_ACE as described in 2.4.4.15\nhttps://msdn.microsoft.com/en-us/library/hh877837.aspx\nStructure is identical to ACCESS_ALLOWED_CALLBACK_ACE\nThe application data however is encoded in CLAIM_SECURITY_ATTRIBUTE_RELATIVE_V1\nformat as described in section 2.4.10.1\nTodo: implement this substructure if needed\n'

class SYSTEM_RESOURCE_ATTRIBUTE_ACE(ACCESS_ALLOWED_CALLBACK_ACE):
    ACE_TYPE = 18
'\nSYSTEM_SCOPED_POLICY_ID_ACE as described in 2.4.4.16\nhttps://msdn.microsoft.com/en-us/library/hh877846.aspx\nStructure is identical to ACCESS_ALLOWED_ACE\nThe Sid data MUST match a CAPID of a CentralAccessPolicy\ncontained in the CentralAccessPoliciesList\nTodo: implement this substructure if needed\nAlso the ACCESS_MASK must always be 0\n'

class SYSTEM_SCOPED_POLICY_ID_ACE(ACCESS_ALLOWED_ACE):
    ACE_TYPE = 19
ACE_TYPES = [ACCESS_ALLOWED_ACE, ACCESS_ALLOWED_OBJECT_ACE, ACCESS_DENIED_ACE, ACCESS_DENIED_OBJECT_ACE, ACCESS_ALLOWED_CALLBACK_ACE, ACCESS_DENIED_CALLBACK_ACE, ACCESS_ALLOWED_CALLBACK_OBJECT_ACE, ACCESS_DENIED_CALLBACK_OBJECT_ACE, SYSTEM_AUDIT_ACE, SYSTEM_AUDIT_OBJECT_ACE, SYSTEM_AUDIT_CALLBACK_ACE, SYSTEM_MANDATORY_LABEL_ACE, SYSTEM_AUDIT_CALLBACK_OBJECT_ACE, SYSTEM_RESOURCE_ATTRIBUTE_ACE, SYSTEM_SCOPED_POLICY_ID_ACE]
ACE_TYPE_MAP = {ace.ACE_TYPE: ace for ace in ACE_TYPES}
'\nACL as described in 2.4.5\nhttps://msdn.microsoft.com/en-us/library/cc230297.aspx\n'

class ACL(Structure):
    structure = (('AclRevision', 'B'), ('Sbz1', 'B'), ('AclSize', '<H'), ('AceCount', '<H'), ('Sbz2', '<H'), ('DataLen', '_-Data', 'self["AclSize"]-8'), ('Data', ':'))

    def fromString(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.aces = []
        Structure.fromString(self, data)
        for i in range(self['AceCount']):
            if len(self['Data']) == 0:
                raise Exception('ACL header indicated there are more ACLs to unpack, but there is no more data')
            ace = ACE(data=self['Data'])
            self.aces.append(ace)
            self['Data'] = self['Data'][ace['AceSize']:]
        self['Data'] = self.aces

    def getData(self):
        if False:
            i = 10
            return i + 15
        self['AceCount'] = len(self.aces)
        self['Data'] = b''.join([ace.getData() for ace in self.aces])
        self['AclSize'] = len(self['Data']) + 8
        data = Structure.getData(self)
        self['Data'] = self.aces
        return data
'\nobjectClass mapping to GUID for some common classes (index is the ldapDisplayName).\nReference:\n    https://msdn.microsoft.com/en-us/library/ms680938(v=vs.85).aspx\nCan also be queried from the Schema\n'
OBJECTTYPE_GUID_MAP = {b'group': 'bf967a9c-0de6-11d0-a285-00aa003049e2', b'domain': '19195a5a-6da0-11d0-afd3-00c04fd930c9', b'organizationalUnit': 'bf967aa5-0de6-11d0-a285-00aa003049e2', b'user': 'bf967aba-0de6-11d0-a285-00aa003049e2', b'groupPolicyContainer': 'f30e3bc2-9ff0-11d1-b603-0000f80367c1'}