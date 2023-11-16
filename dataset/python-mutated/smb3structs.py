from impacket.structure import Structure
SMB2_PACKET_SIZE = 64
SMB2_NEGOTIATE = 0
SMB2_SESSION_SETUP = 1
SMB2_LOGOFF = 2
SMB2_TREE_CONNECT = 3
SMB2_TREE_DISCONNECT = 4
SMB2_CREATE = 5
SMB2_CLOSE = 6
SMB2_FLUSH = 7
SMB2_READ = 8
SMB2_WRITE = 9
SMB2_LOCK = 10
SMB2_IOCTL = 11
SMB2_CANCEL = 12
SMB2_ECHO = 13
SMB2_QUERY_DIRECTORY = 14
SMB2_CHANGE_NOTIFY = 15
SMB2_QUERY_INFO = 16
SMB2_SET_INFO = 17
SMB2_OPLOCK_BREAK = 18
SMB2_FLAGS_SERVER_TO_REDIR = 1
SMB2_FLAGS_ASYNC_COMMAND = 2
SMB2_FLAGS_RELATED_OPERATIONS = 4
SMB2_FLAGS_SIGNED = 8
SMB2_FLAGS_DFS_OPERATIONS = 268435456
SMB2_FLAGS_REPLAY_OPERATION = 2147483648
SYMLINK_FLAG_ABSOLUTE = 0
SYMLINK_FLAG_RELATIVE = 1
SMB2_NEGOTIATE_SIGNING_ENABLED = 1
SMB2_NEGOTIATE_SIGNING_REQUIRED = 2
SMB2_GLOBAL_CAP_DFS = 1
SMB2_GLOBAL_CAP_LEASING = 2
SMB2_GLOBAL_CAP_LARGE_MTU = 4
SMB2_GLOBAL_CAP_MULTI_CHANNEL = 8
SMB2_GLOBAL_CAP_PERSISTENT_HANDLES = 16
SMB2_GLOBAL_CAP_DIRECTORY_LEASING = 32
SMB2_GLOBAL_CAP_ENCRYPTION = 64
SMB2_DIALECT_002 = 514
SMB2_DIALECT_21 = 528
SMB2_DIALECT_30 = 768
SMB2_DIALECT_WILDCARD = 767
SMB2_SESSION_FLAG_BINDING = 1
SMB2_SESSION_FLAG_IS_GUEST = 1
SMB2_SESSION_FLAG_IS_NULL = 2
SMB2_SESSION_FLAG_ENCRYPT_DATA = 4
SMB2_SHARE_TYPE_DISK = 1
SMB2_SHARE_TYPE_PIPE = 2
SMB2_SHARE_TYPE_PRINT = 3
SMB2_SHAREFLAG_MANUAL_CACHING = 0
SMB2_SHAREFLAG_AUTO_CACHING = 16
SMB2_SHAREFLAG_VDO_CACHING = 32
SMB2_SHAREFLAG_NO_CACHING = 48
SMB2_SHAREFLAG_DFS = 1
SMB2_SHAREFLAG_DFS_ROOT = 2
SMB2_SHAREFLAG_RESTRICT_EXCLUSIVE_OPENS = 256
SMB2_SHAREFLAG_FORCE_SHARED_DELETE = 512
SMB2_SHAREFLAG_ALLOW_NAMESPACE_CACHING = 1024
SMB2_SHAREFLAG_ACCESS_BASED_DIRECTORY_ENUM = 2048
SMB2_SHAREFLAG_FORCE_LEVELII_OPLOCK = 4096
SMB2_SHAREFLAG_ENABLE_HASH_V1 = 8192
SMB2_SHAREFLAG_ENABLE_HASH_V2 = 16384
SMB2_SHAREFLAG_ENCRYPT_DATA = 32768
SMB2_SHARE_CAP_DFS = 8
SMB2_SHARE_CAP_CONTINUOUS_AVAILABILITY = 16
SMB2_SHARE_CAP_SCALEOUT = 32
SMB2_SHARE_CAP_CLUSTER = 64
SMB2_OPLOCK_LEVEL_NONE = 0
SMB2_OPLOCK_LEVEL_II = 1
SMB2_OPLOCK_LEVEL_EXCLUSIVE = 8
SMB2_OPLOCK_LEVEL_BATCH = 9
SMB2_OPLOCK_LEVEL_LEASE = 255
SMB2_IL_ANONYMOUS = 0
SMB2_IL_IDENTIFICATION = 1
SMB2_IL_IMPERSONATION = 2
SMB2_IL_DELEGATE = 3
FILE_ATTRIBUTE_ARCHIVE = 32
FILE_ATTRIBUTE_COMPRESSED = 2048
FILE_ATTRIBUTE_DIRECTORY = 16
FILE_ATTRIBUTE_ENCRYPTED = 16384
FILE_ATTRIBUTE_HIDDEN = 2
FILE_ATTRIBUTE_NORMAL = 128
FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 8192
FILE_ATTRIBUTE_OFFLINE = 4096
FILE_ATTRIBUTE_READONLY = 1
FILE_ATTRIBUTE_REPARSE_POINT = 1024
FILE_ATTRIBUTE_SPARSE_FILE = 512
FILE_ATTRIBUTE_SYSTEM = 4
FILE_ATTRIBUTE_TEMPORARY = 256
FILE_ATTRIBUTE_INTEGRITY_STREAM = 2048
FILE_ATTRIBUTE_NO_SCRUB_DATA = 131072
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
FILE_SHARE_DELETE = 4
FILE_SUPERSEDE = 0
FILE_OPEN = 1
FILE_CREATE = 2
FILE_OPEN_IF = 3
FILE_OVERWRITE = 4
FILE_OVERWRITE_IF = 5
FILE_DIRECTORY_FILE = 1
FILE_WRITE_THROUGH = 2
FILE_SEQUENTIAL_ONLY = 4
FILE_NO_INTERMEDIATE_BUFFERING = 8
FILE_SYNCHRONOUS_IO_ALERT = 16
FILE_SYNCHRONOUS_IO_NONALERT = 32
FILE_NON_DIRECTORY_FILE = 64
FILE_COMPLETE_IF_OPLOCKED = 256
FILE_NO_EA_KNOWLEDGE = 512
FILE_RANDOM_ACCESS = 2048
FILE_DELETE_ON_CLOSE = 4096
FILE_OPEN_BY_FILE_ID = 8192
FILE_OPEN_FOR_BACKUP_INTENT = 16384
FILE_NO_COMPRESSION = 32768
FILE_RESERVE_OPFILTER = 1048576
FILE_OPEN_REPARSE_POINT = 2097152
FILE_OPEN_NO_RECALL = 4194304
FILE_OPEN_FOR_FREE_SPACE_QUERY = 8388608
FILE_READ_DATA = 1
FILE_WRITE_DATA = 2
FILE_APPEND_DATA = 4
FILE_READ_EA = 8
FILE_WRITE_EA = 16
FILE_EXECUTE = 32
FILE_READ_ATTRIBUTES = 128
FILE_WRITE_ATTRIBUTES = 256
DELETE = 65536
READ_CONTROL = 131072
WRITE_DAC = 262144
WRITE_OWNER = 524288
SYNCHRONIZE = 1048576
ACCESS_SYSTEM_SECURITY = 16777216
MAXIMUM_ALLOWED = 33554432
GENERIC_ALL = 268435456
GENERIC_EXECUTE = 536870912
GENERIC_WRITE = 1073741824
GENERIC_READ = 2147483648
FILE_LIST_DIRECTORY = 1
FILE_ADD_FILE = 2
FILE_ADD_SUBDIRECTORY = 4
FILE_TRAVERSE = 32
FILE_DELETE_CHILD = 64
SMB2_CREATE_EA_BUFFER = 1165521985
SMB2_CREATE_SD_BUFFER = 1399153476
SMB2_CREATE_DURABLE_HANDLE_REQUEST = 1145597521
SMB2_CREATE_DURABLE_HANDLE_RECONNECT = 1145597507
SMB2_CREATE_ALLOCATION_SIZE = 1097618281
SMB2_CREATE_QUERY_MAXIMAL_ACCESS_REQUEST = 1299726691
SMB2_CREATE_TIMEWARP_TOKEN = 1415017072
SMB2_CREATE_QUERY_ON_DISK_ID = 1363568996
SMB2_CREATE_REQUEST = 1383156851
SMB2_CREATE_REQUEST_LEASE_V2 = 1383156851
SMB2_CREATE_DURABLE_HANDLE_REQUEST_V2 = 1145582161
SMB2_CREATE_DURABLE_HANDLE_RECONNECT_V2 = 1145582147
SMB2_CREATE_APP_INSTANCE_ID = 92696258870913151272972845903485619572
SMB2_CREATE_FLAG_REPARSEPOINT = 1
FILE_NEED_EA = 128
FILE_SUPERSEDED = 0
FILE_OPENED = 1
FILE_CREATED = 2
FILE_OVERWRITTEN = 3
SMB2_LEASE_NONE = 0
SMB2_LEASE_READ_CACHING = 1
SMB2_LEASE_HANDLE_CACHING = 2
SMB2_LEASE_WRITE_CACHING = 4
SMB2_LEASE_FLAG_PARENT_LEASE_KEY_SET = 4
SMB2_DHANDLE_FLAG_PERSISTENT = 2
SMB2_CLOSE_FLAG_POSTQUERY_ATTRIB = 1
SMB2_CHANNEL_NONE = 0
SMB2_CHANNEL_RDMA_V1 = 1
SMB2_WRITEFLAG_WRITE_THROUGH = 1
SMB2_NOTIFY_BREAK_LEASE_FLAG_ACK_REQUIRED = 1
SMB2_LOCKFLAG_SHARED_LOCK = 1
SMB2_LOCKFLAG_EXCLUSIVE_LOCK = 2
SMB2_LOCKFLAG_UNLOCK = 4
SMB2_LOCKFLAG_FAIL_IMMEDIATELY = 16
FSCTL_DFS_GET_REFERRALS = 393620
FSCTL_PIPE_PEEK = 1130508
FSCTL_PIPE_WAIT = 1114136
FSCTL_PIPE_TRANSCEIVE = 1163287
FSCTL_SRV_COPYCHUNK = 1327346
FSCTL_SRV_ENUMERATE_SNAPSHOTS = 1327204
FSCTL_SRV_REQUEST_RESUME_KEY = 1310840
FSCTL_SRV_READ_HASH = 1327547
FSCTL_SRV_COPYCHUNK_WRITE = 1343730
FSCTL_LMR_REQUEST_RESILIENCY = 1311188
FSCTL_QUERY_NETWORK_INTERFACE_INFO = 1311228
FSCTL_SET_REPARSE_POINT = 589988
FSCTL_DFS_GET_REFERRALS_EX = 393648
FSCTL_FILE_LEVEL_TRIM = 623112
FSCTL_VALIDATE_NEGOTIATE_INFO = 1311236
SMB2_0_IOCTL_IS_FSCTL = 1
SRV_HASH_TYPE_PEER_DIST = 1
SRV_HASH_VER_1 = 1
SRV_HASH_VER_2 = 2
SRV_HASH_RETRIEVE_HASH_BASED = 1
SRV_HASH_RETRIEVE_FILE_BASED = 2
RSS_CAPABLE = 1
RDMA_CAPABLE = 2
FILE_DIRECTORY_INFORMATION = 1
FILE_FULL_DIRECTORY_INFORMATION = 2
FILEID_FULL_DIRECTORY_INFORMATION = 38
FILE_BOTH_DIRECTORY_INFORMATION = 3
FILEID_BOTH_DIRECTORY_INFORMATION = 37
FILENAMES_INFORMATION = 12
SMB2_RESTART_SCANS = 1
SMB2_RETURN_SINGLE_ENTRY = 2
SMB2_INDEX_SPECIFIED = 4
SMB2_REOPEN = 16
SMB2_WATCH_TREE = 1
FILE_NOTIFY_CHANGE_FILE_NAME = 1
FILE_NOTIFY_CHANGE_DIR_NAME = 2
FILE_NOTIFY_CHANGE_ATTRIBUTES = 4
FILE_NOTIFY_CHANGE_SIZE = 8
FILE_NOTIFY_CHANGE_LAST_WRITE = 16
FILE_NOTIFY_CHANGE_LAST_ACCESS = 32
FILE_NOTIFY_CHANGE_CREATION = 64
FILE_NOTIFY_CHANGE_EA = 128
FILE_NOTIFY_CHANGE_SECURITY = 256
FILE_NOTIFY_CHANGE_STREAM_NAME = 512
FILE_NOTIFY_CHANGE_STREAM_SIZE = 1024
FILE_NOTIFY_CHANGE_STREAM_WRITE = 2048
FILE_ACTION_ADDED = 1
FILE_ACTION_REMOVED = 2
FILE_ACTION_MODIFIED = 3
FILE_ACTION_RENAMED_OLD_NAME = 4
FILE_ACTION_RENAMED_NEW_NAME = 5
SMB2_0_INFO_FILE = 1
SMB2_0_INFO_FILESYSTEM = 2
SMB2_0_INFO_SECURITY = 3
SMB2_0_INFO_QUOTA = 4
SMB2_FILE_ACCESS_INFO = 8
SMB2_FILE_ALIGNMENT_INFO = 17
SMB2_FILE_ALL_INFO = 18
SMB2_FILE_ALLOCATION_INFO = 19
SMB2_FILE_ALTERNATE_NAME_INFO = 21
SMB2_ATTRIBUTE_TAG_INFO = 35
SMB2_FILE_BASIC_INFO = 4
SMB2_FILE_BOTH_DIRECTORY_INFO = 3
SMB2_FILE_COMPRESSION_INFO = 28
SMB2_FILE_DIRECTORY_INFO = 1
SMB2_FILE_DISPOSITION_INFO = 13
SMB2_FILE_EA_INFO = 7
SMB2_FILE_END_OF_FILE_INFO = 20
SMB2_FULL_DIRECTORY_INFO = 2
SMB2_FULL_EA_INFO = 15
SMB2_FILE_HARDLINK_INFO = 46
SMB2_FILE_ID_BOTH_DIRECTORY_INFO = 37
SMB2_FILE_ID_FULL_DIRECTORY_INFO = 38
SMB2_FILE_ID_GLOBAL_TX_DIRECTORY_INFO = 50
SMB2_FILE_INTERNAL_INFO = 6
SMB2_FILE_LINK_INFO = 11
SMB2_FILE_MAILSLOT_QUERY_INFO = 26
SMB2_FILE_MAILSLOT_SET_INFO = 27
SMB2_FILE_MODE_INFO = 16
SMB2_FILE_MOVE_CLUSTER_INFO = 31
SMB2_FILE_NAME_INFO = 9
SMB2_FILE_NAMES_INFO = 12
SMB2_FILE_NETWORK_OPEN_INFO = 34
SMB2_FILE_NORMALIZED_NAME_INFO = 48
SMB2_FILE_OBJECT_ID_INFO = 29
SMB2_FILE_PIPE_INFO = 23
SMB2_FILE_PIPE_LOCAL_INFO = 24
SMB2_FILE_PIPE_REMOTE_INFO = 25
SMB2_FILE_POSITION_INFO = 14
SMB2_FILE_QUOTA_INFO = 32
SMB2_FILE_RENAME_INFO = 10
SMB2_FILE_REPARSE_POINT_INFO = 33
SMB2_FILE_SFIO_RESERVE_INFO = 44
SMB2_FILE_SHORT_NAME_INFO = 45
SMB2_FILE_STANDARD_INFO = 5
SMB2_FILE_STANDARD_LINK_INFO = 54
SMB2_FILE_STREAM_INFO = 22
SMB2_FILE_TRACKING_INFO = 36
SMB2_FILE_VALID_DATA_LENGTH_INFO = 39
SMB2_FILESYSTEM_VOLUME_INFO = 1
SMB2_FILESYSTEM_LABEL_INFO = 2
SMB2_FILESYSTEM_SIZE_INFO = 3
SMB2_FILESYSTEM_DEVICE_INFO = 4
SMB2_FILESYSTEM_ATTRIBUTE_INFO = 5
SMB2_FILESYSTEM_CONTROL_INFO = 6
SMB2_FILESYSTEM_FULL_SIZE_INFO = 7
SMB2_FILESYSTEM_OBJECT_ID_INFO = 8
SMB2_FILESYSTEM_DRIVER_PATH_INFO = 9
SMB2_FILESYSTEM_SECTOR_SIZE_INFO = 11
OWNER_SECURITY_INFORMATION = 1
GROUP_SECURITY_INFORMATION = 2
DACL_SECURITY_INFORMATION = 4
SACL_SECURITY_INFORMATION = 8
LABEL_SECURITY_INFORMATION = 16
SL_RESTART_SCAN = 1
SL_RETURN_SINGLE_ENTRY = 2
SL_INDEX_SPECIFIED = 4
SMB2_ENCRYPTION_AES128_CCM = 1

class SMBPacketBase(Structure):

    def addCommand(self, command):
        if False:
            i = 10
            return i + 15
        raise 'Implement This!'

    def isValidAnswer(self, status):
        if False:
            i = 10
            return i + 15
        if self['Status'] != status:
            import smb3
            raise smb3.SessionError(self['Status'], self)
        return True

    def __init__(self, data=None):
        if False:
            return 10
        Structure.__init__(self, data)
        if data is None:
            self['TreeID'] = 0

class SMB2PacketAsync(SMBPacketBase):
    structure = (('ProtocolID', '"þSMB'), ('StructureSize', '<H=64'), ('CreditCharge', '<H=0'), ('Status', '<L=0'), ('Command', '<H=0'), ('CreditRequestResponse', '<H=0'), ('Flags', '<L=0'), ('NextCommand', '<L=0'), ('MessageID', '<Q=0'), ('AsyncID', '<Q=0'), ('SessionID', '<Q=0'), ('Signature', '16s=""'), ('Data', ':=""'))

class SMB3PacketAsync(SMBPacketBase):
    structure = (('ProtocolID', '"þSMB'), ('StructureSize', '<H=64'), ('CreditCharge', '<H=0'), ('ChannelSequence', '<H=0'), ('Reserved', '<H=0'), ('Command', '<H=0'), ('CreditRequestResponse', '<H=0'), ('Flags', '<L=0'), ('NextCommand', '<L=0'), ('MessageID', '<Q=0'), ('AsyncID', '<Q=0'), ('SessionID', '<Q=0'), ('Signature', '16s=""'), ('Data', ':=""'))

class SMB2Packet(SMBPacketBase):
    structure = (('ProtocolID', '"þSMB'), ('StructureSize', '<H=64'), ('CreditCharge', '<H=0'), ('Status', '<L=0'), ('Command', '<H=0'), ('CreditRequestResponse', '<H=0'), ('Flags', '<L=0'), ('NextCommand', '<L=0'), ('MessageID', '<Q=0'), ('Reserved', '<L=0'), ('TreeID', '<L=0'), ('SessionID', '<Q=0'), ('Signature', '16s=""'), ('Data', ':=""'))

class SMB3Packet(SMBPacketBase):
    structure = (('ProtocolID', '"þSMB'), ('StructureSize', '<H=64'), ('CreditCharge', '<H=0'), ('ChannelSequence', '<H=0'), ('Reserved', '<H=0'), ('Command', '<H=0'), ('CreditRequestResponse', '<H=0'), ('Flags', '<L=0'), ('NextCommand', '<L=0'), ('MessageID', '<Q=0'), ('Reserved', '<L=0'), ('TreeID', '<L=0'), ('SessionID', '<Q=0'), ('Signature', '16s=""'), ('Data', ':=""'))

class SMB2Error(Structure):
    structure = (('StructureSize', '<H=9'), ('Reserved', '<H=0'), ('ByteCount', '<L=0'), ('_ErrorData', '_-ErrorData', 'self["ByteCount"]'), ('ErrorData', '"ÿ'))

class SMB2ErrorSymbolicLink(Structure):
    structure = (('SymLinkLength', '<L=0'), ('SymLinkErrorTag', '<L=0'), ('ReparseTag', '<L=0'), ('ReparseDataLenght', '<H=0'), ('UnparsedPathLength', '<H=0'), ('SubstituteNameOffset', '<H=0'), ('SubstituteNameLength', '<H=0'), ('PrintNameOffset', '<H=0'), ('PrintNameLength', '<H=0'), ('Flags', '<L=0'), ('PathBuffer', ':'))

class SMB2Negotiate(Structure):
    structure = (('StructureSize', '<H=36'), ('DialectCount', '<H=0'), ('SecurityMode', '<H=0'), ('Reserved', '<H=0'), ('Capabilities', '<L=0'), ('ClientGuid', '16s=""'), ('ClientStartTime', '<Q=0'), ('Dialects', '*<H'))

class SMB2Negotiate_Response(Structure):
    structure = (('StructureSize', '<H=65'), ('SecurityMode', '<H=0'), ('DialectRevision', '<H=0'), ('Reserved', '<H=0'), ('ServerGuid', '16s=""'), ('Capabilities', '<L=0'), ('MaxTransactSize', '<L=0'), ('MaxReadSize', '<L=0'), ('MaxWriteSize', '<L=0'), ('SystemTime', '<Q=0'), ('ServerStartTime', '<Q=0'), ('SecurityBufferOffset', '<H=0'), ('SecurityBufferLength', '<H=0'), ('Reserved2', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["SecurityBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["SecurityBufferLength"]'), ('Buffer', ':'))

class SMB2SessionSetup(Structure):
    SIZE = 24
    structure = (('StructureSize', '<H=25'), ('Flags', '<B=0'), ('SecurityMode', '<B=0'), ('Capabilities', '<L=0'), ('Channel', '<L=0'), ('SecurityBufferOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('SecurityBufferLength', '<H=0'), ('PreviousSessionId', '<Q=0'), ('_AlignPad', '_-AlignPad', 'self["SecurityBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["SecurityBufferLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            while True:
                i = 10
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

    def getData(self):
        if False:
            for i in range(10):
                print('nop')
        return Structure.getData(self)

class SMB2SessionSetup_Response(Structure):
    structure = (('StructureSize', '<H=9'), ('SessionFlags', '<H=0'), ('SecurityBufferOffset', '<H=0'), ('SecurityBufferLength', '<H=0'), ('_AlignPad', '_-AlignPad', 'self["SecurityBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["SecurityBufferLength"]'), ('Buffer', ':'))

class SMB2Logoff(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2Logoff_Response(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2TreeConnect(Structure):
    SIZE = 8
    structure = (('StructureSize', '<H=9'), ('Reserved', '<H=0'), ('PathOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('PathLength', '<H=0'), ('_AlignPad', '_-AlignPad', 'self["PathOffset"] - (64 + self.SIZE - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["PathLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2TreeConnect_Response(Structure):
    structure = (('StructureSize', '<H=16'), ('ShareType', '<B=0'), ('Reserved', '<B=0'), ('ShareFlags', '<L=0'), ('Capabilities', '<L=0'), ('MaximalAccess', '<L=0'))

class SMB2TreeDisconnect(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2TreeDisconnect_Response(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2Create(Structure):
    SIZE = 56
    structure = (('StructureSize', '<H=57'), ('SecurityFlags', '<B=0'), ('RequestedOplockLevel', '<B=0'), ('ImpersonationLevel', '<L=0'), ('SmbCreateFlags', '<Q=0'), ('Reserved', '<Q=0'), ('DesiredAccess', '<L=0'), ('FileAttributes', '<L=0'), ('ShareAccess', '<L=0'), ('CreateDisposition', '<L=0'), ('CreateOptions', '<L=0'), ('NameOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('NameLength', '<H=0'), ('CreateContextsOffset', '<L=0'), ('CreateContextsLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["NameOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["CreateContextsLength"]+self["NameLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2CreateContext(Structure):
    structure = (('Next', '<L=0'), ('NameOffset', '<H=0'), ('NameLength', '<H=0'), ('Reserved', '<H=0'), ('DataOffset', '<H=0'), ('DataLength', '<L=0'), ('_Buffer', '_-Buffer', 'self["DataLength"]+self["NameLength"]'), ('Buffer', ':'))

class SMB2_FILEID(Structure):
    structure = (('Persistent', '<Q=0'), ('Volatile', '<Q=0'))

class SMB2Create_Response(Structure):
    structure = (('StructureSize', '<H=89'), ('OplockLevel', '<B=0'), ('Flags', '<B=0'), ('CreateAction', '<L=0'), ('CreationTime', '<Q=0'), ('LastAccessTime', '<Q=0'), ('LastWriteTime', '<Q=0'), ('ChangeTime', '<Q=0'), ('AllocationSize', '<Q=0'), ('EndOfFile', '<Q=0'), ('FileAttributes', '<L=0'), ('Reserved2', '<L=0'), ('FileID', ':', SMB2_FILEID), ('CreateContextsOffset', '<L=0'), ('CreateContextsLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["CreateContextsOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["CreateContextsLength"]'), ('Buffer', ':'))

class FILE_FULL_EA_INFORMATION(Structure):
    structure = (('NextEntryOffset', '<L=0'), ('Flags', '<B=0'), ('EaNameLength', '<B=0'), ('EaValueLength', '<H=0'), ('_EaName', '_-EaName', 'self["EaNameLength"]'), ('EaName', ':'), ('_EaValue', '_-EaValue', 'self["EaValue"]'), ('EaValue', ':'))

class SMB2_CREATE_DURABLE_HANDLE_RECONNECT(Structure):
    structure = (('Data', ':', SMB2_FILEID),)

class SMB2_CREATE_DURABLE_HANDLE_REQUEST(Structure):
    structure = (('DurableRequest', '16s=""'),)

class SMB2_CREATE_DURABLE_HANDLE_RESPONSE(Structure):
    structure = (('Reserved', '<Q=0'),)

class SMB2_CREATE_QUERY_MAXIMAL_ACCESS_REQUEST(Structure):
    structure = (('Timestamp', '<Q=0'),)

class SMB2_CREATE_QUERY_MAXIMAL_ACCESS_RESPONSE(Structure):
    structure = (('QueryStatus', '<L=0'), ('MaximalAccess', '<L=0'))

class SMB2_CREATE_ALLOCATION_SIZE(Structure):
    structure = (('AllocationSize', '<Q=0'),)

class SMB2_CREATE_TIMEWARP_TOKEN(Structure):
    structure = (('AllocationSize', '<Q=0'),)

class SMB2_CREATE_REQUEST_LEASE(Structure):
    structure = (('LeaseKey', '16s=""'), ('LeaseState', '<L=0'), ('LeaseFlags', '<L=0'), ('LeaseDuration', '<Q=0'))
SMB2_CREATE_RESPONSE_LEASE = SMB2_CREATE_REQUEST_LEASE

class SMB2_CREATE_REQUEST_LEASE_V2(Structure):
    structure = (('LeaseKey', '16s=""'), ('LeaseState', '<L=0'), ('Flags', '<L=0'), ('LeaseDuration', '<Q=0'), ('ParentLeaseKey', '16s=""'), ('Epoch', '<H=0'), ('Reserved', '<H=0'))
SMB2_CREATE_RESPONSE_LEASE_V2 = SMB2_CREATE_REQUEST_LEASE_V2

class SMB2_CREATE_DURABLE_HANDLE_REQUEST_V2(Structure):
    structure = (('Timeout', '<L=0'), ('Flags', '<L=0'), ('Reserved', '8s=""'), ('CreateGuid', '16s=""'))

class SMB2_CREATE_DURABLE_HANDLE_RESPONSE_V2(Structure):
    structure = (('Timeout', '<L=0'), ('Flags', '<L=0'))

class SMB2_CREATE_DURABLE_HANDLE_RECONNECT_V2(Structure):
    structure = (('FileID', ':', SMB2_FILEID), ('CreateGuid', '16s=""'), ('Flags', '<L=0'))

class SMB2_CREATE_APP_INSTANCE_ID(Structure):
    structure = (('StructureSize', '<H=0'), ('Reserved', '<H=0'), ('AppInstanceId', '16s=""'))

class SMB2_CREATE_QUERY_ON_DISK_ID(Structure):
    structure = (('DiskIDBuffer', '32s=""'),)

class SMB2Close(Structure):
    structure = (('StructureSize', '<H=24'), ('Flags', '<H=0'), ('Reserved', '<L=0'), ('FileID', ':', SMB2_FILEID))

class SMB2Close_Response(Structure):
    structure = (('StructureSize', '<H=60'), ('Flags', '<H=0'), ('Reserved', '<L=0'), ('CreationTime', '<Q=0'), ('LastAccessTime', '<Q=0'), ('LastWriteTime', '<Q=0'), ('ChangeTime', '<Q=0'), ('AllocationSize', '<Q=0'), ('EndofFile', '<Q=0'), ('FileAttributes', '<L=0'))

class SMB2Flush(Structure):
    structure = (('StructureSize', '<H=24'), ('Reserved1', '<H=0'), ('Reserved2', '<L=0'), ('FileID', ':', SMB2_FILEID))

class SMB2Flush_Response(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2Read(Structure):
    SIZE = 48
    structure = (('StructureSize', '<H=49'), ('Padding', '<B=0'), ('Reserved', '<B=0'), ('Length', '<L=0'), ('Offset', '<Q=0'), ('FileID', ':', SMB2_FILEID), ('MinimumCount', '<L=0'), ('Channel', '<L=0'), ('RemainingBytes', '<L=0'), ('ReadChannelInfoOffset', '<H=0'), ('ReadChannelInfoLength', '<H=0'), ('_AlignPad', '_-AlignPad', 'self["ReadChannelInfoOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["ReadChannelInfoLength"]'), ('Buffer', ':=0'))

    def __init__(self, data=None):
        if False:
            while True:
                i = 10
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2Read_Response(Structure):
    structure = (('StructureSize', '<H=17'), ('DataOffset', '<B=0'), ('Reserved', '<B=0'), ('DataLength', '<L=0'), ('DataRemaining', '<L=0'), ('Reserved2', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["DataOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["DataLength"]'), ('Buffer', ':'))

class SMB2Write(Structure):
    SIZE = 48
    structure = (('StructureSize', '<H=49'), ('DataOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('Length', '<L=0'), ('Offset', '<Q=0'), ('FileID', ':', SMB2_FILEID), ('Channel', '<L=0'), ('RemainingBytes', '<L=0'), ('WriteChannelInfoOffset', '<H=0'), ('WriteChannelInfoLength', '<H=0'), ('_AlignPad', '_-AlignPad', 'self["DataOffset"] + self["WriteChannelInfoOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('Flags', '<L=0'), ('_Buffer', '_-Buffer', 'self["Length"]+self["WriteChannelInfoLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2Write_Response(Structure):
    structure = (('StructureSize', '<H=17'), ('Reserved', '<H=0'), ('Count', '<L=0'), ('Remaining', '<L=0'), ('WriteChannelInfoOffset', '<H=0'), ('WriteChannelInfoLength', '<H=0'))

class SMB2OplockBreakNotification(Structure):
    structure = (('StructureSize', '<H=24'), ('OplockLevel', '<B=0'), ('Reserved', '<B=0'), ('Reserved2', '<L=0'), ('FileID', ':', SMB2_FILEID))
SMB2OplockBreakAcknowledgment = SMB2OplockBreakNotification
SMB2OplockBreakResponse = SMB2OplockBreakNotification

class SMB2LeaseBreakNotification(Structure):
    structure = (('StructureSize', '<H=44'), ('NewEpoch', '<H=0'), ('Flags', '<L=0'), ('LeaseKey', '16s=""'), ('CurrentLeaseState', '<L=0'), ('NewLeaseState', '<L=0'), ('BreakReason', '<L=0'), ('AccessMaskHint', '<L=0'), ('ShareMaskHint', '<L=0'))

class SMB2LeaseBreakAcknowledgement(Structure):
    structure = (('StructureSize', '<H=36'), ('Reserved', '<H=0'), ('Flags', '<L=0'), ('LeaseKey', '16s=""'), ('LeaseState', '<L=0'), ('LeaseDuration', '<Q=0'))
SMB2LeaseBreakResponse = SMB2LeaseBreakAcknowledgement

class SMB2_LOCK_ELEMENT(Structure):
    structure = (('Offset', '<Q=0'), ('Length', '<Q=0'), ('Flags', '<L=0'), ('Reserved', '<L=0'))

class SMB2Lock(Structure):
    structure = (('StructureSize', '<H=48'), ('LockCount', '<H=0'), ('LockSequence', '<L=0'), ('FileID', ':', SMB2_FILEID), ('_Locks', '_-Locks', 'self["LockCount"]*24'), ('Locks', ':'))

class SMB2Lock_Response(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2Echo(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))
SMB2Echo_Response = SMB2Echo

class SMB2Cancel(Structure):
    structure = (('StructureSize', '<H=4'), ('Reserved', '<H=0'))

class SMB2Ioctl(Structure):
    SIZE = 56
    structure = (('StructureSize', '<H=57'), ('Reserved', '<H=0'), ('CtlCode', '<L=0'), ('FileID', ':', SMB2_FILEID), ('InputOffset', '<L=(self.SIZE + 64 + len(self["AlignPad"]))'), ('InputCount', '<L=0'), ('MaxInputResponse', '<L=0'), ('OutputOffset', '<L=(self.SIZE + 64 + len(self["AlignPad"]) + self["InputCount"])'), ('OutputCount', '<L=0'), ('MaxOutputResponse', '<L=0'), ('Flags', '<L=0'), ('Reserved2', '<L=0'), ('_Buffer', '_-Buffer', 'self["InputCount"]+self["OutputCount"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            while True:
                i = 10
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class FSCTL_PIPE_WAIT_STRUCTURE(Structure):
    structure = (('Timeout', '<q=0'), ('NameLength', '<L=0'), ('TimeoutSpecified', '<B=0'), ('Padding', '<B=0'), ('_Name', '_-Name', 'self["NameLength"]'), ('Name', ':'))

class SRV_COPYCHUNK_COPY(Structure):
    structure = (('SourceKey', '24s=""'), ('ChunkCount', '<L=0'), ('Reserved', '<L=0'), ('_Chunks', '_-Chunks', 'self["ChunkCount"]*len(SRV_COPYCHUNK)'), ('Chunks', ':'))

class SRV_COPYCHUNK(Structure):
    structure = (('SourceOffset', '<Q=0'), ('TargetOffset', '<Q=0'), ('Length', '<L=0'), ('Reserved', '<L=0'))

class SRV_COPYCHUNK_RESPONSE(Structure):
    structure = (('ChunksWritten', '<L=0'), ('ChunkBytesWritten', '<L=0'), ('TotalBytesWritten', '<L=0'))

class SRV_READ_HASH(Structure):
    structure = (('HashType', '<L=0'), ('HashVersion', '<L=0'), ('HashRetrievalType', '<L=0'), ('Length', '<L=0'), ('Offset', '<Q=0'))

class NETWORK_RESILIENCY_REQUEST(Structure):
    structure = (('Timeout', '<L=0'), ('Reserved', '<L=0'))

class VALIDATE_NEGOTIATE_INFO(Structure):
    structure = (('Capabilities', '<L=0'), ('Guid', '16s=""'), ('SecurityMode', '<H=0'), ('Dialects', '<H*<H'))

class SRV_SNAPSHOT_ARRAY(Structure):
    structure = (('NumberOfSnapShots', '<L=0'), ('NumberOfSnapShotsReturned', '<L=0'), ('SnapShotArraySize', '<L=0'), ('_SnapShots', '_-SnapShots', 'self["SnapShotArraySize"]'), ('SnapShots', ':'))

class SRV_REQUEST_RESUME_KEY(Structure):
    structure = (('ResumeKey', '24s=""'), ('ContextLength', '<L=0'), ('_Context', '_-Context', 'self["ContextLength"]'), ('Context', ':'))

class HASH_HEADER(Structure):
    structure = (('HashType', '<L=0'), ('HashVersion', '<L=0'), ('SourceFileChangeTime', '<Q=0'), ('SourceFileSize', '<Q=0'), ('HashBlobLength', '<L=0'), ('HashBlobOffset', '<L=0'), ('Dirty', '<H=0'), ('SourceFileNameLength', '<L=0'), ('_SourceFileName', '_-SourceFileName', 'self["SourceFileNameLength"]'), ('SourceFileName', ':'))

class SRV_HASH_RETRIEVE_HASH_BASED(Structure):
    structure = (('Offset', '<Q=0'), ('BufferLength', '<L=0'), ('Reserved', '<L=0'), ('_Buffer', '_-Buffer', 'self["BufferLength"]'), ('Buffer', ':'))

class SRV_HASH_RETRIEVE_FILE_BASED(Structure):
    structure = (('FileDataOffset', '<Q=0'), ('FileDataLength', '<Q=0'), ('BufferLength', '<L=0'), ('Reserved', '<L=0'), ('_Buffer', '_-Buffer', 'self["BufferLength"]'), ('Buffer', ':'))

class NETWORK_INTERFACE_INFO(Structure):
    structure = (('Next', '<L=0'), ('IfIndex', '<L=0'), ('Capability', '<L=0'), ('Reserved', '<L=0'), ('LinkSpeed', '<Q=0'), ('SockAddr_Storage', '128s=""'))

class SMB2Ioctl_Response(Structure):
    structure = (('StructureSize', '<H=49'), ('Reserved', '<H=0'), ('CtlCode', '<L=0'), ('FileID', ':', SMB2_FILEID), ('InputOffset', '<L=0'), ('InputCount', '<L=0'), ('OutputOffset', '<L=0'), ('OutputCount', '<L=0'), ('Flags', '<L=0'), ('Reserved2', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["OutputOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["InputCount"]+self["OutputCount"]'), ('Buffer', ':'))

class SMB2QueryDirectory(Structure):
    SIZE = 32
    structure = (('StructureSize', '<H=33'), ('FileInformationClass', '<B=0'), ('Flags', '<B=0'), ('FileIndex', '<L=0'), ('FileID', ':', SMB2_FILEID), ('FileNameOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('FileNameLength', '<H=0'), ('OutputBufferLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["FileNameOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["FileNameLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            for i in range(10):
                print('nop')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2QueryDirectory_Response(Structure):
    structure = (('StructureSize', '<H=9'), ('OutputBufferOffset', '<H=0'), ('OutputBufferLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["OutputBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["OutputBufferLength"]'), ('Buffer', ':'))

class SMB2ChangeNotify(Structure):
    structure = (('StructureSize', '<H=32'), ('Flags', '<H=0'), ('OutputBufferLength', '<L=0'), ('FileID', ':', SMB2_FILEID), ('CompletionFilter', '<L=0'), ('Reserved', '<L=0'))

class SMB2ChangeNotify_Response(Structure):
    structure = (('StructureSize', '<H=9'), ('OutputBufferOffset', '<H=0'), ('OutputBufferLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["OutputBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["OutputBufferLength"]'), ('Buffer', ':'))

class FILE_NOTIFY_INFORMATION(Structure):
    structure = (('NextEntryOffset', '<L=0'), ('Action', '<L=0'), ('FileNameLength', '<L=0'), ('_FileName', '_-FileName', 'self["FileNameLength"]'), ('FileName', ':'))

class SMB2QueryInfo(Structure):
    SIZE = 40
    structure = (('StructureSize', '<H=41'), ('InfoType', '<B=0'), ('FileInfoClass', '<B=0'), ('OutputBufferLength', '<L=0'), ('InputBufferOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('Reserved', '<H=0'), ('InputBufferLength', '<L=0'), ('AdditionalInformation', '<L=0'), ('Flags', '<L=0'), ('FileID', ':', SMB2_FILEID), ('_AlignPad', '_-AlignPad', 'self["InputBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["InputBufferLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            for i in range(10):
                print('nop')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2_QUERY_QUOTA_INFO(Structure):
    structure = (('ReturnSingle', '<B=0'), ('RestartScan', '<B=0'), ('Reserved', '<H=0'), ('SidListLength', '<L=0'), ('StartSidLength', '<L=0'), ('StartSidOffset', '<L=0'), ('SidBuffer', ':'))

class SMB2QueryInfo_Response(Structure):
    structure = (('StructureSize', '<H=9'), ('OutputBufferOffset', '<H=0'), ('OutputBufferLength', '<L=0'), ('_AlignPad', '_-AlignPad', 'self["OutputBufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["OutputBufferLength"]'), ('Buffer', ':'))

class SMB2SetInfo(Structure):
    SIZE = 32
    structure = (('StructureSize', '<H=33'), ('InfoType', '<B=0'), ('FileInfoClass', '<B=0'), ('BufferLength', '<L=0'), ('BufferOffset', '<H=(self.SIZE + 64 + len(self["AlignPad"]))'), ('Reserved', '<H=0'), ('AdditionalInformation', '<L=0'), ('FileID', ':', SMB2_FILEID), ('_AlignPad', '_-AlignPad', 'self["BufferOffset"] - (64 + self["StructureSize"] - 1)'), ('AlignPad', ':=""'), ('_Buffer', '_-Buffer', 'self["BufferLength"]'), ('Buffer', ':'))

    def __init__(self, data=None):
        if False:
            print('Hello World!')
        Structure.__init__(self, data)
        if data is None:
            self['AlignPad'] = ''

class SMB2SetInfo_Response(Structure):
    structure = (('StructureSize', '<H=2'),)

class FILE_RENAME_INFORMATION_TYPE_2(Structure):
    structure = (('ReplaceIfExists', '<B=0'), ('Reserved', '7s=""'), ('RootDirectory', '<Q=0'), ('FileNameLength', '<L=0'), ('_FileName', '_-FileName', 'self["FileNameLength"]'), ('FileName', ':'))

class SMB2_TRANSFORM_HEADER(Structure):
    structure = (('ProtocolID', '"ýSMB'), ('Signature', '16s=""'), ('Nonce', '16s=""'), ('OriginalMessageSize', '<L=0'), ('Reserved', '<H=0'), ('EncryptionAlgorithm', '<H=0'), ('SessionID', '<Q=0'))

class FileInternalInformation(Structure):
    structure = (('IndexNumber', '<q=0'),)