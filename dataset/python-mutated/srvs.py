from __future__ import division
from __future__ import print_function
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.dcerpc.v5.ndr import NDRCALL, NDR, NDRSTRUCT, NDRUNION, NDRPOINTER, NDRUniConformantArray, NDRUniFixedArray, NDRBOOLEAN, NDRUniConformantVaryingArray, PNDRUniConformantArray
from impacket.dcerpc.v5.dtypes import NULL, DWORD, LPWSTR, LPBYTE, LMSTR, ULONG, GUID, LPLONG, WSTR, SECURITY_INFORMATION, WCHAR
from impacket import system_errors
from impacket.uuid import uuidtup_to_bin
MSRPC_UUID_SRVS = uuidtup_to_bin(('4B324FC8-1670-01D3-1278-5A47BF6EE188', '3.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            while True:
                i = 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        key = self.error_code
        if key in system_errors.ERROR_MESSAGES:
            error_msg_short = system_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = system_errors.ERROR_MESSAGES[key][1]
            return 'SRVS SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'SRVS SessionError: unknown error code: 0x%x' % self.error_code
SRVSVC_HANDLE = WCHAR

class PSRVSVC_HANDLE(NDRPOINTER):
    referent = (('Data', SRVSVC_HANDLE),)

class SHARE_DEL_HANDLE(NDRSTRUCT):
    align = 1
    structure = (('Data', '20s=""'),)

class PSHARE_DEL_HANDLE(NDRPOINTER):
    referent = (('Data', SHARE_DEL_HANDLE),)
MAX_PREFERRED_LENGTH = -1
SESS_GUEST = 1
SESS_NOENCRYPTION = 2
STYPE_DISKTREE = 0
STYPE_PRINTQ = 1
STYPE_DEVICE = 2
STYPE_IPC = 3
STYPE_CLUSTER_FS = 33554432
STYPE_CLUSTER_SOFS = 67108864
STYPE_CLUSTER_DFS = 134217728
STYPE_SPECIAL = 2147483648
STYPE_TEMPORARY = 1073741824
STYPE_MASK = 255
CSC_CACHE_MANUAL_REINT = 0
CSC_CACHE_AUTO_REINT = 16
CSC_CACHE_VDO = 32
CSC_CACHE_NONE = 48
PLATFORM_ID_DOS = 300
PLATFORM_ID_OS2 = 400
PLATFORM_ID_NT = 500
PLATFORM_ID_OSF = 600
PLATFORM_ID_VMS = 700
SV_TYPE_WORKSTATION = 1
SV_TYPE_SERVER = 2
SV_TYPE_SQLSERVER = 4
SV_TYPE_DOMAIN_CTRL = 8
SV_TYPE_DOMAIN_BAKCTRL = 16
SV_TYPE_TIME_SOURCE = 32
SV_TYPE_AFP = 64
SV_TYPE_NOVELL = 128
SV_TYPE_DOMAIN_MEMBER = 256
SV_TYPE_LOCAL_LIST_ONLY = 1073741824
SV_TYPE_PRINTQ_SERVER = 512
SV_TYPE_DIALIN_SERVER = 1024
SV_TYPE_XENIX_SERVER = 2048
SV_TYPE_SERVER_MFPN = 16384
SV_TYPE_NT = 4096
SV_TYPE_WFW = 8192
SV_TYPE_SERVER_NT = 32768
SV_TYPE_POTENTIAL_BROWSER = 65536
SV_TYPE_BACKUP_BROWSER = 131072
SV_TYPE_MASTER_BROWSER = 262144
SV_TYPE_DOMAIN_MASTER = 524288
SV_TYPE_DOMAIN_ENUM = 2147483648
SV_TYPE_WINDOWS = 4194304
SV_TYPE_ALL = 4294967295
SV_TYPE_TERMINALSERVER = 33554432
SV_TYPE_CLUSTER_NT = 268435456
SV_TYPE_CLUSTER_VS_NT = 67108864
NAMETYPE_USER = 1
NAMETYPE_PASSWORD = 2
NAMETYPE_GROUP = 3
NAMETYPE_COMPUTER = 4
NAMETYPE_EVENT = 5
NAMETYPE_DOMAIN = 6
NAMETYPE_SERVICE = 7
NAMETYPE_NET = 8
NAMETYPE_SHARE = 9
NAMETYPE_MESSAGE = 10
NAMETYPE_MESSAGEDEST = 11
NAMETYPE_SHAREPASSWORD = 12
NAMETYPE_WORKGROUP = 13
ITYPE_UNC_COMPNAME = 4144
ITYPE_UNC_WC = 4145
ITYPE_UNC = 4096
ITYPE_UNC_WC_PATH = 4097
ITYPE_UNC_SYS_SEM = 6400
ITYPE_UNC_SYS_SHMEM = 6656
ITYPE_UNC_SYS_MSLOT = 6144
ITYPE_UNC_SYS_PIPE = 6912
ITYPE_UNC_SYS_QUEUE = 7680
ITYPE_PATH_ABSND = 8194
ITYPE_PATH_ABSD = 8198
ITYPE_PATH_RELND = 8192
ITYPE_PATH_RELD = 8196
ITYPE_PATH_ABSND_WC = 8195
ITYPE_PATH_ABSD_WC = 8199
ITYPE_PATH_RELND_WC = 8193
ITYPE_PATH_RELD_WC = 8197
ITYPE_PATH_SYS_SEM = 10498
ITYPE_PATH_SYS_SHMEM = 10754
ITYPE_PATH_SYS_MSLOT = 10242
ITYPE_PATH_SYS_PIPE = 11010
ITYPE_PATH_SYS_COMM = 11266
ITYPE_PATH_SYS_PRINT = 11522
ITYPE_PATH_SYS_QUEUE = 11778
ITYPE_PATH_SYS_SEM_M = 43266
ITYPE_PATH_SYS_SHMEM_M = 43522
ITYPE_PATH_SYS_MSLOT_M = 43010
ITYPE_PATH_SYS_PIPE_M = 43778
ITYPE_PATH_SYS_COMM_M = 44034
ITYPE_PATH_SYS_PRINT_M = 44290
ITYPE_PATH_SYS_QUEUE_M = 44546
ITYPE_DEVICE_DISK = 16384
ITYPE_DEVICE_LPT = 16400
ITYPE_DEVICE_COM = 16416
ITYPE_DEVICE_CON = 16448
ITYPE_DEVICE_NUL = 16464
SHARE_NETNAME_PARMNUM = 1
SHARE_TYPE_PARMNUM = 3
SHARE_REMARK_PARMNUM = 4
SHARE_PERMISSIONS_PARMNUM = 5
SHARE_MAX_USES_PARMNUM = 6
SHARE_CURRENT_USES_PARMNUM = 7
SHARE_PATH_PARMNUM = 8
SHARE_PASSWD_PARMNUM = 9
SHARE_FILE_SD_PARMNUM = 501
SV_PLATFORM_ID_PARMNUM = 101
SV_NAME_PARMNUM = 102
SV_VERSION_MAJOR_PARMNUM = 103
SV_VERSION_MINOR_PARMNUM = 104
SV_TYPE_PARMNUM = 105
SV_COMMENT_PARMNUM = 5
SV_USERS_PARMNUM = 107
SV_DISC_PARMNUM = 10
SV_HIDDEN_PARMNUM = 16
SV_ANNOUNCE_PARMNUM = 17
SV_ANNDELTA_PARMNUM = 18
SV_USERPATH_PARMNUM = 112
SV_SESSOPENS_PARMNUM = 501
SV_SESSVCS_PARMNUM = 502
SV_OPENSEARCH_PARMNUM = 503
SV_SIZREQBUF_PARMNUM = 504
SV_INITWORKITEMS_PARMNUM = 505
SV_MAXWORKITEMS_PARMNUM = 506
SV_RAWWORKITEMS_PARMNUM = 507
SV_IRPSTACKSIZE_PARMNUM = 508
SV_MAXRAWBUFLEN_PARMNUM = 509
SV_SESSUSERS_PARMNUM = 510
SV_SESSCONNS_PARMNUM = 511
SV_MAXNONPAGEDMEMORYUSAGE_PARMNUM = 512
SV_MAXPAGEDMEMORYUSAGE_PARMNUM = 513
SV_ENABLESOFTCOMPAT_PARMNUM = 514
SV_ENABLEFORCEDLOGOFF_PARMNUM = 515
SV_TIMESOURCE_PARMNUM = 516
SV_ACCEPTDOWNLEVELAPIS_PARMNUM = 517
SV_LMANNOUNCE_PARMNUM = 518
SV_DOMAIN_PARMNUM = 519
SV_MAXCOPYREADLEN_PARMNUM = 520
SV_MAXCOPYWRITELEN_PARMNUM = 521
SV_MINKEEPSEARCH_PARMNUM = 522
SV_MAXKEEPSEARCH_PARMNUM = 523
SV_MINKEEPCOMPLSEARCH_PARMNUM = 524
SV_MAXKEEPCOMPLSEARCH_PARMNUM = 525
SV_THREADCOUNTADD_PARMNUM = 526
SV_NUMBLOCKTHREADS_PARMNUM = 527
SV_SCAVTIMEOUT_PARMNUM = 528
SV_MINRCVQUEUE_PARMNUM = 529
SV_MINFREEWORKITEMS_PARMNUM = 530
SV_XACTMEMSIZE_PARMNUM = 531
SV_THREADPRIORITY_PARMNUM = 532
SV_MAXMPXCT_PARMNUM = 533
SV_OPLOCKBREAKWAIT_PARMNUM = 534
SV_OPLOCKBREAKRESPONSEWAIT_PARMNUM = 535
SV_ENABLEOPLOCKS_PARMNUM = 536
SV_ENABLEOPLOCKFORCECLOSE_PARMNUM = 537
SV_ENABLEFCBOPENS_PARMNUM = 538
SV_ENABLERAW_PARMNUM = 539
SV_ENABLESHAREDNETDRIVES_PARMNUM = 540
SV_MINFREECONNECTIONS_PARMNUM = 541
SV_MAXFREECONNECTIONS_PARMNUM = 542
SV_INITSESSTABLE_PARMNUM = 543
SV_INITCONNTABLE_PARMNUM = 544
SV_INITFILETABLE_PARMNUM = 545
SV_INITSEARCHTABLE_PARMNUM = 546
SV_ALERTSCHEDULE_PARMNUM = 547
SV_ERRORTHRESHOLD_PARMNUM = 548
SV_NETWORKERRORTHRESHOLD_PARMNUM = 549
SV_DISKSPACETHRESHOLD_PARMNUM = 550
SV_MAXLINKDELAY_PARMNUM = 552
SV_MINLINKTHROUGHPUT_PARMNUM = 553
SV_LINKINFOVALIDTIME_PARMNUM = 554
SV_SCAVQOSINFOUPDATETIME_PARMNUM = 555
SV_MAXWORKITEMIDLETIME_PARMNUM = 556
PKT_ENTRY_TYPE_CAIRO = 1
PKT_ENTRY_TYPE_MACHINE = 2
PKT_ENTRY_TYPE_NONCAIRO = 4
PKT_ENTRY_TYPE_LEAFONLY = 8
PKT_ENTRY_TYPE_OUTSIDE_MY_DOM = 16
PKT_ENTRY_TYPE_INSITE_ONLY = 32
PKT_ENTRY_TYPE_REFERRAL_SVC = 128
PKT_ENTRY_TYPE_PERMANENT = 256
PKT_ENTRY_TYPE_LOCAL = 1024
PKT_ENTRY_TYPE_LOCAL_XPOINT = 2048
PKT_ENTRY_TYPE_MACH_SHARE = 4096
PKT_ENTRY_TYPE_OFFLINE = 8192
PERM_FILE_READ = 1
PERM_FILE_WRITE = 2
PERM_FILE_CREATE = 4
ACCESS_EXEC = 8
ACCESS_DELETE = 16
ACCESS_ATRIB = 32
ACCESS_PERM = 64
SHI1005_FLAGS_DFS = 1
SHI1005_FLAGS_DFS_ROOT = 2
CSC_MASK = 48
SHI1005_FLAGS_RESTRICT_EXCLUSIVE_OPENS = 256
SHI1005_FLAGS_FORCE_SHARED_DELETE = 512
SHI1005_FLAGS_ALLOW_NAMESPACE_CACHING = 1024
SHI1005_FLAGS_ACCESS_BASED_DIRECTORY_ENUM = 2048
SHI1005_FLAGS_FORCE_LEVELII_OPLOCK = 4096
SHI1005_FLAGS_ENABLE_HASH = 8192
SHI1005_FLAGS_ENABLE_CA = 16384
SHI1005_FLAGS_ENCRYPT_DATA = 32768
SRV_SUPPORT_HASH_GENERATION = 1
SRV_HASH_GENERATION_ACTIVE = 2
SVTI2_REMAP_PIPE_NAMES = 2
SVTI2_SCOPED_NAME = 4
DFS_SITE_PRIMARY = 1
DFS_SERVICE_TYPE_MASTER = 1
DFS_SERVICE_TYPE_READONLY = 2
DFS_SERVICE_TYPE_LOCAL = 4
DFS_SERVICE_TYPE_REFERRAL = 8
DFS_SERVICE_TYPE_ACTIVE = 16
DFS_SERVICE_TYPE_DOWN_LEVEL = 32
DFS_SERVICE_TYPE_COSTLIER = 64
DFS_SERVICE_TYPE_OFFLINE = 128
FILE_SUPERSEDE = 0
FILE_OPEN = 1
FILE_CREATE = 2

class CONNECTION_INFO_0(NDRSTRUCT):
    structure = (('coni0_id', DWORD),)

class CONNECTION_INFO_0_ARRAY(NDRUniConformantArray):
    item = CONNECTION_INFO_0

class LPCONNECTION_INFO_0_ARRAY(NDRPOINTER):
    referent = (('Data', CONNECTION_INFO_0_ARRAY),)

class CONNECTION_INFO_1(NDRSTRUCT):
    structure = (('coni1_id', DWORD), ('coni1_type', DWORD), ('coni1_num_opens', DWORD), ('coni1_num_users', DWORD), ('coni1_time', DWORD), ('coni1_username', LPWSTR), ('coni1_netname', LPWSTR))

class CONNECTION_INFO_1_ARRAY(NDRUniConformantArray):
    item = CONNECTION_INFO_1

class LPCONNECTION_INFO_1_ARRAY(NDRPOINTER):
    referent = (('Data', CONNECTION_INFO_1_ARRAY),)

class CONNECT_INFO_0_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPCONNECTION_INFO_0_ARRAY))

class LPCONNECT_INFO_0_CONTAINER(NDRPOINTER):
    referent = (('Data', CONNECT_INFO_0_CONTAINER),)

class CONNECT_INFO_1_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPCONNECTION_INFO_1_ARRAY))

class LPCONNECT_INFO_1_CONTAINER(NDRPOINTER):
    referent = (('Data', CONNECT_INFO_1_CONTAINER),)

class CONNECT_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Level0', LPCONNECT_INFO_0_CONTAINER), 1: ('Level1', LPCONNECT_INFO_1_CONTAINER)}

class CONNECT_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('ConnectInfo', CONNECT_ENUM_UNION))

class FILE_INFO_2(NDRSTRUCT):
    structure = (('fi2_id', DWORD),)

class LPFILE_INFO_2(NDRPOINTER):
    referent = (('Data', FILE_INFO_2),)

class FILE_INFO_2_ARRAY(NDRUniConformantArray):
    item = FILE_INFO_2

class LPFILE_INFO_2_ARRAY(NDRPOINTER):
    referent = (('Data', FILE_INFO_2_ARRAY),)

class FILE_INFO_3(NDRSTRUCT):
    structure = (('fi3_id', DWORD), ('fi3_permissions', DWORD), ('fi3_num_locks', DWORD), ('fi3_path_name', LPWSTR), ('fi3_username', LPWSTR))

class LPFILE_INFO_3(NDRPOINTER):
    referent = (('Data', FILE_INFO_3),)

class FILE_INFO_3_ARRAY(NDRUniConformantArray):
    item = FILE_INFO_3

class LPFILE_INFO_3_ARRAY(NDRPOINTER):
    referent = (('Data', FILE_INFO_3_ARRAY),)

class FILE_INFO_2_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPFILE_INFO_2_ARRAY))

class LPFILE_INFO_2_CONTAINER(NDRPOINTER):
    referent = (('Data', FILE_INFO_2_CONTAINER),)

class FILE_INFO_3_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPFILE_INFO_3_ARRAY))

class LPFILE_INFO_3_CONTAINER(NDRPOINTER):
    referent = (('Data', FILE_INFO_3_CONTAINER),)

class FILE_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {2: ('Level2', LPFILE_INFO_2_CONTAINER), 3: ('Level3', LPFILE_INFO_3_CONTAINER)}

class FILE_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('FileInfo', FILE_ENUM_UNION))

class SESSION_INFO_0(NDRSTRUCT):
    structure = (('sesi0_cname', LPWSTR),)

class LPSESSION_INFO_0(NDRPOINTER):
    referent = (('Data', SESSION_INFO_0),)

class SESSION_INFO_0_ARRAY(NDRUniConformantArray):
    item = SESSION_INFO_0

class LPSESSION_INFO_0_ARRAY(NDRPOINTER):
    referent = (('Data', SESSION_INFO_0_ARRAY),)

class SESSION_INFO_1(NDRSTRUCT):
    structure = (('sesi1_cname', LPWSTR), ('sesi1_username', LPWSTR), ('sesi1_num_opens', DWORD), ('sesi1_time', DWORD), ('sesi1_idle_time', DWORD), ('sesi1_user_flags', DWORD))

class LPSESSION_INFO_1(NDRPOINTER):
    referent = (('Data', SESSION_INFO_1),)

class SESSION_INFO_1_ARRAY(NDRUniConformantArray):
    item = SESSION_INFO_1

class LPSESSION_INFO_1_ARRAY(NDRPOINTER):
    referent = (('Data', SESSION_INFO_1_ARRAY),)

class SESSION_INFO_2(NDRSTRUCT):
    structure = (('sesi2_cname', LPWSTR), ('sesi2_username', LPWSTR), ('sesi2_num_opens', DWORD), ('sesi2_time', DWORD), ('sesi2_idle_time', DWORD), ('sesi2_user_flags', DWORD), ('sesi2_cltype_name', LPWSTR))

class LPSESSION_INFO_2(NDRPOINTER):
    referent = (('Data', SESSION_INFO_2),)

class SESSION_INFO_2_ARRAY(NDRUniConformantArray):
    item = SESSION_INFO_2

class LPSESSION_INFO_2_ARRAY(NDRPOINTER):
    referent = (('Data', SESSION_INFO_2_ARRAY),)

class SESSION_INFO_10(NDRSTRUCT):
    structure = (('sesi10_cname', LPWSTR), ('sesi10_username', LPWSTR), ('sesi10_time', DWORD), ('sesi10_idle_time', DWORD))

class LPSESSION_INFO_10(NDRPOINTER):
    referent = (('Data', SESSION_INFO_10),)

class SESSION_INFO_10_ARRAY(NDRUniConformantArray):
    item = SESSION_INFO_10

class LPSESSION_INFO_10_ARRAY(NDRPOINTER):
    referent = (('Data', SESSION_INFO_10_ARRAY),)

class SESSION_INFO_502(NDRSTRUCT):
    structure = (('sesi502_cname', LPWSTR), ('sesi502_username', LPWSTR), ('sesi502_num_opens', DWORD), ('sesi502_time', DWORD), ('sesi502_idle_time', DWORD), ('sesi502_user_flags', DWORD), ('sesi502_cltype_name', LPWSTR), ('sesi502_transport', LPWSTR))

class LPSESSION_INFO_502(NDRPOINTER):
    referent = (('Data', SESSION_INFO_502),)

class SESSION_INFO_502_ARRAY(NDRUniConformantArray):
    item = SESSION_INFO_502

class LPSESSION_INFO_502_ARRAY(NDRPOINTER):
    referent = (('Data', SESSION_INFO_502_ARRAY),)

class SESSION_INFO_0_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSESSION_INFO_0_ARRAY))

class LPSESSION_INFO_0_CONTAINER(NDRPOINTER):
    referent = (('Data', SESSION_INFO_0_CONTAINER),)

class SESSION_INFO_1_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSESSION_INFO_1_ARRAY))

class LPSESSION_INFO_1_CONTAINER(NDRPOINTER):
    referent = (('Data', SESSION_INFO_1_CONTAINER),)

class SESSION_INFO_2_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSESSION_INFO_2_ARRAY))

class LPSESSION_INFO_2_CONTAINER(NDRPOINTER):
    referent = (('Data', SESSION_INFO_2_CONTAINER),)

class SESSION_INFO_10_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSESSION_INFO_10_ARRAY))

class LPSESSION_INFO_10_CONTAINER(NDRPOINTER):
    referent = (('Data', SESSION_INFO_10_CONTAINER),)

class SESSION_INFO_502_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSESSION_INFO_502_ARRAY))

class LPSESSION_INFO_502_CONTAINER(NDRPOINTER):
    referent = (('Data', SESSION_INFO_502_CONTAINER),)

class SESSION_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Level0', LPSESSION_INFO_0_CONTAINER), 1: ('Level1', LPSESSION_INFO_1_CONTAINER), 2: ('Level2', LPSESSION_INFO_2_CONTAINER), 10: ('Level10', LPSESSION_INFO_10_CONTAINER), 502: ('Level502', LPSESSION_INFO_502_CONTAINER)}

class SESSION_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('SessionInfo', SESSION_ENUM_UNION))

class SHARE_INFO_0(NDRSTRUCT):
    structure = (('shi0_netname', LPWSTR),)

class LPSHARE_INFO_0(NDRPOINTER):
    referent = (('Data', SHARE_INFO_0),)

class SHARE_INFO_0_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_0

class LPSHARE_INFO_0_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_0_ARRAY),)

class SHARE_INFO_1(NDRSTRUCT):
    structure = (('shi1_netname', LPWSTR), ('shi1_type', DWORD), ('shi1_remark', LPWSTR))

class LPSHARE_INFO_1(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1),)

class SHARE_INFO_1_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_1

class LPSHARE_INFO_1_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1_ARRAY),)

class SHARE_INFO_2(NDRSTRUCT):
    structure = (('shi2_netname', LPWSTR), ('shi2_type', DWORD), ('shi2_remark', LPWSTR), ('shi2_permissions', DWORD), ('shi2_max_uses', DWORD), ('shi2_current_uses', DWORD), ('shi2_path', LPWSTR), ('shi2_passwd', LPWSTR))

class LPSHARE_INFO_2(NDRPOINTER):
    referent = (('Data', SHARE_INFO_2),)

class SHARE_INFO_2_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_2

class LPSHARE_INFO_2_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_2_ARRAY),)

class SHARE_INFO_501(NDRSTRUCT):
    structure = (('shi501_netname', LPWSTR), ('shi501_type', DWORD), ('shi501_remark', LPWSTR), ('shi501_flags', DWORD))

class LPSHARE_INFO_501(NDRPOINTER):
    referent = (('Data', SHARE_INFO_501),)

class SHARE_INFO_501_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_501

class LPSHARE_INFO_501_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_501_ARRAY),)

class SHARE_INFO_502(NDRSTRUCT):
    structure = (('shi502_netname', LPWSTR), ('shi502_type', DWORD), ('shi502_remark', LPWSTR), ('shi502_permissions', DWORD), ('shi502_max_uses', DWORD), ('shi502_current_uses', DWORD), ('shi502_path', LPWSTR), ('shi502_passwd', LPWSTR), ('shi502_reserved', DWORD), ('shi502_security_descriptor', LPBYTE))

class LPSHARE_INFO_502(NDRPOINTER):
    referent = (('Data', SHARE_INFO_502),)

class SHARE_INFO_502_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_502

class LPSHARE_INFO_502_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_502_ARRAY),)

class SHARE_INFO_503(NDRSTRUCT):
    structure = (('shi503_netname', LPWSTR), ('shi503_type', DWORD), ('shi503_remark', LPWSTR), ('shi503_permissions', DWORD), ('shi503_max_uses', DWORD), ('shi503_current_uses', DWORD), ('shi503_path', LPWSTR), ('shi503_passwd', LPWSTR), ('shi503_servername', LPWSTR), ('shi503_reserved', DWORD), ('shi503_security_descriptor', LPBYTE))

class LPSHARE_INFO_503(NDRPOINTER):
    referent = (('Data', SHARE_INFO_503),)

class SHARE_INFO_503_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_503

class LPSHARE_INFO_503_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_503_ARRAY),)

class SHARE_INFO_1004(NDRSTRUCT):
    structure = (('shi1004_remark', LPWSTR),)

class LPSHARE_INFO_1004(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1004),)

class SHARE_INFO_1004_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_1004

class LPSHARE_INFO_1004_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1004_ARRAY),)

class SHARE_INFO_1005(NDRSTRUCT):
    structure = (('shi1005_flags', DWORD),)

class LPSHARE_INFO_1005(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1005),)

class SHARE_INFO_1005_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_1004

class LPSHARE_INFO_1005_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1005_ARRAY),)

class SHARE_INFO_1006(NDRSTRUCT):
    structure = (('shi1006_max_uses', DWORD),)

class LPSHARE_INFO_1006(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1006),)

class SHARE_INFO_1006_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_1006

class LPSHARE_INFO_1006_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1006_ARRAY),)

class SHARE_INFO_1501(NDRSTRUCT):
    structure = (('shi1501_reserved', DWORD), ('shi1501_security_descriptor', NDRUniConformantArray))

class LPSHARE_INFO_1501(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1501),)

class SHARE_INFO_1501_ARRAY(NDRUniConformantArray):
    item = SHARE_INFO_1501

class LPSHARE_INFO_1501_ARRAY(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1501_ARRAY),)

class SHARE_INFO_0_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_0_ARRAY))

class LPSHARE_INFO_0_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_0_CONTAINER),)

class SHARE_INFO_1_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_1_ARRAY))

class LPSHARE_INFO_1_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_1_CONTAINER),)

class SHARE_INFO_2_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_2_ARRAY))

class LPSHARE_INFO_2_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_2_CONTAINER),)

class SHARE_INFO_501_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_501_ARRAY))

class LPSHARE_INFO_501_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_501_CONTAINER),)

class SHARE_INFO_502_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_502_ARRAY))

class LPSHARE_INFO_502_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_502_CONTAINER),)

class SHARE_INFO_503_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSHARE_INFO_503_ARRAY))

class LPSHARE_INFO_503_CONTAINER(NDRPOINTER):
    referent = (('Data', SHARE_INFO_503_CONTAINER),)

class SHARE_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Level0', LPSHARE_INFO_0_CONTAINER), 1: ('Level1', LPSHARE_INFO_1_CONTAINER), 2: ('Level2', LPSHARE_INFO_2_CONTAINER), 501: ('Level501', LPSHARE_INFO_501_CONTAINER), 502: ('Level502', LPSHARE_INFO_502_CONTAINER), 503: ('Level503', LPSHARE_INFO_503_CONTAINER)}

class SHARE_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('ShareInfo', SHARE_ENUM_UNION))

class STAT_SERVER_0(NDRSTRUCT):
    structure = (('sts0_start', DWORD), ('sts0_fopens', DWORD), ('sts0_devopens', DWORD), ('sts0_jobsqueued', DWORD), ('sts0_sopens', DWORD), ('sts0_stimedout', DWORD), ('sts0_serrorout', DWORD), ('sts0_pwerrors', DWORD), ('sts0_permerrors', DWORD), ('sts0_syserrors', DWORD), ('sts0_bytessent_low', DWORD), ('sts0_bytessent_high', DWORD), ('sts0_bytesrcvd_low', DWORD), ('sts0_bytesrcvd_high', DWORD), ('sts0_avresponse', DWORD), ('sts0_reqbufneed', DWORD), ('sts0_bigbufneed', DWORD))

class LPSTAT_SERVER_0(NDRPOINTER):
    referent = (('Data', STAT_SERVER_0),)

class SERVER_INFO_100(NDRSTRUCT):
    structure = (('sv100_platform_id', DWORD), ('sv100_name', LPWSTR))

class LPSERVER_INFO_100(NDRPOINTER):
    referent = (('Data', SERVER_INFO_100),)

class SERVER_INFO_101(NDRSTRUCT):
    structure = (('sv101_platform_id', DWORD), ('sv101_name', LPWSTR), ('sv101_version_major', DWORD), ('sv101_version_minor', DWORD), ('sv101_type', DWORD), ('sv101_comment', LPWSTR))

class LPSERVER_INFO_101(NDRPOINTER):
    referent = (('Data', SERVER_INFO_101),)

class SERVER_INFO_102(NDRSTRUCT):
    structure = (('sv102_platform_id', DWORD), ('sv102_name', LPWSTR), ('sv102_version_major', DWORD), ('sv102_version_minor', DWORD), ('sv102_type', DWORD), ('sv102_comment', LPWSTR), ('sv102_users', DWORD), ('sv102_disc', DWORD), ('sv102_hidden', DWORD), ('sv102_announce', DWORD), ('sv102_anndelta', DWORD), ('sv102_licenses', DWORD), ('sv102_userpath', LPWSTR))

class LPSERVER_INFO_102(NDRPOINTER):
    referent = (('Data', SERVER_INFO_102),)

class SERVER_INFO_103(NDRSTRUCT):
    structure = (('sv103_platform_id', DWORD), ('sv103_name', LPWSTR), ('sv103_version_major', DWORD), ('sv103_version_minor', DWORD), ('sv103_type', DWORD), ('sv103_comment', LPWSTR), ('sv103_users', DWORD), ('sv103_disc', DWORD), ('sv103_hidden', DWORD), ('sv103_announce', DWORD), ('sv103_anndelta', DWORD), ('sv103_licenses', DWORD), ('sv103_userpath', LPWSTR), ('sv103_capabilities', DWORD))

class LPSERVER_INFO_103(NDRPOINTER):
    referent = (('Data', SERVER_INFO_103),)

class SERVER_INFO_502(NDRSTRUCT):
    structure = (('sv502_sessopens', DWORD), ('sv502_sessvcs', DWORD), ('sv502_opensearch', DWORD), ('sv502_sizreqbuf', DWORD), ('sv502_initworkitems', DWORD), ('sv502_maxworkitems', DWORD), ('sv502_rawworkitems', DWORD), ('sv502_irpstacksize', DWORD), ('sv502_maxrawbuflen', DWORD), ('sv502_sessusers', DWORD), ('sv502_sessconns', DWORD), ('sv502_maxpagedmemoryusage', DWORD), ('sv502_maxnonpagedmemoryusage', DWORD), ('sv502_enablesoftcompat', DWORD), ('sv502_enableforcedlogoff', DWORD), ('sv502_timesource', DWORD), ('sv502_acceptdownlevelapis', DWORD), ('sv502_lmannounce', DWORD))

class LPSERVER_INFO_502(NDRPOINTER):
    referent = (('Data', SERVER_INFO_502),)

class SERVER_INFO_503(NDRSTRUCT):
    structure = (('sv503_sessopens', DWORD), ('sv503_sessvcs', DWORD), ('sv503_opensearch', DWORD), ('sv503_sizreqbuf', DWORD), ('sv503_initworkitems', DWORD), ('sv503_maxworkitems', DWORD), ('sv503_rawworkitems', DWORD), ('sv503_irpstacksize', DWORD), ('sv503_maxrawbuflen', DWORD), ('sv503_sessusers', DWORD), ('sv503_sessconns', DWORD), ('sv503_maxpagedmemoryusage', DWORD), ('sv503_maxnonpagedmemoryusage', DWORD), ('sv503_enablesoftcompat', DWORD), ('sv503_enableforcedlogoff', DWORD), ('sv503_timesource', DWORD), ('sv503_acceptdownlevelapis', DWORD), ('sv503_lmannounce', DWORD), ('sv503_domain', LPWSTR), ('sv503_maxcopyreadlen', DWORD), ('sv503_maxcopywritelen', DWORD), ('sv503_minkeepsearch', DWORD), ('sv503_maxkeepsearch', DWORD), ('sv503_minkeepcomplsearch', DWORD), ('sv503_maxkeepcomplsearch', DWORD), ('sv503_threadcountadd', DWORD), ('sv503_numblockthreads', DWORD), ('sv503_scavtimeout', DWORD), ('sv503_minrcvqueue', DWORD), ('sv503_minfreeworkitems', DWORD), ('sv503_xactmemsize', DWORD), ('sv503_threadpriority', DWORD), ('sv503_maxmpxct', DWORD), ('sv503_oplockbreakwait', DWORD), ('sv503_oplockbreakresponsewait', DWORD), ('sv503_enableoplocks', DWORD), ('sv503_enableoplockforceclose', DWORD), ('sv503_enablefcbopens', DWORD), ('sv503_enableraw', DWORD), ('sv503_enablesharednetdrives', DWORD), ('sv503_minfreeconnections', DWORD), ('sv503_maxfreeconnections', DWORD))

class LPSERVER_INFO_503(NDRPOINTER):
    referent = (('Data', SERVER_INFO_503),)

class SERVER_INFO_599(NDRSTRUCT):
    structure = (('sv599_sessopens', DWORD), ('sv599_sessvcs', DWORD), ('sv599_opensearch', DWORD), ('sv599_sizreqbuf', DWORD), ('sv599_initworkitems', DWORD), ('sv599_maxworkitems', DWORD), ('sv599_rawworkitems', DWORD), ('sv599_irpstacksize', DWORD), ('sv599_maxrawbuflen', DWORD), ('sv599_sessusers', DWORD), ('sv599_sessconns', DWORD), ('sv599_maxpagedmemoryusage', DWORD), ('sv599_maxnonpagedmemoryusage', DWORD), ('sv599_enablesoftcompat', DWORD), ('sv599_enableforcedlogoff', DWORD), ('sv599_timesource', DWORD), ('sv599_acceptdownlevelapis', DWORD), ('sv599_lmannounce', DWORD), ('sv599_domain', LPWSTR), ('sv599_maxcopyreadlen', DWORD), ('sv599_maxcopywritelen', DWORD), ('sv599_minkeepsearch', DWORD), ('sv599_maxkeepsearch', DWORD), ('sv599_minkeepcomplsearch', DWORD), ('sv599_maxkeepcomplsearch', DWORD), ('sv599_threadcountadd', DWORD), ('sv599_numblockthreads', DWORD), ('sv599_scavtimeout', DWORD), ('sv599_minrcvqueue', DWORD), ('sv599_minfreeworkitems', DWORD), ('sv599_xactmemsize', DWORD), ('sv599_threadpriority', DWORD), ('sv599_maxmpxct', DWORD), ('sv599_oplockbreakwait', DWORD), ('sv599_oplockbreakresponsewait', DWORD), ('sv599_enableoplocks', DWORD), ('sv599_enableoplockforceclose', DWORD), ('sv599_enablefcbopens', DWORD), ('sv599_enableraw', DWORD), ('sv599_enablesharednetdrives', DWORD), ('sv599_minfreeconnections', DWORD), ('sv599_maxfreeconnections', DWORD), ('sv599_initsesstable', DWORD), ('sv599_initconntable', DWORD), ('sv599_initfiletable', DWORD), ('sv599_initsearchtable', DWORD), ('sv599_alertschedule', DWORD), ('sv599_errorthreshold', DWORD), ('sv599_networkerrorthreshold', DWORD), ('sv599_diskspacethreshold', DWORD), ('sv599_reserved', DWORD), ('sv599_maxlinkdelay', DWORD), ('sv599_minlinkthroughput', DWORD), ('sv599_linkinfovalidtime', DWORD), ('sv599_scavqosinfoupdatetime', DWORD), ('sv599_maxworkitemidletime', DWORD))

class LPSERVER_INFO_599(NDRPOINTER):
    referent = (('Data', SERVER_INFO_599),)

class SERVER_INFO_1005(NDRSTRUCT):
    structure = (('sv1005_comment', LPWSTR),)

class LPSERVER_INFO_1005(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1005),)

class SERVER_INFO_1107(NDRSTRUCT):
    structure = (('sv1107_users', DWORD),)

class LPSERVER_INFO_1107(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1107),)

class SERVER_INFO_1010(NDRSTRUCT):
    structure = (('sv1010_disc', DWORD),)

class LPSERVER_INFO_1010(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1010),)

class SERVER_INFO_1016(NDRSTRUCT):
    structure = (('sv1016_hidden', DWORD),)

class LPSERVER_INFO_1016(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1016),)

class SERVER_INFO_1017(NDRSTRUCT):
    structure = (('sv1017_announce', DWORD),)

class LPSERVER_INFO_1017(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1017),)

class SERVER_INFO_1018(NDRSTRUCT):
    structure = (('sv1018_anndelta', DWORD),)

class LPSERVER_INFO_1018(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1018),)

class SERVER_INFO_1501(NDRSTRUCT):
    structure = (('sv1501_sessopens', DWORD),)

class LPSERVER_INFO_1501(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1501),)

class SERVER_INFO_1502(NDRSTRUCT):
    structure = (('sv1502_sessvcs', DWORD),)

class LPSERVER_INFO_1502(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1502),)

class SERVER_INFO_1503(NDRSTRUCT):
    structure = (('sv1503_opensearch', DWORD),)

class LPSERVER_INFO_1503(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1503),)

class SERVER_INFO_1506(NDRSTRUCT):
    structure = (('sv1506_maxworkitems', DWORD),)

class LPSERVER_INFO_1506(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1506),)

class SERVER_INFO_1510(NDRSTRUCT):
    structure = (('sv1510_sessusers', DWORD),)

class LPSERVER_INFO_1510(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1510),)

class SERVER_INFO_1511(NDRSTRUCT):
    structure = (('sv1511_sessconns', DWORD),)

class LPSERVER_INFO_1511(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1511),)

class SERVER_INFO_1512(NDRSTRUCT):
    structure = (('sv1512_maxnonpagedmemoryusage', DWORD),)

class LPSERVER_INFO_1512(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1512),)

class SERVER_INFO_1513(NDRSTRUCT):
    structure = (('sv1513_maxpagedmemoryusage', DWORD),)

class LPSERVER_INFO_1513(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1513),)

class SERVER_INFO_1514(NDRSTRUCT):
    structure = (('sv1514_enablesoftcompat', DWORD),)

class LPSERVER_INFO_1514(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1514),)

class SERVER_INFO_1515(NDRSTRUCT):
    structure = (('sv1515_enableforcedlogoff', DWORD),)

class LPSERVER_INFO_1515(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1515),)

class SERVER_INFO_1516(NDRSTRUCT):
    structure = (('sv1516_timesource', DWORD),)

class LPSERVER_INFO_1516(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1516),)

class SERVER_INFO_1518(NDRSTRUCT):
    structure = (('sv1518_lmannounce', DWORD),)

class LPSERVER_INFO_1518(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1518),)

class SERVER_INFO_1523(NDRSTRUCT):
    structure = (('sv1523_maxkeepsearch', DWORD),)

class LPSERVER_INFO_1523(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1523),)

class SERVER_INFO_1528(NDRSTRUCT):
    structure = (('sv1528_scavtimeout', DWORD),)

class LPSERVER_INFO_1528(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1528),)

class SERVER_INFO_1529(NDRSTRUCT):
    structure = (('sv1529_minrcvqueue', DWORD),)

class LPSERVER_INFO_1529(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1529),)

class SERVER_INFO_1530(NDRSTRUCT):
    structure = (('sv1530_minfreeworkitems', DWORD),)

class LPSERVER_INFO_1530(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1530),)

class SERVER_INFO_1533(NDRSTRUCT):
    structure = (('sv1533_maxmpxct', DWORD),)

class LPSERVER_INFO_1533(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1533),)

class SERVER_INFO_1534(NDRSTRUCT):
    structure = (('sv1534_oplockbreakwait', DWORD),)

class LPSERVER_INFO_1534(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1534),)

class SERVER_INFO_1535(NDRSTRUCT):
    structure = (('sv1535_oplockbreakresponsewait', DWORD),)

class LPSERVER_INFO_1535(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1535),)

class SERVER_INFO_1536(NDRSTRUCT):
    structure = (('sv1536_enableoplocks', DWORD),)

class LPSERVER_INFO_1536(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1536),)

class SERVER_INFO_1538(NDRSTRUCT):
    structure = (('sv1538_enablefcbopens', DWORD),)

class LPSERVER_INFO_1538(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1538),)

class SERVER_INFO_1539(NDRSTRUCT):
    structure = (('sv1539_enableraw', DWORD),)

class LPSERVER_INFO_1539(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1539),)

class SERVER_INFO_1540(NDRSTRUCT):
    structure = (('sv1540_enablesharednetdrives', DWORD),)

class LPSERVER_INFO_1540(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1540),)

class SERVER_INFO_1541(NDRSTRUCT):
    structure = (('sv1541_minfreeconnections', DWORD),)

class LPSERVER_INFO_1541(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1541),)

class SERVER_INFO_1542(NDRSTRUCT):
    structure = (('sv1542_maxfreeconnections', DWORD),)

class LPSERVER_INFO_1542(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1542),)

class SERVER_INFO_1543(NDRSTRUCT):
    structure = (('sv1543_initsesstable', DWORD),)

class LPSERVER_INFO_1543(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1543),)

class SERVER_INFO_1544(NDRSTRUCT):
    structure = (('sv1544_initconntable', DWORD),)

class LPSERVER_INFO_1544(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1544),)

class SERVER_INFO_1545(NDRSTRUCT):
    structure = (('sv1545_initfiletable', DWORD),)

class LPSERVER_INFO_1545(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1545),)

class SERVER_INFO_1546(NDRSTRUCT):
    structure = (('sv1546_initsearchtable', DWORD),)

class LPSERVER_INFO_1546(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1546),)

class SERVER_INFO_1547(NDRSTRUCT):
    structure = (('sv1547_alertschedule', DWORD),)

class LPSERVER_INFO_1547(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1547),)

class SERVER_INFO_1548(NDRSTRUCT):
    structure = (('sv1548_errorthreshold', DWORD),)

class LPSERVER_INFO_1548(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1548),)

class SERVER_INFO_1549(NDRSTRUCT):
    structure = (('sv1549_networkerrorthreshold', DWORD),)

class LPSERVER_INFO_1549(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1549),)

class SERVER_INFO_1550(NDRSTRUCT):
    structure = (('sv1550_diskspacethreshold', DWORD),)

class LPSERVER_INFO_1550(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1550),)

class SERVER_INFO_1552(NDRSTRUCT):
    structure = (('sv1552_maxlinkdelay', DWORD),)

class LPSERVER_INFO_1552(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1552),)

class SERVER_INFO_1553(NDRSTRUCT):
    structure = (('sv1553_minlinkthroughput', DWORD),)

class LPSERVER_INFO_1553(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1553),)

class SERVER_INFO_1554(NDRSTRUCT):
    structure = (('sv1554_linkinfovalidtime', DWORD),)

class LPSERVER_INFO_1554(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1554),)

class SERVER_INFO_1555(NDRSTRUCT):
    structure = (('sv1555_scavqosinfoupdatetime', DWORD),)

class LPSERVER_INFO_1555(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1555),)

class SERVER_INFO_1556(NDRSTRUCT):
    structure = (('sv1556_maxworkitemidletime', DWORD),)

class LPSERVER_INFO_1556(NDRPOINTER):
    referent = (('Data', SERVER_INFO_1556),)

class WCHAR_ARRAY(NDRSTRUCT):
    commonHdr = (('Offset', '<L=0'), ('ActualCount', '<L=len(Data)//2'))
    commonHdr64 = (('Offset', '<Q=0'), ('ActualCount', '<Q=len(Data)//2'))
    structure = (('Data', ':'),)

    def dump(self, msg=None, indent=0):
        if False:
            while True:
                i = 10
        if msg is None:
            msg = self.__class__.__name__
        if msg != '':
            print('%s' % msg, end=' ')
        print(' %r' % self['Data'], end=' ')

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        if key == 'Data':
            try:
                self.fields[key] = value.encode('utf-16le')
            except UnicodeDecodeError:
                import sys
                self.fields[key] = value.decode(sys.getfilesystemencoding()).encode('utf-16le')
            self.fields['ActualCount'] = None
            self.data = None
        else:
            return NDR.__setitem__(self, key, value)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        if key == 'Data':
            return self.fields[key].decode('utf-16le')
        else:
            return NDR.__getitem__(self, key)

    def getDataLen(self, data, offset=0):
        if False:
            print('Hello World!')
        return self['ActualCount'] * 2

class DISK_INFO(NDRSTRUCT):
    structure = (('Disk', WCHAR_ARRAY),)

class LPDISK_INFO(NDRPOINTER):
    referent = (('Data', DISK_INFO),)

class DISK_INFO_ARRAY(NDRUniConformantVaryingArray):
    item = DISK_INFO

class LPDISK_INFO_ARRAY(NDRPOINTER):
    referent = (('Data', DISK_INFO_ARRAY),)

class DISK_ENUM_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPDISK_INFO_ARRAY))

class LPDISK_ENUM_CONTAINER(NDRPOINTER):
    referent = (('Data', DISK_ENUM_CONTAINER),)

class SERVER_TRANSPORT_INFO_0(NDRSTRUCT):
    structure = (('svti0_numberofvcs', DWORD), ('svti0_transportname', LPWSTR), ('svti0_transportaddress', PNDRUniConformantArray), ('svti0_transportaddresslength', DWORD), ('svti0_networkaddress', LPWSTR))

class LPSERVER_TRANSPORT_INFO_0(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_0),)

class SERVER_TRANSPORT_INFO_0_ARRAY(NDRUniConformantArray):
    item = SERVER_TRANSPORT_INFO_0

class LPSERVER_TRANSPORT_INFO_0_ARRAY(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_0_ARRAY),)

class SERVER_TRANSPORT_INFO_1(NDRSTRUCT):
    structure = (('svti1_numberofvcs', DWORD), ('svti1_transportname', LPWSTR), ('svti1_transportaddress', PNDRUniConformantArray), ('svti1_transportaddresslength', DWORD), ('svti1_networkaddress', LPWSTR), ('svti1_domain', LPWSTR))

class LPSERVER_TRANSPORT_INFO_1(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_1),)

class SERVER_TRANSPORT_INFO_1_ARRAY(NDRUniConformantArray):
    item = SERVER_TRANSPORT_INFO_1

class LPSERVER_TRANSPORT_INFO_1_ARRAY(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_1_ARRAY),)

class SERVER_TRANSPORT_INFO_2(NDRSTRUCT):
    structure = (('svti2_numberofvcs', DWORD), ('svti2_transportname', LPWSTR), ('svti2_transportaddress', PNDRUniConformantArray), ('svti2_transportaddresslength', DWORD), ('svti2_networkaddress', LPWSTR), ('svti2_domain', LPWSTR), ('svti2_flags', DWORD))

class LPSERVER_TRANSPORT_INFO_2(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_2),)

class SERVER_TRANSPORT_INFO_2_ARRAY(NDRUniConformantArray):
    item = SERVER_TRANSPORT_INFO_2

class LPSERVER_TRANSPORT_INFO_2_ARRAY(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_2_ARRAY),)

class PASSWORD_ARRAY(NDRUniFixedArray):

    def getDataLen(self, data, offset=0):
        if False:
            return 10
        return 256

class SERVER_TRANSPORT_INFO_3(NDRSTRUCT):
    structure = (('svti3_numberofvcs', DWORD), ('svti3_transportname', LPWSTR), ('svti3_transportaddress', PNDRUniConformantArray), ('svti3_transportaddresslength', DWORD), ('svti3_networkaddress', LPWSTR), ('svti3_domain', LPWSTR), ('svti3_flags', DWORD), ('svti3_passwordlength', DWORD), ('svti3_password', PASSWORD_ARRAY))

class LPSERVER_TRANSPORT_INFO_3(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_3),)

class SERVER_TRANSPORT_INFO_3_ARRAY(NDRUniConformantArray):
    item = SERVER_TRANSPORT_INFO_3

class LPSERVER_TRANSPORT_INFO_3_ARRAY(NDRPOINTER):
    referent = (('Data', SERVER_TRANSPORT_INFO_3_ARRAY),)

class SERVER_XPORT_INFO_0_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSERVER_TRANSPORT_INFO_0_ARRAY))

class LPSERVER_XPORT_INFO_0_CONTAINER(NDRPOINTER):
    referent = (('Data', SERVER_XPORT_INFO_0_CONTAINER),)

class SERVER_XPORT_INFO_1_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSERVER_TRANSPORT_INFO_1_ARRAY))

class LPSERVER_XPORT_INFO_1_CONTAINER(NDRPOINTER):
    referent = (('Data', SERVER_XPORT_INFO_1_CONTAINER),)

class SERVER_XPORT_INFO_2_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSERVER_TRANSPORT_INFO_2_ARRAY))

class LPSERVER_XPORT_INFO_2_CONTAINER(NDRPOINTER):
    referent = (('Data', SERVER_XPORT_INFO_2_CONTAINER),)

class SERVER_XPORT_INFO_3_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSERVER_TRANSPORT_INFO_3_ARRAY))

class LPSERVER_XPORT_INFO_3_CONTAINER(NDRPOINTER):
    referent = (('Data', SERVER_XPORT_INFO_3_CONTAINER),)

class SERVER_XPORT_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Level0', LPSERVER_XPORT_INFO_0_CONTAINER), 1: ('Level1', LPSERVER_XPORT_INFO_1_CONTAINER), 2: ('Level2', LPSERVER_XPORT_INFO_2_CONTAINER), 3: ('Level3', LPSERVER_XPORT_INFO_3_CONTAINER)}

class SERVER_XPORT_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('XportInfo', SERVER_XPORT_ENUM_UNION))

class SERVER_ALIAS_INFO_0(NDRSTRUCT):
    structure = (('srvai0_alias', LMSTR), ('srvai0_target', LMSTR), ('srvai0_default', NDRBOOLEAN), ('srvai0_reserved', ULONG))

class LPSERVER_ALIAS_INFO_0(NDRPOINTER):
    referent = (('Data', SERVER_ALIAS_INFO_0),)

class SERVER_ALIAS_INFO_0_ARRAY(NDRUniConformantArray):
    item = SERVER_ALIAS_INFO_0

class LPSERVER_ALIAS_INFO_0_ARRAY(NDRPOINTER):
    referent = (('Data', SERVER_ALIAS_INFO_0_ARRAY),)

class SERVER_ALIAS_INFO_0_CONTAINER(NDRSTRUCT):
    structure = (('EntriesRead', DWORD), ('Buffer', LPSERVER_ALIAS_INFO_0_ARRAY))

class LPSERVER_ALIAS_INFO_0_CONTAINER(NDRPOINTER):
    referent = (('Data', SERVER_ALIAS_INFO_0_CONTAINER),)

class SERVER_ALIAS_ENUM_UNION(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Level0', LPSERVER_ALIAS_INFO_0_CONTAINER)}

class SERVER_ALIAS_ENUM_STRUCT(NDRSTRUCT):
    structure = (('Level', DWORD), ('ServerAliasInfo', SERVER_ALIAS_ENUM_UNION))

class TIME_OF_DAY_INFO(NDRSTRUCT):
    structure = (('tod_elapsedt', DWORD), ('tod_msecs', DWORD), ('tod_hours', DWORD), ('tod_mins', DWORD), ('tod_secs', DWORD), ('tod_hunds', DWORD), ('tod_timezone', DWORD), ('tod_tinterval', DWORD), ('tod_day', DWORD), ('tod_month', DWORD), ('tod_year', DWORD), ('tod_weekday', DWORD))

class LPTIME_OF_DAY_INFO(NDRPOINTER):
    referent = (('Data', TIME_OF_DAY_INFO),)

class ADT_SECURITY_DESCRIPTOR(NDRSTRUCT):
    structure = (('Length', DWORD), ('Buffer', PNDRUniConformantArray))

class PADT_SECURITY_DESCRIPTOR(NDRPOINTER):
    referent = (('Data', ADT_SECURITY_DESCRIPTOR),)

class NET_DFS_ENTRY_ID(NDRSTRUCT):
    structure = (('Uid', GUID), ('Prefix', LPWSTR))

class NET_DFS_ENTRY_ID_ARRAY(NDRUniConformantArray):
    item = NET_DFS_ENTRY_ID

class LPNET_DFS_ENTRY_ID_ARRAY(NDRPOINTER):
    referent = (('Data', NET_DFS_ENTRY_ID_ARRAY),)

class NET_DFS_ENTRY_ID_CONTAINER(NDRSTRUCT):
    structure = (('Count', DWORD), ('Buffer', LPNET_DFS_ENTRY_ID_ARRAY))

class DFS_SITENAME_INFO(NDRSTRUCT):
    structure = (('SiteFlags', DWORD), ('SiteName', LPWSTR))

class DFS_SITENAME_INFO_ARRAY(NDRUniConformantArray):
    item = DFS_SITENAME_INFO

class DFS_SITELIST_INFO(NDRSTRUCT):
    structure = (('cSites', DWORD), ('Site', DFS_SITENAME_INFO_ARRAY))

class LPDFS_SITELIST_INFO(NDRPOINTER):
    referent = (('Data', DFS_SITELIST_INFO),)

class FILE_INFO(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {2: ('FileInfo2', LPFILE_INFO_2), 3: ('FileInfo3', LPFILE_INFO_3)}

class SHARE_INFO(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('ShareInfo0', LPSHARE_INFO_0), 1: ('ShareInfo1', LPSHARE_INFO_1), 2: ('ShareInfo2', LPSHARE_INFO_2), 502: ('ShareInfo502', LPSHARE_INFO_502), 1004: ('ShareInfo1004', LPSHARE_INFO_1004), 1006: ('ShareInfo1006', LPSHARE_INFO_1006), 1501: ('ShareInfo1501', LPSHARE_INFO_1501), 1005: ('ShareInfo1005', LPSHARE_INFO_1005), 501: ('ShareInfo501', LPSHARE_INFO_501), 503: ('ShareInfo503', LPSHARE_INFO_503)}

class SERVER_INFO(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {100: ('ServerInfo100', LPSERVER_INFO_100), 101: ('ServerInfo101', LPSERVER_INFO_101), 102: ('ServerInfo102', LPSERVER_INFO_102), 103: ('ServerInfo103', LPSERVER_INFO_103), 502: ('ServerInfo502', LPSERVER_INFO_502), 503: ('ServerInfo503', LPSERVER_INFO_503), 599: ('ServerInfo599', LPSERVER_INFO_599), 1005: ('ServerInfo1005', LPSERVER_INFO_1005), 1107: ('ServerInfo1107', LPSERVER_INFO_1107), 1010: ('ServerInfo1010', LPSERVER_INFO_1010), 1016: ('ServerInfo1016', LPSERVER_INFO_1016), 1017: ('ServerInfo1017', LPSERVER_INFO_1017), 1018: ('ServerInfo1018', LPSERVER_INFO_1018), 1501: ('ServerInfo1501', LPSERVER_INFO_1501), 1502: ('ServerInfo1502', LPSERVER_INFO_1502), 1503: ('ServerInfo1503', LPSERVER_INFO_1503), 1506: ('ServerInfo1506', LPSERVER_INFO_1506), 1510: ('ServerInfo1510', LPSERVER_INFO_1510), 1511: ('ServerInfo1511', LPSERVER_INFO_1511), 1512: ('ServerInfo1512', LPSERVER_INFO_1512), 1513: ('ServerInfo1513', LPSERVER_INFO_1513), 1514: ('ServerInfo1514', LPSERVER_INFO_1514), 1515: ('ServerInfo1515', LPSERVER_INFO_1515), 1516: ('ServerInfo1516', LPSERVER_INFO_1516), 1518: ('ServerInfo1518', LPSERVER_INFO_1518), 1523: ('ServerInfo1523', LPSERVER_INFO_1523), 1528: ('ServerInfo1528', LPSERVER_INFO_1528), 1529: ('ServerInfo1529', LPSERVER_INFO_1529), 1530: ('ServerInfo1530', LPSERVER_INFO_1530), 1533: ('ServerInfo1533', LPSERVER_INFO_1533), 1534: ('ServerInfo1534', LPSERVER_INFO_1534), 1535: ('ServerInfo1535', LPSERVER_INFO_1535), 1536: ('ServerInfo1536', LPSERVER_INFO_1536), 1538: ('ServerInfo1538', LPSERVER_INFO_1538), 1539: ('ServerInfo1539', LPSERVER_INFO_1539), 1540: ('ServerInfo1540', LPSERVER_INFO_1540), 1541: ('ServerInfo1541', LPSERVER_INFO_1541), 1542: ('ServerInfo1542', LPSERVER_INFO_1542), 1543: ('ServerInfo1543', LPSERVER_INFO_1543), 1544: ('ServerInfo1544', LPSERVER_INFO_1544), 1545: ('ServerInfo1545', LPSERVER_INFO_1545), 1546: ('ServerInfo1546', LPSERVER_INFO_1546), 1547: ('ServerInfo1547', LPSERVER_INFO_1547), 1548: ('ServerInfo1548', LPSERVER_INFO_1548), 1549: ('ServerInfo1549', LPSERVER_INFO_1549), 1550: ('ServerInfo1550', LPSERVER_INFO_1550), 1552: ('ServerInfo1552', LPSERVER_INFO_1552), 1553: ('ServerInfo1553', LPSERVER_INFO_1553), 1554: ('ServerInfo1554', LPSERVER_INFO_1554), 1555: ('ServerInfo1555', LPSERVER_INFO_1555), 1556: ('ServerInfo1556', LPSERVER_INFO_1556)}

class TRANSPORT_INFO(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('Transport0', SERVER_TRANSPORT_INFO_0), 1: ('Transport1', SERVER_TRANSPORT_INFO_1), 2: ('Transport2', SERVER_TRANSPORT_INFO_2), 3: ('Transport3', SERVER_TRANSPORT_INFO_3)}

class SERVER_ALIAS_INFO(NDRUNION):
    commonHdr = (('tag', DWORD),)
    union = {0: ('ServerAliasInfo0', LPSERVER_ALIAS_INFO_0)}

class NetrConnectionEnum(NDRCALL):
    opnum = 8
    structure = (('ServerName', PSRVSVC_HANDLE), ('Qualifier', LPWSTR), ('InfoStruct', CONNECT_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrConnectionEnumResponse(NDRCALL):
    structure = (('InfoStruct', CONNECT_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrFileEnum(NDRCALL):
    opnum = 9
    structure = (('ServerName', PSRVSVC_HANDLE), ('BasePath', LPWSTR), ('UserName', LPWSTR), ('InfoStruct', FILE_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrFileEnumResponse(NDRCALL):
    structure = (('InfoStruct', FILE_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrFileGetInfo(NDRCALL):
    opnum = 10
    structure = (('ServerName', PSRVSVC_HANDLE), ('FileId', DWORD), ('Level', DWORD))

class NetrFileGetInfoResponse(NDRCALL):
    structure = (('InfoStruct', FILE_INFO), ('ErrorCode', ULONG))

class NetrFileClose(NDRCALL):
    opnum = 11
    structure = (('ServerName', PSRVSVC_HANDLE), ('FileId', DWORD))

class NetrFileCloseResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrSessionEnum(NDRCALL):
    opnum = 12
    structure = (('ServerName', PSRVSVC_HANDLE), ('ClientName', LPWSTR), ('UserName', LPWSTR), ('InfoStruct', SESSION_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrSessionEnumResponse(NDRCALL):
    structure = (('InfoStruct', SESSION_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrSessionDel(NDRCALL):
    opnum = 13
    structure = (('ServerName', PSRVSVC_HANDLE), ('ClientName', LPWSTR), ('UserName', LPWSTR))

class NetrSessionDelResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrShareAdd(NDRCALL):
    opnum = 14
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('InfoStruct', SHARE_INFO), ('ParmErr', LPLONG))

class NetrShareAddResponse(NDRCALL):
    structure = (('ParmErr', LPLONG), ('ErrorCode', ULONG))

class NetrShareEnum(NDRCALL):
    opnum = 15
    structure = (('ServerName', PSRVSVC_HANDLE), ('InfoStruct', SHARE_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrShareEnumResponse(NDRCALL):
    structure = (('InfoStruct', SHARE_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrShareEnumSticky(NDRCALL):
    opnum = 36
    structure = (('ServerName', PSRVSVC_HANDLE), ('InfoStruct', SHARE_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrShareEnumStickyResponse(NDRCALL):
    structure = (('InfoStruct', SHARE_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrShareGetInfo(NDRCALL):
    opnum = 16
    structure = (('ServerName', PSRVSVC_HANDLE), ('NetName', WSTR), ('Level', DWORD))

class NetrShareGetInfoResponse(NDRCALL):
    structure = (('InfoStruct', SHARE_INFO), ('ErrorCode', ULONG))

class NetrShareSetInfo(NDRCALL):
    opnum = 17
    structure = (('ServerName', PSRVSVC_HANDLE), ('NetName', WSTR), ('Level', DWORD), ('ShareInfo', SHARE_INFO), ('ParmErr', LPLONG))

class NetrShareSetInfoResponse(NDRCALL):
    structure = (('ParmErr', LPLONG), ('ErrorCode', ULONG))

class NetrShareDel(NDRCALL):
    opnum = 18
    structure = (('ServerName', PSRVSVC_HANDLE), ('NetName', WSTR), ('Reserved', DWORD))

class NetrShareDelResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrShareDelSticky(NDRCALL):
    opnum = 19
    structure = (('ServerName', PSRVSVC_HANDLE), ('NetName', WSTR), ('Reserved', DWORD))

class NetrShareDelStickyResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrShareDelStart(NDRCALL):
    opnum = 37
    structure = (('ServerName', PSRVSVC_HANDLE), ('NetName', WSTR), ('Reserved', DWORD))

class NetrShareDelStartResponse(NDRCALL):
    structure = (('ContextHandle', SHARE_DEL_HANDLE), ('ErrorCode', ULONG))

class NetrShareDelCommit(NDRCALL):
    opnum = 38
    structure = (('ContextHandle', SHARE_DEL_HANDLE),)

class NetrShareDelCommitResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrShareCheck(NDRCALL):
    opnum = 20
    structure = (('ServerName', PSRVSVC_HANDLE), ('Device', WSTR))

class NetrShareCheckResponse(NDRCALL):
    structure = (('Type', DWORD), ('ErrorCode', ULONG))

class NetrServerGetInfo(NDRCALL):
    opnum = 21
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD))

class NetrServerGetInfoResponse(NDRCALL):
    structure = (('InfoStruct', SERVER_INFO), ('ErrorCode', ULONG))

class NetrServerSetInfo(NDRCALL):
    opnum = 22
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('InfoStruct', SERVER_INFO))

class NetrServerSetInfoResponse(NDRCALL):
    structure = (('ParmErr', LPLONG), ('ErrorCode', ULONG))

class NetrServerDiskEnum(NDRCALL):
    opnum = 23
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('DiskInfoStruct', DISK_ENUM_CONTAINER), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrServerDiskEnumResponse(NDRCALL):
    structure = (('DiskInfoStruct', DISK_ENUM_CONTAINER), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrServerStatisticsGet(NDRCALL):
    opnum = 24
    structure = (('ServerName', PSRVSVC_HANDLE), ('Service', LPWSTR), ('Level', DWORD), ('Options', DWORD))

class NetrServerStatisticsGetResponse(NDRCALL):
    structure = (('InfoStruct', LPSTAT_SERVER_0), ('ErrorCode', ULONG))

class NetrRemoteTOD(NDRCALL):
    opnum = 28
    structure = (('ServerName', PSRVSVC_HANDLE),)

class NetrRemoteTODResponse(NDRCALL):
    structure = (('BufferPtr', LPTIME_OF_DAY_INFO), ('ErrorCode', ULONG))

class NetrServerTransportAdd(NDRCALL):
    opnum = 25
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('Buffer', SERVER_TRANSPORT_INFO_0))

class NetrServerTransportAddResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrServerTransportAddEx(NDRCALL):
    opnum = 41
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('Buffer', TRANSPORT_INFO))

class NetrServerTransportAddExResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrServerTransportEnum(NDRCALL):
    opnum = 26
    structure = (('ServerName', PSRVSVC_HANDLE), ('InfoStruct', SERVER_XPORT_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrServerTransportEnumResponse(NDRCALL):
    structure = (('InfoStruct', SERVER_XPORT_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrServerTransportDel(NDRCALL):
    opnum = 27
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('Buffer', SERVER_TRANSPORT_INFO_0))

class NetrServerTransportDelResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrServerTransportDelEx(NDRCALL):
    opnum = 53
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('Buffer', TRANSPORT_INFO))

class NetrServerTransportDelExResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrpGetFileSecurity(NDRCALL):
    opnum = 39
    structure = (('ServerName', PSRVSVC_HANDLE), ('ShareName', LPWSTR), ('lpFileName', WSTR), ('RequestedInformation', SECURITY_INFORMATION))

class NetrpGetFileSecurityResponse(NDRCALL):
    structure = (('SecurityDescriptor', PADT_SECURITY_DESCRIPTOR), ('ErrorCode', ULONG))

class NetrpSetFileSecurity(NDRCALL):
    opnum = 40
    structure = (('ServerName', PSRVSVC_HANDLE), ('ShareName', LPWSTR), ('lpFileName', WSTR), ('SecurityInformation', SECURITY_INFORMATION), ('SecurityDescriptor', ADT_SECURITY_DESCRIPTOR))

class NetrpSetFileSecurityResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetprPathType(NDRCALL):
    opnum = 30
    structure = (('ServerName', PSRVSVC_HANDLE), ('PathName', WSTR), ('Flags', DWORD))

class NetprPathTypeResponse(NDRCALL):
    structure = (('PathType', DWORD), ('ErrorCode', ULONG))

class NetprPathCanonicalize(NDRCALL):
    opnum = 31
    structure = (('ServerName', PSRVSVC_HANDLE), ('PathName', WSTR), ('OutbufLen', DWORD), ('Prefix', WSTR), ('PathType', DWORD), ('Flags', DWORD))

class NetprPathCanonicalizeResponse(NDRCALL):
    structure = (('Outbuf', NDRUniConformantArray), ('PathType', DWORD), ('ErrorCode', ULONG))

class NetprPathCompare(NDRCALL):
    opnum = 32
    structure = (('ServerName', PSRVSVC_HANDLE), ('PathName1', WSTR), ('PathName2', WSTR), ('PathType', DWORD), ('Flags', DWORD))

class NetprPathCompareResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetprNameValidate(NDRCALL):
    opnum = 33
    structure = (('ServerName', PSRVSVC_HANDLE), ('Name', WSTR), ('NameType', DWORD), ('Flags', DWORD))

class NetprNameValidateResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetprNameCanonicalize(NDRCALL):
    opnum = 34
    structure = (('ServerName', PSRVSVC_HANDLE), ('Name', WSTR), ('OutbufLen', DWORD), ('NameType', DWORD), ('Flags', DWORD))

class NetprNameCanonicalizeResponse(NDRCALL):
    structure = (('Outbuf', NDRUniConformantArray), ('NameType', DWORD), ('ErrorCode', ULONG))

class NetprNameCompare(NDRCALL):
    opnum = 35
    structure = (('ServerName', PSRVSVC_HANDLE), ('Name1', WSTR), ('Name2', WSTR), ('NameType', DWORD), ('Flags', DWORD))

class NetprNameCompareResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsGetVersion(NDRCALL):
    opnum = 43
    structure = (('ServerName', PSRVSVC_HANDLE),)

class NetrDfsGetVersionResponse(NDRCALL):
    structure = (('Version', DWORD), ('ErrorCode', ULONG))

class NetrDfsCreateLocalPartition(NDRCALL):
    opnum = 44
    structure = (('ServerName', PSRVSVC_HANDLE), ('ShareName', WSTR), ('EntryUid', GUID), ('EntryPrefix', WSTR), ('ShortName', WSTR), ('RelationInfo', NET_DFS_ENTRY_ID_CONTAINER), ('Force', DWORD))

class NetrDfsCreateLocalPartitionResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsDeleteLocalPartition(NDRCALL):
    opnum = 45
    structure = (('ServerName', PSRVSVC_HANDLE), ('Uid', GUID), ('Prefix', WSTR))

class NetrDfsDeleteLocalPartitionResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsSetLocalVolumeState(NDRCALL):
    opnum = 46
    structure = (('ServerName', PSRVSVC_HANDLE), ('Uid', GUID), ('Prefix', WSTR), ('State', DWORD))

class NetrDfsSetLocalVolumeStateResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsCreateExitPoint(NDRCALL):
    opnum = 48
    structure = (('ServerName', PSRVSVC_HANDLE), ('Uid', GUID), ('Prefix', WSTR), ('Type', DWORD), ('ShortPrefixLen', DWORD))

class NetrDfsCreateExitPointResponse(NDRCALL):
    structure = (('ShortPrefix', WCHAR_ARRAY), ('ErrorCode', ULONG))

class NetrDfsModifyPrefix(NDRCALL):
    opnum = 50
    structure = (('ServerName', PSRVSVC_HANDLE), ('Uid', GUID), ('Prefix', WSTR))

class NetrDfsModifyPrefixResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsDeleteExitPoint(NDRCALL):
    opnum = 49
    structure = (('ServerName', PSRVSVC_HANDLE), ('Uid', GUID), ('Prefix', WSTR), ('Type', DWORD))

class NetrDfsDeleteExitPointResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsFixLocalVolume(NDRCALL):
    opnum = 51
    structure = (('ServerName', PSRVSVC_HANDLE), ('VolumeName', WSTR), ('EntryType', DWORD), ('ServiceType', DWORD), ('StgId', WSTR), ('EntryUid', GUID), ('EntryPrefix', WSTR), ('RelationInfo', NET_DFS_ENTRY_ID_CONTAINER), ('CreateDisposition', DWORD))

class NetrDfsFixLocalVolumeResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrDfsManagerReportSiteInfo(NDRCALL):
    opnum = 52
    structure = (('ServerName', PSRVSVC_HANDLE), ('ppSiteInfo', LPDFS_SITELIST_INFO))

class NetrDfsManagerReportSiteInfoResponse(NDRCALL):
    structure = (('ppSiteInfo', LPDFS_SITELIST_INFO), ('ErrorCode', ULONG))

class NetrServerAliasAdd(NDRCALL):
    opnum = 54
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('InfoStruct', SERVER_ALIAS_INFO))

class NetrServerAliasAddResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrServerAliasEnum(NDRCALL):
    opnum = 55
    structure = (('ServerName', PSRVSVC_HANDLE), ('InfoStruct', SERVER_ALIAS_ENUM_STRUCT), ('PreferedMaximumLength', DWORD), ('ResumeHandle', LPLONG))

class NetrServerAliasEnumResponse(NDRCALL):
    structure = (('InfoStruct', SERVER_ALIAS_ENUM_STRUCT), ('TotalEntries', DWORD), ('ResumeHandle', LPLONG), ('ErrorCode', ULONG))

class NetrServerAliasDel(NDRCALL):
    opnum = 56
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('InfoStruct', SERVER_ALIAS_INFO))

class NetrServerAliasDelResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)

class NetrShareDelEx(NDRCALL):
    opnum = 57
    structure = (('ServerName', PSRVSVC_HANDLE), ('Level', DWORD), ('ShareInfo', SHARE_INFO))

class NetrShareDelExResponse(NDRCALL):
    structure = (('ErrorCode', ULONG),)
OPNUMS = {8: (NetrConnectionEnum, NetrConnectionEnumResponse), 9: (NetrFileEnum, NetrFileEnumResponse), 10: (NetrFileGetInfo, NetrFileGetInfoResponse), 11: (NetrFileClose, NetrFileCloseResponse), 12: (NetrSessionEnum, NetrSessionEnumResponse), 13: (NetrSessionDel, NetrSessionDelResponse), 14: (NetrShareAdd, NetrShareAddResponse), 15: (NetrShareEnum, NetrShareEnumResponse), 16: (NetrShareGetInfo, NetrShareGetInfoResponse), 17: (NetrShareSetInfo, NetrShareSetInfoResponse), 18: (NetrShareDel, NetrShareDelResponse), 19: (NetrShareDelSticky, NetrShareDelStickyResponse), 20: (NetrShareCheck, NetrShareCheckResponse), 21: (NetrServerGetInfo, NetrServerGetInfoResponse), 22: (NetrServerSetInfo, NetrServerSetInfoResponse), 23: (NetrServerDiskEnum, NetrServerDiskEnumResponse), 24: (NetrServerStatisticsGet, NetrServerStatisticsGetResponse), 25: (NetrServerTransportAdd, NetrServerTransportAddResponse), 26: (NetrServerTransportEnum, NetrServerTransportEnumResponse), 27: (NetrServerTransportDel, NetrServerTransportDelResponse), 28: (NetrRemoteTOD, NetrRemoteTODResponse), 30: (NetprPathType, NetprPathTypeResponse), 31: (NetprPathCanonicalize, NetprPathCanonicalizeResponse), 32: (NetprPathCompare, NetprPathCompareResponse), 33: (NetprNameValidate, NetprNameValidateResponse), 34: (NetprNameCanonicalize, NetprNameCanonicalizeResponse), 35: (NetprNameCompare, NetprNameCompareResponse), 36: (NetrShareEnumSticky, NetrShareEnumStickyResponse), 37: (NetrShareDelStart, NetrShareDelStartResponse), 38: (NetrShareDelCommit, NetrShareDelCommitResponse), 39: (NetrpGetFileSecurity, NetrpGetFileSecurityResponse), 40: (NetrpSetFileSecurity, NetrpSetFileSecurityResponse), 41: (NetrServerTransportAddEx, NetrServerTransportAddExResponse), 43: (NetrDfsGetVersion, NetrDfsGetVersionResponse), 44: (NetrDfsCreateLocalPartition, NetrDfsCreateLocalPartitionResponse), 45: (NetrDfsDeleteLocalPartition, NetrDfsDeleteLocalPartitionResponse), 46: (NetrDfsSetLocalVolumeState, NetrDfsSetLocalVolumeStateResponse), 48: (NetrDfsCreateExitPoint, NetrDfsCreateExitPointResponse), 49: (NetrDfsDeleteExitPoint, NetrDfsDeleteExitPointResponse), 50: (NetrDfsModifyPrefix, NetrDfsModifyPrefixResponse), 51: (NetrDfsFixLocalVolume, NetrDfsFixLocalVolumeResponse), 52: (NetrDfsManagerReportSiteInfo, NetrDfsManagerReportSiteInfoResponse), 53: (NetrServerTransportDelEx, NetrServerTransportDelExResponse), 54: (NetrServerAliasAdd, NetrServerAliasAddResponse), 55: (NetrServerAliasEnum, NetrServerAliasEnumResponse), 56: (NetrServerAliasDel, NetrServerAliasDelResponse), 57: (NetrShareDelEx, NetrShareDelExResponse)}

def hNetrConnectionEnum(dce, qualifier, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        i = 10
        return i + 15
    request = NetrConnectionEnum()
    request['ServerName'] = NULL
    request['Qualifier'] = qualifier
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['ConnectInfo']['tag'] = level
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    return dce.request(request)

def hNetrFileEnum(dce, basePath, userName, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        for i in range(10):
            print('nop')
    request = NetrFileEnum()
    request['ServerName'] = NULL
    request['BasePath'] = basePath
    request['UserName'] = userName
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['FileInfo']['tag'] = level
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    return dce.request(request)

def hNetrFileGetInfo(dce, fileId, level):
    if False:
        print('Hello World!')
    request = NetrFileGetInfo()
    request['ServerName'] = NULL
    request['FileId'] = fileId
    request['Level'] = level
    return dce.request(request)

def hNetrFileClose(dce, fileId):
    if False:
        i = 10
        return i + 15
    request = NetrFileClose()
    request['ServerName'] = NULL
    request['FileId'] = fileId
    return dce.request(request)

def hNetrSessionEnum(dce, clientName, userName, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        while True:
            i = 10
    request = NetrSessionEnum()
    request['ServerName'] = NULL
    request['ClientName'] = clientName
    request['UserName'] = userName
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['SessionInfo']['tag'] = level
    request['InfoStruct']['SessionInfo']['Level%d' % level]['Buffer'] = NULL
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    return dce.request(request)

def hNetrSessionDel(dce, clientName, userName):
    if False:
        return 10
    request = NetrSessionDel()
    request['ServerName'] = NULL
    request['ClientName'] = clientName
    request['UserName'] = userName
    return dce.request(request)

def hNetrShareAdd(dce, level, infoStruct):
    if False:
        for i in range(10):
            print('nop')
    request = NetrShareAdd()
    request['ServerName'] = NULL
    request['Level'] = level
    request['InfoStruct']['tag'] = level
    request['InfoStruct']['ShareInfo%d' % level] = infoStruct
    return dce.request(request)

def hNetrShareDel(dce, netName):
    if False:
        while True:
            i = 10
    request = NetrShareDel()
    request['ServerName'] = NULL
    request['NetName'] = netName
    return dce.request(request)

def hNetrShareEnum(dce, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        for i in range(10):
            print('nop')
    request = NetrShareEnum()
    request['ServerName'] = '\x00'
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['ShareInfo']['tag'] = level
    request['InfoStruct']['ShareInfo']['Level%d' % level]['Buffer'] = NULL
    return dce.request(request)

def hNetrShareEnumSticky(dce, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        print('Hello World!')
    request = NetrShareEnumSticky()
    request['ServerName'] = NULL
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['ShareInfo']['tag'] = level
    request['InfoStruct']['ShareInfo']['Level%d' % level]['Buffer'] = NULL
    return dce.request(request)

def hNetrShareGetInfo(dce, netName, level):
    if False:
        return 10
    request = NetrShareGetInfo()
    request['ServerName'] = NULL
    request['NetName'] = netName
    request['Level'] = level
    return dce.request(request)

def hNetrShareSetInfo(dce, netName, level, shareInfo):
    if False:
        i = 10
        return i + 15
    request = NetrShareSetInfo()
    request['ServerName'] = NULL
    request['NetName'] = netName
    request['Level'] = level
    request['ShareInfo']['tag'] = level
    request['ShareInfo']['ShareInfo%d' % level] = shareInfo
    return dce.request(request)

def hNetrShareDelSticky(dce, netName):
    if False:
        print('Hello World!')
    request = NetrShareDelSticky()
    request['ServerName'] = NULL
    request['NetName'] = netName
    return dce.request(request)

def hNetrShareDelStart(dce, netName):
    if False:
        i = 10
        return i + 15
    request = NetrShareDelStart()
    request['ServerName'] = NULL
    request['NetName'] = netName
    return dce.request(request)

def hNetrShareDelCommit(dce, contextHandle):
    if False:
        i = 10
        return i + 15
    request = NetrShareDelCommit()
    request['ContextHandle'] = contextHandle
    return dce.request(request)

def hNetrShareCheck(dce, device):
    if False:
        print('Hello World!')
    request = NetrShareCheck()
    request['ServerName'] = NULL
    request['Device'] = device
    return dce.request(request)

def hNetrServerGetInfo(dce, level):
    if False:
        return 10
    request = NetrServerGetInfo()
    request['ServerName'] = NULL
    request['Level'] = level
    return dce.request(request)

def hNetrServerDiskEnum(dce, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        return 10
    request = NetrServerDiskEnum()
    request['ServerName'] = NULL
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    request['Level'] = level
    request['DiskInfoStruct']['Buffer'] = NULL
    return dce.request(request)

def hNetrServerStatisticsGet(dce, service, level, options):
    if False:
        while True:
            i = 10
    request = NetrServerStatisticsGet()
    request['ServerName'] = NULL
    request['Service'] = service
    request['Level'] = level
    request['Options'] = options
    return dce.request(request)

def hNetrRemoteTOD(dce):
    if False:
        while True:
            i = 10
    request = NetrRemoteTOD()
    request['ServerName'] = NULL
    return dce.request(request)

def hNetrServerTransportEnum(dce, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        return 10
    request = NetrServerTransportEnum()
    request['ServerName'] = NULL
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['XportInfo']['tag'] = level
    request['InfoStruct']['XportInfo']['Level%d' % level]['Buffer'] = NULL
    return dce.request(request)

def hNetrpGetFileSecurity(dce, shareName, lpFileName, requestedInformation):
    if False:
        for i in range(10):
            print('nop')
    request = NetrpGetFileSecurity()
    request['ServerName'] = NULL
    request['ShareName'] = shareName
    request['lpFileName'] = lpFileName
    request['RequestedInformation'] = requestedInformation
    retVal = dce.request(request)
    return b''.join(retVal['SecurityDescriptor']['Buffer'])

def hNetrpSetFileSecurity(dce, shareName, lpFileName, securityInformation, securityDescriptor):
    if False:
        i = 10
        return i + 15
    request = NetrpSetFileSecurity()
    request['ServerName'] = NULL
    request['ShareName'] = shareName
    request['lpFileName'] = lpFileName
    request['SecurityInformation'] = securityInformation
    request['SecurityDescriptor']['Length'] = len(securityDescriptor)
    request['SecurityDescriptor']['Buffer'] = list(securityDescriptor)
    return dce.request(request)

def hNetprPathType(dce, pathName, flags):
    if False:
        return 10
    request = NetprPathType()
    request['ServerName'] = NULL
    request['PathName'] = pathName
    request['Flags'] = flags
    return dce.request(request)

def hNetprPathCanonicalize(dce, pathName, prefix, outbufLen=50, pathType=0, flags=0):
    if False:
        return 10
    request = NetprPathCanonicalize()
    request['ServerName'] = NULL
    request['PathName'] = pathName
    request['OutbufLen'] = outbufLen
    request['Prefix'] = prefix
    request['PathType'] = pathType
    request['Flags'] = flags
    return dce.request(request)

def hNetprPathCompare(dce, pathName1, pathName2, pathType=0, flags=0):
    if False:
        for i in range(10):
            print('nop')
    request = NetprPathCompare()
    request['ServerName'] = NULL
    request['PathName1'] = pathName1
    request['PathName2'] = pathName2
    request['PathType'] = pathType
    request['Flags'] = flags
    return dce.request(request)

def hNetprNameValidate(dce, name, nameType, flags=0):
    if False:
        print('Hello World!')
    request = NetprNameValidate()
    request['ServerName'] = NULL
    request['Name'] = name
    request['NameType'] = nameType
    request['Flags'] = flags
    return dce.request(request)

def hNetprNameCanonicalize(dce, name, outbufLen=50, nameType=0, flags=0):
    if False:
        while True:
            i = 10
    request = NetprNameCanonicalize()
    request['ServerName'] = NULL
    request['Name'] = name
    request['OutbufLen'] = outbufLen
    request['NameType'] = nameType
    request['Flags'] = flags
    return dce.request(request)

def hNetprNameCompare(dce, name1, name2, nameType=0, flags=0):
    if False:
        return 10
    request = NetprNameCompare()
    request['ServerName'] = NULL
    request['Name1'] = name1
    request['Name2'] = name2
    request['NameType'] = nameType
    request['Flags'] = flags
    return dce.request(request)

def hNetrDfsGetVersion(dce):
    if False:
        i = 10
        return i + 15
    request = NetrDfsGetVersion()
    request['ServerName'] = NULL
    return dce.request(request)

def hNetrServerAliasAdd(dce, level, aliasInfo):
    if False:
        for i in range(10):
            print('nop')
    request = NetrServerAliasAdd()
    request['ServerName'] = NULL
    request['Level'] = level
    request['InfoStruct']['tag'] = level
    request['InfoStruct']['ServerAliasInfo%d' % level] = aliasInfo
    return dce.request(request)

def hNetrServerAliasDel(dce, level, aliasInfo):
    if False:
        print('Hello World!')
    request = NetrServerAliasDel()
    request['ServerName'] = NULL
    request['Level'] = level
    request['InfoStruct']['tag'] = level
    request['InfoStruct']['ServerAliasInfo%d' % level] = aliasInfo
    return dce.request(request)

def hNetrServerAliasEnum(dce, level, resumeHandle=0, preferedMaximumLength=4294967295):
    if False:
        while True:
            i = 10
    request = NetrServerAliasEnum()
    request['ServerName'] = NULL
    request['InfoStruct']['Level'] = level
    request['InfoStruct']['ServerAliasInfo']['tag'] = level
    request['InfoStruct']['ServerAliasInfo']['Level%d' % level]['Buffer'] = NULL
    request['PreferedMaximumLength'] = preferedMaximumLength
    request['ResumeHandle'] = resumeHandle
    return dce.request(request)