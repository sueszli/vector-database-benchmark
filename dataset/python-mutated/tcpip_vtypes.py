import volatility.obj as obj
tcpip_vtypes = {'_ADDRESS_OBJECT': [104, {'Next': [0, ['pointer', ['_ADDRESS_OBJECT']]], 'LocalIpAddress': [44, ['IpAddress']], 'LocalPort': [48, ['unsigned be short']], 'Protocol': [50, ['unsigned short']], 'Pid': [328, ['unsigned long']], 'CreateTime': [344, ['WinTimeStamp', dict(is_utc=True)]]}], '_TCPT_OBJECT': [32, {'Next': [0, ['pointer', ['_TCPT_OBJECT']]], 'RemoteIpAddress': [12, ['IpAddress']], 'LocalIpAddress': [16, ['IpAddress']], 'RemotePort': [20, ['unsigned be short']], 'LocalPort': [22, ['unsigned be short']], 'Pid': [24, ['unsigned long']]}]}
tcpip_vtypes_2003_x64 = {'_ADDRESS_OBJECT': [None, {'Next': [0, ['pointer', ['_ADDRESS_OBJECT']]], 'LocalIpAddress': [88, ['IpAddress']], 'LocalPort': [92, ['unsigned be short']], 'Protocol': [94, ['unsigned short']], 'Pid': [568, ['unsigned long']], 'CreateTime': [584, ['WinTimeStamp', dict(is_utc=True)]]}], '_TCPT_OBJECT': [None, {'Next': [0, ['pointer', ['_TCPT_OBJECT']]], 'RemoteIpAddress': [20, ['IpAddress']], 'LocalIpAddress': [24, ['IpAddress']], 'RemotePort': [28, ['unsigned be short']], 'LocalPort': [30, ['unsigned be short']], 'Pid': [32, ['unsigned long']]}]}
tcpip_vtypes_2003_sp1_sp2 = {'_ADDRESS_OBJECT': [104, {'Next': [0, ['pointer', ['_ADDRESS_OBJECT']]], 'LocalIpAddress': [48, ['IpAddress']], 'LocalPort': [52, ['unsigned be short']], 'Protocol': [54, ['unsigned short']], 'Pid': [332, ['unsigned long']], 'CreateTime': [344, ['WinTimeStamp', dict(is_utc=True)]]}]}
TCP_STATE_ENUM = {0: 'CLOSED', 1: 'LISTENING', 2: 'SYN_SENT', 3: 'SYN_RCVD', 4: 'ESTABLISHED', 5: 'FIN_WAIT1', 6: 'FIN_WAIT2', 7: 'CLOSE_WAIT', 8: 'CLOSING', 9: 'LAST_ACK', 12: 'TIME_WAIT', 13: 'DELETE_TCB'}
tcpip_vtypes_vista = {'_IN_ADDR': [None, {'addr4': [0, ['IpAddress']], 'addr6': [0, ['Ipv6Address']]}], '_LOCAL_ADDRESS': [None, {'pData': [12, ['pointer', ['pointer', ['_IN_ADDR']]]]}], '_TCP_LISTENER': [None, {'Owner': [24, ['pointer', ['_EPROCESS']]], 'CreateTime': [32, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [52, ['pointer', ['_LOCAL_ADDRESS']]], 'InetAF': [56, ['pointer', ['_INETAF']]], 'Port': [62, ['unsigned be short']]}], '_TCP_ENDPOINT': [None, {'InetAF': [12, ['pointer', ['_INETAF']]], 'AddrInfo': [16, ['pointer', ['_ADDRINFO']]], 'ListEntry': [20, ['_LIST_ENTRY']], 'State': [40, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [44, ['unsigned be short']], 'RemotePort': [46, ['unsigned be short']], 'Owner': [352, ['pointer', ['_EPROCESS']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_TCP_SYN_ENDPOINT': [None, {'ListEntry': [8, ['_LIST_ENTRY']], 'InetAF': [24, ['pointer', ['_INETAF']]], 'LocalPort': [60, ['unsigned be short']], 'RemotePort': [62, ['unsigned be short']], 'LocalAddr': [28, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [40, ['pointer', ['_IN_ADDR']]], 'Owner': [32, ['pointer', ['_SYN_OWNER']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_SYN_OWNER': [None, {'Process': [24, ['pointer', ['_EPROCESS']]]}], '_TCP_TIMEWAIT_ENDPOINT': [None, {'ListEntry': [20, ['_LIST_ENTRY']], 'InetAF': [12, ['pointer', ['_INETAF']]], 'LocalPort': [28, ['unsigned be short']], 'RemotePort': [30, ['unsigned be short']], 'LocalAddr': [32, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [36, ['pointer', ['_IN_ADDR']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_INETAF': [None, {'AddressFamily': [12, ['unsigned short']]}], '_ADDRINFO': [None, {'Local': [0, ['pointer', ['_LOCAL_ADDRESS']]], 'Remote': [8, ['pointer', ['_IN_ADDR']]]}], '_UDP_ENDPOINT': [None, {'Owner': [24, ['pointer', ['_EPROCESS']]], 'CreateTime': [48, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [56, ['pointer', ['_LOCAL_ADDRESS']]], 'InetAF': [20, ['pointer', ['_INETAF']]], 'Port': [72, ['unsigned be short']]}]}
tcpip_vtypes_7 = {'_TCP_ENDPOINT': [None, {'InetAF': [12, ['pointer', ['_INETAF']]], 'AddrInfo': [16, ['pointer', ['_ADDRINFO']]], 'ListEntry': [20, ['_LIST_ENTRY']], 'State': [52, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [56, ['unsigned be short']], 'RemotePort': [58, ['unsigned be short']], 'Owner': [372, ['pointer', ['_EPROCESS']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_TCP_SYN_ENDPOINT': [None, {'ListEntry': [8, ['_LIST_ENTRY']], 'InetAF': [36, ['pointer', ['_INETAF']]], 'LocalPort': [72, ['unsigned be short']], 'RemotePort': [74, ['unsigned be short']], 'LocalAddr': [40, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [52, ['pointer', ['_IN_ADDR']]], 'Owner': [44, ['pointer', ['_SYN_OWNER']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_TCP_TIMEWAIT_ENDPOINT': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'InetAF': [24, ['pointer', ['_INETAF']]], 'LocalPort': [40, ['unsigned be short']], 'RemotePort': [42, ['unsigned be short']], 'LocalAddr': [44, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [48, ['pointer', ['_IN_ADDR']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}]}
tcpip_vtypes_vista_64 = {'_IN_ADDR': [None, {'addr4': [0, ['IpAddress']], 'addr6': [0, ['Ipv6Address']]}], '_TCP_LISTENER': [None, {'Owner': [40, ['pointer', ['_EPROCESS']]], 'CreateTime': [32, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [88, ['pointer', ['_LOCAL_ADDRESS']]], 'InetAF': [96, ['pointer', ['_INETAF']]], 'Port': [106, ['unsigned be short']]}], '_INETAF': [None, {'AddressFamily': [20, ['unsigned short']]}], '_LOCAL_ADDRESS': [None, {'pData': [16, ['pointer', ['pointer', ['_IN_ADDR']]]]}], '_ADDRINFO': [None, {'Local': [0, ['pointer', ['_LOCAL_ADDRESS']]], 'Remote': [16, ['pointer', ['_IN_ADDR']]]}], '_TCP_ENDPOINT': [None, {'InetAF': [24, ['pointer', ['_INETAF']]], 'AddrInfo': [32, ['pointer', ['_ADDRINFO']]], 'ListEntry': [40, ['_LIST_ENTRY']], 'State': [80, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [84, ['unsigned be short']], 'RemotePort': [86, ['unsigned be short']], 'Owner': [520, ['pointer', ['_EPROCESS']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_TCP_SYN_ENDPOINT': [None, {'ListEntry': [16, ['_LIST_ENTRY']], 'InetAF': [48, ['pointer', ['_INETAF']]], 'LocalPort': [100, ['unsigned be short']], 'RemotePort': [102, ['unsigned be short']], 'LocalAddr': [56, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [80, ['pointer', ['_IN_ADDR']]], 'Owner': [64, ['pointer', ['_SYN_OWNER']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_SYN_OWNER': [None, {'Process': [40, ['pointer', ['_EPROCESS']]]}], '_TCP_TIMEWAIT_ENDPOINT': [None, {'ListEntry': [0, ['_LIST_ENTRY']], 'InetAF': [24, ['pointer', ['_INETAF']]], 'LocalPort': [48, ['unsigned be short']], 'RemotePort': [50, ['unsigned be short']], 'LocalAddr': [56, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [64, ['pointer', ['_IN_ADDR']]], 'CreateTime': [0, ['WinTimeStamp', dict(value=0, is_utc=True)]]}], '_UDP_ENDPOINT': [None, {'Owner': [40, ['pointer', ['_EPROCESS']]], 'CreateTime': [88, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [96, ['pointer', ['_LOCAL_ADDRESS']]], 'InetAF': [32, ['pointer', ['_INETAF']]], 'Port': [128, ['unsigned be short']]}]}
tcpip_vtypes_win_10_x64 = {'_IN_ADDR': [None, {'addr4': [0, ['IpAddress']], 'addr6': [0, ['Ipv6Address']]}], '_INETAF': [None, {'AddressFamily': [24, ['unsigned short']]}], '_LOCAL_ADDRESS_WIN10_UDP': [None, {'pData': [0, ['pointer', ['_IN_ADDR']]]}], '_LOCAL_ADDRESS': [None, {'pData': [16, ['pointer', ['pointer', ['_IN_ADDR']]]]}], '_ADDRINFO': [None, {'Local': [0, ['pointer', ['_LOCAL_ADDRESS']]], 'Remote': [16, ['pointer', ['_IN_ADDR']]]}], '_TCP_LISTENER': [None, {'Owner': [48, ['pointer', ['_EPROCESS']]], 'CreateTime': [64, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [96, ['pointer', ['_LOCAL_ADDRESS']]], 'InetAF': [40, ['pointer', ['_INETAF']]], 'Port': [114, ['unsigned be short']]}], '_TCP_ENDPOINT': [None, {'InetAF': [16, ['pointer', ['_INETAF']]], 'AddrInfo': [24, ['pointer', ['_ADDRINFO']]], 'State': [108, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [112, ['unsigned be short']], 'RemotePort': [114, ['unsigned be short']], 'Owner': [600, ['pointer', ['_EPROCESS']]], 'CreateTime': [616, ['WinTimeStamp', dict(is_utc=True)]]}], '_UDP_ENDPOINT': [None, {'Owner': [40, ['pointer', ['_EPROCESS']]], 'CreateTime': [88, ['WinTimeStamp', dict(is_utc=True)]], 'LocalAddr': [128, ['pointer', ['_LOCAL_ADDRESS_WIN10_UDP']]], 'InetAF': [32, ['pointer', ['_INETAF']]], 'Port': [120, ['unsigned be short']]}]}

class _ADDRESS_OBJECT(obj.CType):

    def is_valid(self):
        if False:
            return 10
        return obj.CType.is_valid(self) and self.CreateTime.v() > 0

class WinXP2003AddressObject(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows', 'major': lambda x: x == 5}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.object_classes.update({'_ADDRESS_OBJECT': _ADDRESS_OBJECT})

class WinXP2003Tcpipx64(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2}

    def modification(self, profile):
        if False:
            return 10
        profile.vtypes.update(tcpip_vtypes_2003_x64)

class Win2003SP12Tcpip(obj.ProfileModification):
    before = ['WindowsVTypes']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 5, 'minor': lambda x: x == 2, 'build': lambda x: x != 3789}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update(tcpip_vtypes_2003_sp1_sp2)

class Vista2008Tcpip(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x >= 0}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(tcpip_vtypes_vista)

class Win7Tcpip(obj.ProfileModification):
    before = ['Vista2008Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 1}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.vtypes.update(tcpip_vtypes_7)

class Win7Vista2008x64Tcpip(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x >= 0}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update(tcpip_vtypes_vista_64)

class VistaSP12x64Tcpip(obj.ProfileModification):
    before = ['Win7Vista2008x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 0, 'build': lambda x: x >= 6001}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [528, ['pointer', ['_EPROCESS']]]}]})

class Win7x64Tcpip(obj.ProfileModification):
    before = ['Win7Vista2008x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 1}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'State': [104, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [108, ['unsigned be short']], 'RemotePort': [110, ['unsigned be short']], 'Owner': [568, ['pointer', ['_EPROCESS']]]}], '_TCP_SYN_ENDPOINT': [None, {'InetAF': [72, ['pointer', ['_INETAF']]], 'LocalPort': [124, ['unsigned be short']], 'RemotePort': [126, ['unsigned be short']], 'LocalAddr': [80, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [104, ['pointer', ['_IN_ADDR']]], 'Owner': [88, ['pointer', ['_SYN_OWNER']]]}], '_TCP_TIMEWAIT_ENDPOINT': [None, {'InetAF': [48, ['pointer', ['_INETAF']]], 'LocalPort': [72, ['unsigned be short']], 'RemotePort': [74, ['unsigned be short']], 'LocalAddr': [80, ['pointer', ['_LOCAL_ADDRESS']]], 'RemoteAddress': [88, ['pointer', ['_IN_ADDR']]]}]})

class Win8Tcpip(obj.ProfileModification):
    before = ['Vista2008Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x >= 2}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'InetAF': [8, ['pointer', ['_INETAF']]], 'AddrInfo': [12, ['pointer', ['_ADDRINFO']]], 'State': [56, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [60, ['unsigned be short']], 'RemotePort': [62, ['unsigned be short']], 'Owner': [372, ['pointer', ['_EPROCESS']]]}], '_ADDRINFO': [None, {'Remote': [12, ['pointer', ['_IN_ADDR']]]}]})

class Win81Tcpip(obj.ProfileModification):
    before = ['Win8Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 3}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [424, ['pointer', ['_EPROCESS']]]}]})

class Win10Tcpip(obj.ProfileModification):
    before = ['Win8Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '32bit', 'major': lambda x: x == 6, 'minor': lambda x: x >= 4}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'_ADDRINFO': [None, {'Local': [0, ['pointer', ['_LOCAL_ADDRESS']]], 'Remote': [12, ['pointer', ['_IN_ADDR']]]}], '_TCP_ENDPOINT': [None, {'InetAF': [8, ['pointer', ['_INETAF']]], 'AddrInfo': [12, ['pointer', ['_ADDRINFO']]], 'State': [56, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [60, ['unsigned be short']], 'RemotePort': [62, ['unsigned be short']], 'Owner': [432, ['pointer', ['_EPROCESS']]]}]})
        build = profile.metadata.get('build')
        if build == 14393:
            profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [436, ['pointer', ['_EPROCESS']]]}]})
        elif build >= 15063:
            profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [460, ['pointer', ['_EPROCESS']]]}]})

class Win8x64Tcpip(obj.ProfileModification):
    before = ['Win7Vista2008x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x >= 2}

    def modification(self, profile):
        if False:
            while True:
                i = 10
        profile.merge_overlay({'_INETAF': [None, {'AddressFamily': [24, ['unsigned short']]}], '_TCP_ENDPOINT': [None, {'InetAF': [16, ['pointer', ['_INETAF']]], 'AddrInfo': [24, ['pointer', ['_ADDRINFO']]], 'State': [108, ['Enumeration', dict(target='long', choices=TCP_STATE_ENUM)]], 'LocalPort': [112, ['unsigned be short']], 'RemotePort': [114, ['unsigned be short']], 'Owner': [592, ['pointer', ['_EPROCESS']]]}]})

class Win81x64Tcpip(obj.ProfileModification):
    before = ['Win8x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 3}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [600, ['pointer', ['_EPROCESS']]]}]})

class Win10x64Tcpip(obj.ProfileModification):
    before = ['Win81x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update(tcpip_vtypes_win_10_x64)

class Win10x64_15063_Tcpip(obj.ProfileModification):
    """TCP Endpoint for Creators and Fall Creators"""
    before = ['Win10x64Tcpip']
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit', 'major': lambda x: x == 6, 'minor': lambda x: x == 4, 'build': lambda x: x >= 15063}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        profile.merge_overlay({'_TCP_ENDPOINT': [None, {'Owner': [624, ['pointer', ['_EPROCESS']]]}]})