from pupylib.PupyErrors import PupyModuleError
NPCAP_NOT_FOUND = '\nWinPCAP is not installed. You should install NPcap driver.\nStandard version can be found here: https://github.com/nmap/npcap/releases\nOEM Version which supports silent install can be extracted from NMap installer.\nNmap 7.70 can be found here: https://nmap.org/dist/nmap-7.70-setup.exe.\nOnly OEM installer supports silent install (/S) option.\n'
KNOWN_DRIVERS = ['system32\\NPcap\\Packet.dll', 'system32\\Packet.dll']

def init_winpcap(client):
    if False:
        i = 10
        return i + 15
    exists = client.remote('os.path', 'exists', False)
    getenv = client.remote('os', 'getenv', False)
    environ = client.remote('os', 'environ', False)
    windir = getenv('WINDIR')
    if not windir:
        windir = 'C:\\Windows'
    if not any((exists(windir + '\\' + x) for x in KNOWN_DRIVERS)):
        raise PupyModuleError(NPCAP_NOT_FOUND)
    PATH = getenv('Path')
    if 'NPcap' not in PATH:
        environ['Path'] = PATH + ';' + windir + '\\system32\\NPcap'