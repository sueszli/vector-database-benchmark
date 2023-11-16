"""
Wireshark extcap API utils
https://www.wireshark.org/docs/wsdg_html_chunked/ChCaptureExtcap.html
"""
import collections
import functools
import pathlib
import re
import subprocess
from scapy.config import conf
from scapy.consts import WINDOWS
from scapy.data import MTU
from scapy.error import warning
from scapy.interfaces import network_name, resolve_iface, InterfaceProvider, NetworkInterface
from scapy.packet import Packet
from scapy.supersocket import SuperSocket
from scapy.utils import PcapReader, _create_fifo, _open_fifo
from typing import cast, Any, Dict, List, NoReturn, Optional, Tuple, Type, Union

def _extcap_call(prog: str, args: List[str], format: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, ...]]]:
    if False:
        i = 10
        return i + 15
    '\n    Function used to call a program using the extcap format,\n    then parse the results\n    '
    p = subprocess.Popen([prog] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (data, err) = p.communicate()
    if p.returncode != 0:
        raise OSError('%s returned with error code %s: %s' % (prog, p.returncode, err))
    res = collections.defaultdict(list)
    for ifa in data.split('\n'):
        ifa = ifa.strip()
        for (keyword, values) in format.items():
            if not ifa.startswith(keyword):
                continue

            def _match(val: str, ifa: str) -> str:
                if False:
                    i = 10
                    return i + 15
                m = re.search('{%s=([^}]*)}' % val, ifa)
                if m:
                    return m.group(1)
                return ''
            res[keyword].append(tuple([_match(val, ifa) for val in values]))
            break
    return cast(Dict[str, List[Tuple[str, ...]]], res)

class _ExtcapNetworkInterface(NetworkInterface):
    """
    Extcap NetworkInterface
    """

    def get_extcap_config(self) -> Dict[str, Tuple[str, ...]]:
        if False:
            return 10
        '\n        Return a list of available configuration options on an extcap interface\n        '
        return _extcap_call(self.provider.cmdprog, ['--extcap-interface', self.network_name, '--extcap-config'], {'arg': ['number', 'call', 'display', 'default', 'required'], 'value': ['arg', 'value', 'display', 'default']})

    def get_extcap_cmd(self, **kwarg: Dict[str, str]) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Return the extcap command line options\n        '
        cmds = []
        for x in self.get_extcap_config()['arg']:
            key = x[1].strip('-').replace('-', '_')
            if key in kwarg:
                cmds += [x[1], str(kwarg[key])]
            else:
                if x[4] == 'true':
                    raise ValueError("Missing required argument: '%s' on iface %s." % (key, self.network_name))
                elif not x[3] or x[3] == 'false':
                    continue
                if x[3] == 'true':
                    cmds += [x[1]]
                else:
                    cmds += [x[1], x[3]]
        return cmds

class _ExtcapSocket(SuperSocket):
    """
    Read packets at layer 2 using an extcap command
    """
    nonblocking_socket = True

    @staticmethod
    def select(sockets: List[SuperSocket], remain: Optional[float]=None) -> List[SuperSocket]:
        if False:
            while True:
                i = 10
        return sockets

    def __init__(self, *_: Any, **kwarg: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        cmdprog = kwarg.pop('cmdprog')
        iface = kwarg.pop('iface', None)
        if iface is None:
            raise NameError('Must select an interface for a extcap socket !')
        iface = resolve_iface(iface)
        if not isinstance(iface, _ExtcapNetworkInterface):
            raise ValueError('Interface should be an _ExtcapNetworkInterface')
        args = iface.get_extcap_cmd(**kwarg)
        iface = network_name(iface)
        self.outs = None
        (fifo, fd) = _create_fifo()
        args = ['--extcap-interface', iface, '--capture', '--fifo', fifo] + args
        self.proc = subprocess.Popen([cmdprog] + args)
        self.fd = _open_fifo(fd)
        self.reader = PcapReader(self.fd)
        self.ins = self.reader

    def recv(self, x: int=MTU, **kwargs: Any) -> Packet:
        if False:
            return 10
        return self.reader.recv(x, **kwargs)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self.proc.kill()
        self.proc.wait(timeout=2)
        SuperSocket.close(self)
        self.fd.close()

class _ExtcapInterfaceProvider(InterfaceProvider):
    """
    Interface provider made to hook on a extcap binary
    """
    headers = ('Index', 'Name', 'Address')
    header_sort = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        self.cmdprog = kwargs.pop('cmdprog')
        super(_ExtcapInterfaceProvider, self).__init__(*args, **kwargs)

    def load(self) -> Dict[str, NetworkInterface]:
        if False:
            while True:
                i = 10
        data: Dict[str, NetworkInterface] = {}
        try:
            interfaces = _extcap_call(self.cmdprog, ['--extcap-interfaces'], {'interface': ['value', 'display']})['interface']
        except OSError as ex:
            warning('extcap %s failed to load: %s', self.name, str(ex).strip().split('\n')[-1])
            return {}
        for (netw_name, name) in interfaces:
            _index = re.search('.*(\\d+)', name)
            if _index:
                index = int(_index.group(1)) + 100
            else:
                index = 100
            if_data = {'name': name, 'network_name': netw_name, 'description': name, 'index': index}
            data[netw_name] = _ExtcapNetworkInterface(self, if_data)
        return data

    def _l2listen(self, _: Any) -> Type[SuperSocket]:
        if False:
            print('Hello World!')
        return functools.partial(_ExtcapSocket, cmdprog=self.cmdprog)

    def _l3socket(self, *_: Any) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise ValueError('Only sniffing is available for an extcap provider !')
    _l2socket = _l3socket

    def _is_valid(self, dev: NetworkInterface) -> bool:
        if False:
            return 10
        return True

    def _format(self, dev: NetworkInterface, **kwargs: Any) -> Tuple[Union[str, List[str]], ...]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a tuple of the elements used by show()'
        return (str(dev.index), dev.name, dev.network_name)

def load_extcap() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Load extcap folder from wireshark and populate providers\n    '
    if WINDOWS:
        pattern = re.compile('^[^.]+(?:\\.bat|\\.exe)?$')
    else:
        pattern = re.compile('^[^.]+(?:\\.sh)?$')
    for fld in conf.prog.extcap_folders:
        root = pathlib.Path(fld)
        for _cmdprog in root.glob('*'):
            if not _cmdprog.is_file() or not pattern.match(_cmdprog.name):
                continue
            cmdprog = str((root / _cmdprog).absolute())
            provname = pathlib.Path(cmdprog).name.rsplit('.', 1)[0]

            class _prov(_ExtcapInterfaceProvider):
                name = provname
            conf.ifaces.register_provider(functools.partial(_prov, cmdprog=cmdprog))