import time
from scapy.contrib.automotive import log_automotive
from scapy.contrib.automotive.gm.gmlan import GMLAN, GMLAN_SA, GMLAN_RD, GMLAN_TD, GMLAN_PM, GMLAN_RMBA
from scapy.config import conf
from scapy.packet import Packet
from scapy.supersocket import SuperSocket
from scapy.contrib.isotp import ISOTPSocket
from scapy.utils import PeriodicSenderThread
from typing import Optional, cast, Callable
__all__ = ['GMLAN_TesterPresentSender', 'GMLAN_InitDiagnostics', 'GMLAN_GetSecurityAccess', 'GMLAN_RequestDownload', 'GMLAN_TransferData', 'GMLAN_TransferPayload', 'GMLAN_ReadMemoryByAddress', 'GMLAN_BroadcastSocket']
log_automotive.info('"conf.contribs[\'GMLAN\'][\'treat-response-pending-as-answer\']" set to True). This is required by the GMLAN-Utils module to operate correctly.')
try:
    conf.contribs['GMLAN']['treat-response-pending-as-answer'] = False
except KeyError:
    conf.contribs['GMLAN'] = {'treat-response-pending-as-answer': False}

def _check_response(resp):
    if False:
        i = 10
        return i + 15
    if resp is None:
        log_automotive.debug('Timeout.')
        return False
    log_automotive.debug('%s', repr(resp))
    return resp.service != 127

class GMLAN_TesterPresentSender(PeriodicSenderThread):

    def __init__(self, sock, pkt=GMLAN(service='TesterPresent'), interval=2):
        if False:
            i = 10
            return i + 15
        ' Thread to send GMLAN TesterPresent packets periodically\n\n        :param sock: socket where packet is sent periodically\n        :param pkt: packet to send\n        :param interval: interval between two packets\n        '
        PeriodicSenderThread.__init__(self, sock, pkt, interval)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while not self._stopped.is_set() and (not self._socket.closed):
            for p in self._pkts:
                self._socket.sr1(p, verbose=False, timeout=0.1)
                self._stopped.wait(timeout=self._interval)
                if self._stopped.is_set() or self._socket.closed:
                    break

def GMLAN_InitDiagnostics(sock, broadcast_socket=None, timeout=1, retry=0, unittest=False):
    if False:
        return 10
    ' Send messages to put an ECU into diagnostic/programming state.\n\n    :param sock: socket for communication.\n    :param broadcast_socket: socket for broadcasting. If provided some message\n                             will be sent as broadcast. Recommended when used\n                             on a network with several ECUs.\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :param unittest: disable delays\n    :return: True on success else False\n    '

    def _send_and_check_response(sock, req, timeout):
        if False:
            return 10
        log_automotive.debug('Sending %s', repr(req))
        resp = sock.sr1(req, timeout=timeout, verbose=False)
        return _check_response(resp)
    retry = abs(retry)
    while retry >= 0:
        retry -= 1
        p = GMLAN(service='DisableNormalCommunication')
        if broadcast_socket is None:
            if not _send_and_check_response(sock, p, timeout):
                continue
        else:
            log_automotive.debug('Sending %s as broadcast', repr(p))
            broadcast_socket.send(p)
        if not unittest:
            time.sleep(0.05)
        p = GMLAN(service='ReportProgrammingState')
        if not _send_and_check_response(sock, p, timeout):
            continue
        p = GMLAN() / GMLAN_PM(subfunction='requestProgrammingMode')
        if not _send_and_check_response(sock, p, timeout):
            continue
        if not unittest:
            time.sleep(0.05)
        p = GMLAN() / GMLAN_PM(subfunction='enableProgrammingMode')
        log_automotive.debug('Sending %s', repr(p))
        sock.sr1(p, timeout=0.001, verbose=False)
        return True
    return False

def GMLAN_GetSecurityAccess(sock, key_function, level=1, timeout=None, retry=0, unittest=False):
    if False:
        return 10
    ' Authenticate on ECU. Implements Seey-Key procedure.\n\n    :param sock: socket to send the message on.\n    :param key_function: function implementing the key algorithm.\n    :param level: level of access\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :param unittest: disable internal delays\n    :return: True on success.\n    '
    retry = abs(retry)
    if key_function is None:
        return False
    if level % 2 == 0:
        log_automotive.warning('Parameter Error: Level must be an odd number.')
        return False
    while retry >= 0:
        retry -= 1
        request = GMLAN() / GMLAN_SA(subfunction=level)
        log_automotive.debug('Requesting seed..')
        resp = sock.sr1(request, timeout=timeout, verbose=False)
        if not _check_response(resp):
            if resp is not None and resp.returnCode == 55 and retry:
                log_automotive.debug('RequiredTimeDelayNotExpired. Wait 10s.')
                if not unittest:
                    time.sleep(10)
            log_automotive.debug('Negative Response.')
            continue
        seed = cast(Packet, resp).securitySeed
        if seed == 0:
            log_automotive.debug('ECU security already unlocked. (seed is 0x0000)')
            return True
        keypkt = GMLAN() / GMLAN_SA(subfunction=level + 1, securityKey=key_function(seed))
        log_automotive.debug('Responding with key..')
        resp = sock.sr1(keypkt, timeout=timeout, verbose=False)
        if resp is None:
            log_automotive.debug('Timeout.')
            continue
        log_automotive.debug('%s', repr(resp))
        if resp.service == 103:
            log_automotive.debug('SecurityAccess granted.')
            return True
        elif resp.service == 127 and resp.returnCode == 53:
            log_automotive.debug('Key invalid')
            continue
    return False

def GMLAN_RequestDownload(sock, length, timeout=None, retry=0):
    if False:
        while True:
            i = 10
    " Send RequestDownload message.\n\n        Usually used before calling TransferData.\n\n    :param sock: socket to send the message on.\n    :param length: value for the message's parameter 'unCompressedMemorySize'.\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :return: True on success\n    "
    retry = abs(retry)
    while retry >= 0:
        pkt = GMLAN() / GMLAN_RD(memorySize=length)
        resp = sock.sr1(pkt, timeout=timeout, verbose=False)
        if _check_response(resp):
            return True
        retry -= 1
        if retry >= 0:
            log_automotive.debug('Retrying..')
    return False

def GMLAN_TransferData(sock, addr, payload, maxmsglen=None, timeout=None, retry=0):
    if False:
        i = 10
        return i + 15
    ' Send TransferData message.\n\n    Usually used after calling RequestDownload.\n\n    :param sock: socket to send the message on.\n    :param addr: destination memory address on the ECU.\n    :param payload: data to be sent.\n    :param maxmsglen: maximum length of a single iso-tp message.\n                      default: maximum length\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :return: True on success.\n    '
    retry = abs(retry)
    startretry = retry
    scheme = conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme']
    if addr < 0 or addr >= 2 ** (8 * scheme):
        log_automotive.warning('Error: Invalid address %s for scheme %s', hex(addr), str(scheme))
        return False
    if maxmsglen is None or maxmsglen <= 0 or maxmsglen > 4093 - scheme:
        maxmsglen = 4093 - scheme
    maxmsglen = cast(int, maxmsglen)
    for i in range(0, len(payload), maxmsglen):
        retry = startretry
        while True:
            if len(payload[i:]) > maxmsglen:
                transdata = payload[i:i + maxmsglen]
            else:
                transdata = payload[i:]
            pkt = GMLAN() / GMLAN_TD(startingAddress=addr + i, dataRecord=transdata)
            resp = sock.sr1(pkt, timeout=timeout, verbose=False)
            if _check_response(resp):
                break
            retry -= 1
            if retry >= 0:
                log_automotive.debug('Retrying..')
            else:
                return False
    return True

def GMLAN_TransferPayload(sock, addr, payload, maxmsglen=None, timeout=None, retry=0):
    if False:
        while True:
            i = 10
    ' Send data by using GMLAN services.\n\n    :param sock: socket to send the data on.\n    :param addr: destination memory address on the ECU.\n    :param payload: data to be sent.\n    :param maxmsglen: maximum length of a single iso-tp message.\n                      default: maximum length\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :return: True on success.\n    '
    if not GMLAN_RequestDownload(sock, len(payload), timeout=timeout, retry=retry):
        return False
    if not GMLAN_TransferData(sock, addr, payload, maxmsglen=maxmsglen, timeout=timeout, retry=retry):
        return False
    return True

def GMLAN_ReadMemoryByAddress(sock, addr, length, timeout=None, retry=0):
    if False:
        for i in range(10):
            print('nop')
    ' Read data from ECU memory.\n\n    :param sock: socket to send the data on.\n    :param addr: source memory address on the ECU.\n    :param length: bytes to read.\n    :param timeout: timeout for sending, receiving or sniffing packages.\n    :param retry: number of retries in case of failure.\n    :return: bytes red or None\n    '
    retry = abs(retry)
    scheme = conf.contribs['GMLAN']['GMLAN_ECU_AddressingScheme']
    if addr < 0 or addr >= 2 ** (8 * scheme):
        log_automotive.warning('Error: Invalid address %s for scheme %s', hex(addr), str(scheme))
        return None
    if length <= 0 or length > 4094 - scheme:
        log_automotive.warning('Error: Invalid length %s for scheme %s. Choose between 0x1 and %s', hex(length), str(scheme), hex(4094 - scheme))
        return None
    while retry >= 0:
        pkt = GMLAN() / GMLAN_RMBA(memoryAddress=addr, memorySize=length)
        resp = sock.sr1(pkt, timeout=timeout, verbose=False)
        if _check_response(resp):
            return cast(Packet, resp).dataRecord
        retry -= 1
        if retry >= 0:
            log_automotive.debug('Retrying..')
    return None

def GMLAN_BroadcastSocket(interface):
    if False:
        i = 10
        return i + 15
    ' Returns a GMLAN broadcast socket using interface.\n\n    :param interface: interface name\n    :return: ISOTPSocket configured as GMLAN Broadcast Socket\n    '
    return ISOTPSocket(interface, tx_id=257, rx_id=0, basecls=GMLAN, ext_address=254, padding=True)