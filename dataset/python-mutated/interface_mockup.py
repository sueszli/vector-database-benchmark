import os
import subprocess
import sys
from platform import python_implementation
from scapy.main import load_layer, load_contrib
from scapy.config import conf
from scapy.error import log_runtime, Scapy_Exception
from scapy.consts import LINUX
load_layer('can', globals_dict=globals())
conf.contribs['CAN']['swap-bytes'] = False
iface0 = 'vcan0'
iface1 = 'vcan1'
try:
    _root = os.geteuid() == 0
except AttributeError:
    _root = False
_not_pypy = 'pypy' not in python_implementation().lower()
_socket_can_support = False

def test_and_setup_socket_can(iface_name):
    if False:
        for i in range(10):
            print('nop')
    if 0 != subprocess.call(('cansend %s 000#' % iface_name).split()):
        if 0 != subprocess.call('modprobe vcan'.split()):
            raise Exception('modprobe vcan failed')
        if 0 != subprocess.call(('ip link add name %s type vcan' % iface_name).split()):
            log_runtime.debug('add %s failed: Maybe it was already up?' % iface_name)
        if 0 != subprocess.call(('ip link set dev %s up' % iface_name).split()):
            raise Exception('could not bring up %s' % iface_name)
    if 0 != subprocess.call(('cansend %s 000#12' % iface_name).split()):
        raise Exception("cansend doesn't work")
    sys.__stderr__.write('SocketCAN setup done!\n')
if LINUX and _root and _not_pypy:
    try:
        test_and_setup_socket_can(iface0)
        test_and_setup_socket_can(iface1)
        log_runtime.debug('CAN should work now')
        _socket_can_support = True
    except Exception as e:
        sys.__stderr__.write('ERROR %s!\n' % e)
sys.__stderr__.write('SocketCAN support: %s\n' % _socket_can_support)
if _socket_can_support:
    from scapy.contrib.cansocket_native import *
    new_can_socket = NativeCANSocket
    new_can_socket0 = lambda : NativeCANSocket(iface0)
    new_can_socket1 = lambda : NativeCANSocket(iface1)
    can_socket_string_list = ['-c', iface0]
    sys.__stderr__.write('Using NativeCANSocket\n')
else:
    from scapy.contrib.cansocket_python_can import *
    new_can_socket = lambda iface: PythonCANSocket(bustype='virtual', channel=iface)
    new_can_socket0 = lambda : PythonCANSocket(bustype='virtual', channel=iface0, timeout=0.01)
    new_can_socket1 = lambda : PythonCANSocket(bustype='virtual', channel=iface1, timeout=0.01)
    sys.__stderr__.write('Using PythonCANSocket virtual\n')
s = new_can_socket(iface0)
s.close()
del s
s = new_can_socket(iface1)
s.close()
del s

def cleanup_interfaces():
    if False:
        while True:
            i = 10
    '\n    Helper function to remove virtual CAN interfaces after test\n\n    :return: True on success\n    '
    if _socket_can_support:
        if 0 != subprocess.call(['ip', 'link', 'delete', iface0]):
            raise Exception('%s could not be deleted' % iface0)
        if 0 != subprocess.call(['ip', 'link', 'delete', iface1]):
            raise Exception('%s could not be deleted' % iface1)
    return True

def drain_bus(iface=iface0, assert_empty=True):
    if False:
        while True:
            i = 10
    '\n    Utility function for draining a can interface,\n    asserting that no packets are there\n\n    :param iface: Interface name to drain\n    :param assert_empty: If true, raise exception in case packets were received\n    '
    with new_can_socket(iface) as s:
        pkts = s.sniff(timeout=0.1)
        if assert_empty and (not len(pkts) == 0):
            raise Scapy_Exception('Error in drain_bus. Packets found but no packets expected!')
drain_bus(iface0)
drain_bus(iface1)
log_runtime.debug('CAN sockets should work now')
ISOTP_KERNEL_MODULE_AVAILABLE = False

def exit_if_no_isotp_module():
    if False:
        while True:
            i = 10
    '\n    Helper function to exit a test case if ISOTP kernel module is not available\n    '
    if not ISOTP_KERNEL_MODULE_AVAILABLE:
        err = 'TEST SKIPPED: can-isotp not available\n'
        sys.__stderr__.write(err)
        warning("Can't test ISOTPNativeSocket because kernel module isn't loaded")
        sys.exit(0)
if LINUX and _root and _socket_can_support:
    p1 = subprocess.Popen(['lsmod'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['grep', '^can_isotp'], stdout=subprocess.PIPE, stdin=p1.stdout)
    p1.stdout.close()
    if p1.wait() == 0 and p2.wait() == 0 and (b'can_isotp' in p2.stdout.read()):
        p = subprocess.Popen(['isotpsend', '-s1', '-d0', iface0], stdin=subprocess.PIPE)
        p.communicate(b'01')
        if p.returncode == 0:
            ISOTP_KERNEL_MODULE_AVAILABLE = True
conf.contribs['ISOTP'] = {'use-can-isotp-kernel-module': ISOTP_KERNEL_MODULE_AVAILABLE}
import importlib
if 'scapy.contrib.isotp' in sys.modules:
    importlib.reload(scapy.contrib.isotp)
load_contrib('isotp', globals_dict=globals())
if ISOTP_KERNEL_MODULE_AVAILABLE:
    if ISOTPSocket is not ISOTPNativeSocket:
        raise Scapy_Exception('Error in ISOTPSocket import!')
elif ISOTPSocket is not ISOTPSoftSocket:
    raise Scapy_Exception('Error in ISOTPSocket import!')