import getopt
import sys
import signal
import re
import traceback
from ast import literal_eval
from scapy.config import conf
from scapy.consts import LINUX
if not LINUX or conf.use_pypy:
    conf.contribs['CANSocket'] = {'use-python-can': True}
from scapy.contrib.isotp import ISOTPSocket
from scapy.contrib.cansocket import CANSocket, PYTHON_CAN
from scapy.contrib.automotive.obd.obd import OBD
from scapy.contrib.automotive.obd.scanner import OBD_Scanner, OBD_S01_Enumerator, OBD_S02_Enumerator, OBD_S03_Enumerator, OBD_S06_Enumerator, OBD_S07_Enumerator, OBD_S08_Enumerator, OBD_S09_Enumerator, OBD_S0A_Enumerator

def signal_handler(sig, frame):
    if False:
        return 10
    print('Interrupting scan!')
    sys.exit(0)

def usage(is_error):
    if False:
        for i in range(10):
            print('nop')
    print("usage:\tobdscanner [-i|--interface] [-c|--channel] [-b|--bitrate]\n                                [-a|--python-can_args] [-h|--help]\n                                [-s|--source] [-d|--destination]\n                                [-t|--timeout] [-f|--full]\n                                [-v|--verbose]\n\n    Scan for all possible obd service classes and their subfunctions.\n\n    optional arguments:\n    -c, --channel               python-can channel or Linux SocketCAN interface name\n\n    additional required arguments for WINDOWS or Python 2:\n    -i, --interface             python-can interface for the scan.\n                                Depends on used interpreter and system,\n                                see examples below. Any python-can interface can\n                                be provided. Please see:\n                                https://python-can.readthedocs.io for\n                                further interface examples.\n    optional arguments:\n    -a, --python-can_args       Additional arguments for a python-can Bus object.\n    -h, --help                  show this help message and exit\n    -s, --source                ISOTP-socket source id (hex)\n    -d, --destination           ISOTP-socket destination id (hex)\n    -t, --timeout               Timeout after which the scanner proceeds to next service [seconds]\n    -f, --full                  Full scan on id services\n    -v, --verbose               Display information during scan\n    -1                          Scan OBD Service 01\n    -2                          Scan OBD Service 02\n    -3                          Scan OBD Service 03\n    -6                          Scan OBD Service 06\n    -7                          Scan OBD Service 07\n    -8                          Scan OBD Service 08\n    -9                          Scan OBD Service 09\n    -A                          Scan OBD Service 0A\n\n    Example of use:\n\n    Python2 or Windows:\n    python2 -m scapy.tools.automotive.obdscanner --interface=pcan --channel=PCAN_USBBUS1 --source=0x070 --destination 0x034\n    python2 -m scapy.tools.automotive.obdscanner --interface vector --channel 0 --source 0x000 --destination 0x734\n    python2 -m scapy.tools.automotive.obdscanner --interface socketcan --channel=can0 --source 0x089 --destination 0x234\n    python2 -m scapy.tools.automotive.obdscanner --interface vector --channel 0 --python-can_args 'bitrate=500000, poll_interval=1' --source=0x070 --destination 0x034\n\n    Python3 on Linux:\n    python3 -m scapy.tools.automotive.obdscanner --channel can0 --source 0x123 --destination 0x456 \n", file=sys.stderr if is_error else sys.stdout)

def get_can_socket(channel, interface, python_can_args):
    if False:
        while True:
            i = 10
    if PYTHON_CAN:
        if python_can_args:
            arg_dict = dict(((k, literal_eval(v)) for (k, v) in (pair.split('=') for pair in re.split(', | |,', python_can_args))))
            return CANSocket(bustype=interface, channel=channel, **arg_dict)
        else:
            return CANSocket(bustype=interface, channel=channel)
    else:
        return CANSocket(channel=channel)

def get_isotp_socket(csock, source, destination):
    if False:
        i = 10
        return i + 15
    return ISOTPSocket(csock, source, destination, basecls=OBD, padding=True)

def run_scan(isock, enumerators, full_scan, verbose, timeout):
    if False:
        for i in range(10):
            print('nop')
    s = OBD_Scanner(isock, test_cases=enumerators, full_scan=full_scan, debug=verbose, timeout=timeout)
    print('Starting OBD-Scan...')
    s.scan()
    s.show_testcases()

def main():
    if False:
        for i in range(10):
            print('nop')
    channel = None
    interface = None
    source = 2016
    destination = 2015
    timeout = 0.1
    full_scan = False
    verbose = False
    python_can_args = None
    enumerators = []
    conf.verb = -1
    options = getopt.getopt(sys.argv[1:], 'i:c:s:d:a:t:hfv1236789A', ['interface=', 'channel=', 'source=', 'destination=', 'help', 'timeout=', 'python-can_args=', 'full', 'verbose'])
    try:
        for (opt, arg) in options[0]:
            if opt in ('-i', '--interface'):
                interface = arg
            elif opt in ('-c', '--channel'):
                channel = arg
            elif opt in ('-a', '--python-can_args'):
                python_can_args = arg
            elif opt in ('-s', '--source'):
                source = int(arg, 16)
            elif opt in ('-d', '--destination'):
                destination = int(arg, 16)
            elif opt in ('-h', '--help'):
                usage(False)
                sys.exit(0)
            elif opt in ('-t', '--timeout'):
                timeout = float(arg)
            elif opt in ('-f', '--full'):
                full_scan = True
            elif opt == '-1':
                enumerators += [OBD_S01_Enumerator]
            elif opt == '-2':
                enumerators += [OBD_S02_Enumerator]
            elif opt == '-3':
                enumerators += [OBD_S03_Enumerator]
            elif opt == '-6':
                enumerators += [OBD_S06_Enumerator]
            elif opt == '-7':
                enumerators += [OBD_S07_Enumerator]
            elif opt == '-8':
                enumerators += [OBD_S08_Enumerator]
            elif opt == '-9':
                enumerators += [OBD_S09_Enumerator]
            elif opt == '-A':
                enumerators += [OBD_S0A_Enumerator]
            elif opt in ('-v', '--verbose'):
                verbose = True
    except getopt.GetoptError as msg:
        usage(True)
        print('ERROR:', msg, file=sys.stderr)
        raise SystemExit
    if channel is None or (PYTHON_CAN and interface is None):
        usage(True)
        print('\nPlease provide all required arguments.\n', file=sys.stderr)
        sys.exit(1)
    if 0 > source >= 2048 or 0 > destination >= 2048 or source == destination:
        print('The ids must be >= 0 and < 0x800 and not equal.', file=sys.stderr)
        sys.exit(1)
    if 0 > timeout:
        print('The timeout must be a positive value')
        sys.exit(1)
    csock = None
    isock = None
    try:
        csock = get_can_socket(channel, interface, python_can_args)
        isock = get_isotp_socket(csock, source, destination)
        signal.signal(signal.SIGINT, signal_handler)
        run_scan(isock, enumerators, full_scan, verbose, timeout)
    except Exception as e:
        usage(True)
        print("\nSocket couldn't be created. Check your arguments.\n", file=sys.stderr)
        print(e, file=sys.stderr)
        if verbose:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        if isock:
            isock.close()
        if csock:
            csock.close()
if __name__ == '__main__':
    main()