""" Test that shows that the socket module can properly be used.

"""
import signal
import socket
import sys

def onTimeout(_signum, _frame):
    if False:
        while True:
            i = 10
    sys.exit(0)
try:
    signal.signal(signal.SIGALRM, onTimeout)
    signal.alarm(1)
except AttributeError:
    pass
socket.getfqdn('1.1.1.1')