import signal
import sys
import time

def signal_handler(signal, frame):
    if False:
        while True:
            i = 10
    print('exiting cleanly', flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)
print('ready', flush=True)
timeout = 3.0
while timeout > 0:
    time.sleep(0.05)
    timeout -= 0.05
print('error: signal not received', file=sys.stderr)
sys.exit(1)