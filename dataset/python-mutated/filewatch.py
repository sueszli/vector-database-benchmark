from twisted.application import internet

def watch(fp):
    if False:
        for i in range(10):
            print('nop')
    fp.seek(fp.tell())
    for line in fp.readlines():
        sys.stdout.write(line)
import sys
from twisted.internet import reactor
s = internet.TimerService(0.1, watch, open(sys.argv[1]))
s.startService()
reactor.run()
s.stopService()