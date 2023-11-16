import os
import sys
import re
import json
import logging
LOG = logging.getLogger(__name__)

class CheckProcs(object):
    myPid = 0
    state = ''
    name = ''
    pid = 0
    allProcs = []
    interestingProcs = []
    procDir = '/proc'
    debug = False

    def __init__(self):
        if False:
            while True:
                i = 10
        self.myPid = os.getpid()

    def setup(self, debug=False, pidlist=False):
        if False:
            i = 10
            return i + 15
        self.debug = debug
        self.pidlist = pidlist
        if debug is True:
            print('Debug is on')
        self.allProcs = [procs for procs in os.listdir(self.procDir) if procs.isdigit() and int(procs) != int(self.myPid)]

    def process(self, criteria):
        if False:
            for i in range(10):
                print('nop')
        for p in self.allProcs:
            try:
                fh = open(self.procDir + '/' + p + '/stat')
                pInfo = fh.readline().split()
                cmdfh = open(self.procDir + '/' + p + '/cmdline')
                cmd = cmdfh.readline()
                pInfo[1] = cmd
            except Exception:
                LOG.exception("Error: can't find file or read data.")
                continue
            finally:
                cmdfh.close()
                fh.close()
            if criteria == 'state':
                if pInfo[2] == self.state:
                    self.interestingProcs.append(pInfo)
            elif criteria == 'name':
                if re.search(self.name, pInfo[1]):
                    self.interestingProcs.append(pInfo)
            elif criteria == 'pid':
                if pInfo[0] == self.pid:
                    self.interestingProcs.append(pInfo)

    def byState(self, state):
        if False:
            print('Hello World!')
        self.state = state
        self.process(criteria='state')
        self.show()

    def byPid(self, pid):
        if False:
            while True:
                i = 10
        self.pid = pid
        self.process(criteria='pid')
        self.show()

    def byName(self, name):
        if False:
            return 10
        self.name = name
        self.process(criteria='name')
        self.show()

    def run(self, foo, criteria):
        if False:
            return 10
        if foo == 'state':
            self.byState(criteria)
        elif foo == 'name':
            self.byName(criteria)
        elif foo == 'pid':
            self.byPid(criteria)

    def show(self):
        if False:
            print('Hello World!')
        prettyOut = {}
        if len(self.interestingProcs) > 0:
            for proc in self.interestingProcs:
                prettyOut[proc[0]] = proc[1]
        if self.pidlist is True:
            pidlist = ' '.join(prettyOut.keys())
            sys.stderr.write(pidlist)
        print(json.dumps(prettyOut))
if __name__ == '__main__':
    if 'pidlist' in sys.argv:
        pidlist = True
    else:
        pidlist = False
    foo = CheckProcs()
    foo.setup(debug=False, pidlist=pidlist)
    foo.run(sys.argv[1], sys.argv[2])