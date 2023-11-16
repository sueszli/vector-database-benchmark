import os
import random
import time
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from impacket import LOG
from threading import Thread

class TargetsProcessor:

    def __init__(self, targetListFile=None, singleTarget=None, protocolClients=None, randomize=False):
        if False:
            i = 10
            return i + 15
        self.finishedAttacks = []
        self.protocolClients = protocolClients
        if targetListFile is None:
            self.filename = None
            self.originalTargets = self.processTarget(singleTarget, protocolClients)
        else:
            self.filename = targetListFile
            self.originalTargets = []
            self.readTargets()
        if randomize is True:
            random.shuffle(self.originalTargets)
        self.generalCandidates = [x for x in self.originalTargets if x.username is None]
        self.namedCandidates = [x for x in self.originalTargets if x.username is not None]

    @staticmethod
    def processTarget(target, protocolClients):
        if False:
            return 10
        if target.find('://') <= 0:
            return [urlparse('smb://%s' % target)]
        retVals = []
        if target[:3].upper() == 'ALL':
            strippedTarget = target[3:]
            for protocol in protocolClients:
                retVals.append(urlparse('%s%s' % (protocol, strippedTarget)))
            return retVals
        else:
            return [urlparse(target)]

    def readTargets(self):
        if False:
            return 10
        try:
            with open(self.filename, 'r') as f:
                self.originalTargets = []
                for line in f:
                    target = line.strip()
                    if target != '' and target[0] != '#':
                        self.originalTargets.extend(self.processTarget(target, self.protocolClients))
        except IOError as e:
            LOG.error('Could not open file: %s - %s', self.filename, str(e))
        if len(self.originalTargets) == 0:
            LOG.critical('Warning: no valid targets specified!')
        self.generalCandidates = [x for x in self.originalTargets if x not in self.finishedAttacks and x.username is None]
        self.namedCandidates = [x for x in self.originalTargets if x not in self.finishedAttacks and x.username is not None]

    def logTarget(self, target, gotRelay=False, gotUsername=None):
        if False:
            return 10
        if gotRelay is True:
            if target.username is not None:
                self.finishedAttacks.append(target)
            elif gotUsername is not None:
                newTarget = urlparse('%s://%s@%s%s' % (target.scheme, gotUsername.replace('/', '\\'), target.netloc, target.path))
                self.finishedAttacks.append(newTarget)

    def getTarget(self, identity=None, multiRelay=True):
        if False:
            i = 10
            return i + 15
        if identity is not None and len(self.namedCandidates) > 0:
            for target in self.namedCandidates:
                if target.username is not None:
                    if target.username.upper() == identity.replace('/', '\\'):
                        self.namedCandidates.remove(target)
                        return target
                    if target.username.find('\\') < 0:
                        if target.username.upper() == identity.split('/')[1]:
                            self.namedCandidates.remove(target)
                            return target
        if len(self.generalCandidates) > 0:
            if identity is not None:
                for target in self.generalCandidates:
                    tmpTarget = '%s://%s@%s' % (target.scheme, identity.replace('/', '\\'), target.netloc)
                    match = [x for x in self.finishedAttacks if x.geturl().upper() == tmpTarget.upper()]
                    if len(match) == 0:
                        self.generalCandidates.remove(target)
                        return target
                LOG.debug('No more targets for user %s' % identity)
                return None
            elif multiRelay == False:
                for target in self.generalCandidates:
                    match = [x for x in self.finishedAttacks if x.hostname == target.netloc]
                    if len(match) == 0:
                        self.generalCandidates.remove(target)
                        return target
                LOG.debug('No more targets')
                return None
            else:
                return self.generalCandidates.pop()
        elif len(self.originalTargets) > 0:
            self.generalCandidates = [x for x in self.originalTargets if x not in self.finishedAttacks and x.username is None]
        if len(self.generalCandidates) == 0:
            if len(self.namedCandidates) == 0:
                LOG.info('All targets processed!')
            elif identity is not None:
                LOG.debug('No more targets for user %s' % identity)
            return None
        else:
            return self.getTarget(identity, multiRelay)

class TargetsFileWatcher(Thread):

    def __init__(self, targetprocessor):
        if False:
            for i in range(10):
                print('nop')
        Thread.__init__(self)
        self.targetprocessor = targetprocessor
        self.lastmtime = os.stat(self.targetprocessor.filename).st_mtime

    def run(self):
        if False:
            return 10
        while True:
            mtime = os.stat(self.targetprocessor.filename).st_mtime
            if mtime > self.lastmtime:
                LOG.info('Targets file modified - refreshing')
                self.lastmtime = mtime
                self.targetprocessor.readTargets()
            time.sleep(1.0)