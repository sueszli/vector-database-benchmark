"""This sub-module is responsible for handling scanning agents."""
import glob
import os
import random
import re
import subprocess
import time
from ivre import utils

class Agent:
    """An Agent instance is a (possibly remote) scanner."""

    def __init__(self, host, remotepathbase, localpathbase, name=None, usetor=False, maxwaiting=60):
        if False:
            i = 10
            return i + 15
        self.host = host
        self.remotepathbase = remotepathbase
        self.localpathbase = localpathbase
        self.maxwaiting = maxwaiting
        if host is None:
            self.rsyncbase = remotepathbase
        else:
            self.rsyncbase = '%s:%s' % (host, remotepathbase)
        if self.rsyncbase[-1] not in ':/':
            self.rsyncbase += '/'
        if name is None:
            self.name = localpathbase.lstrip('./')
        else:
            self.name = name
        if usetor:
            self.rsync = ['torify', 'rsync']
        else:
            self.rsync = ['rsync']
        self.campaigns = []

    @classmethod
    def from_string(cls, string, localbase='', maxwaiting=60):
        if False:
            i = 10
            return i + 15
        'Builds an Agent instance from a description string of the\n        form [tor:][hostname:]path.\n\n        '
        string = string.split(':', 1)
        if string[0].lower() == 'tor':
            string = string[1].split(':', 1)
            usetor = True
        else:
            usetor = False
        if len(string) == 1:
            return cls(None, string[0], os.path.join(localbase, string[0].replace('/', '_')), maxwaiting=maxwaiting)
        return cls(string[0], string[1], os.path.join(localbase, '%s_%s' % (string[0].replace('@', '_'), string[1].replace('/', '_'))), usetor=usetor, maxwaiting=maxwaiting)

    def get_local_path(self, dirname):
        if False:
            print('Hello World!')
        'Get local storage path for directory `dirname`.'
        return os.path.join(self.localpathbase, dirname) + '/'

    def get_remote_path(self, dirname):
        if False:
            return 10
        'Get remote storage path for directory `dirname` as an rsync\n        address.\n\n        '
        if dirname and dirname[-1] != '/':
            dirname += '/'
        return self.rsyncbase + dirname

    def create_local_dirs(self):
        if False:
            print('Hello World!')
        'Create local directories used to manage the agent'
        for dirname in ['input', 'remoteinput', 'remotecur', 'remoteoutput', 'remotedata']:
            utils.makedirs(self.get_local_path(dirname))

    def may_receive(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the number of targets that can be sent to the agent\n        (based on the total number of targets currently on hold and\n        the `maxwaiting` attribute value).\n\n        '
        curwaiting = sum((len(os.listdir(self.get_local_path(p))) for p in ['input', 'remoteinput']))
        return self.maxwaiting - curwaiting

    def add_target(self, category, addr):
        if False:
            while True:
                i = 10
        'Add a new target (locally), given its category and address\n        (technically, addr can be a network or a hostname that can be\n        resolved from the agent).\n\n        '
        with open(os.path.join(self.get_local_path('input'), '%s.%s' % (category, addr.replace('/', '_'))), 'w', encoding='utf8') as fdesc:
            fdesc.write('%s\n' % addr)
            return True
        return False

    def sync(self):
        if False:
            i = 10
            return i + 15
        'Synchronize the local and remote directories, and the\n        relevant `Campaign`s.\n\n        '
        subprocess.call(self.rsync + ['-a', self.get_local_path('input'), self.get_local_path('remoteinput')])
        subprocess.call(self.rsync + ['-a', '--remove-source-files', self.get_local_path('input'), self.get_remote_path('input')])
        subprocess.call(self.rsync + ['-a', '--delete', self.get_remote_path('input'), self.get_local_path('remoteinput')])
        subprocess.call(self.rsync + ['-a', '--delete', self.get_remote_path('cur'), self.get_local_path('remotecur')])
        subprocess.call(self.rsync + ['-a', '--remove-source-files', self.get_remote_path('output'), self.get_local_path('remoteoutput')])
        subprocess.call(self.rsync + ['-a', '--remove-source-files', self.get_remote_path('data'), self.get_local_path('remotedata')])
        for campaign in self.campaigns:
            campaign.sync(self)

class Campaign:
    """A Campaign instance is basically the association of a targets
    container (an instance of `target.Target`) with a list of agents
    to run the scans.

    """

    def __init__(self, targets, category, agents, outputpath, visiblecategory=None, maxfeed=None, sleep=2, storedown=True):
        if False:
            for i in range(10):
                print('nop')
        self.targets = targets
        self.targiter = iter(targets)
        self.category = category
        for agent in agents:
            agent.campaigns.append(self)
        self.agents = agents
        self.outputpath = outputpath
        if visiblecategory is None:
            self.visiblecategory = ''.join((chr(random.randrange(65, 91)) for _ in range(10)))
        else:
            self.visiblecategory = visiblecategory
        self.maxfeed = maxfeed
        self.sleep = sleep
        self.storedown = storedown

    def sync(self, agent):
        if False:
            while True:
                i = 10
        'This function should only be called from `agent.sync()`\n        method. It stores the results of terminated scans according to\n        the target status.\n\n        '
        for remfname in glob.glob(os.path.join(agent.get_local_path('remoteoutput'), self.visiblecategory + '.*.xml*')):
            locfname = os.path.basename(remfname).split('.', 4)
            locfname[0] = self.category
            status = 'unknown'
            with utils.open_file(remfname) as remfdesc:
                remfcontent = remfdesc.read()
                if b'<status state="up"' in remfcontent:
                    status = 'up'
                elif b'<status state="down"' in remfcontent:
                    if not self.storedown:
                        remfdesc.close()
                        os.unlink(remfname)
                        continue
                    status = 'down'
                del remfcontent
            locfname = os.path.join(self.outputpath, locfname[0], status, re.sub('[/@:]', '_', agent.name), *locfname[1:])
            utils.makedirs(os.path.dirname(locfname))
            os.rename(remfname, locfname)
        for remfname in glob.glob(os.path.join(agent.get_local_path('remotedata'), self.visiblecategory + '.*.tar*')):
            locfname = os.path.basename(remfname).split('.', 4)
            locfname[0] = self.category
            locfname = os.path.join(self.outputpath, locfname[0], 'data', re.sub('[/@:]', '_', agent.name), *locfname[1:])
            utils.makedirs(os.path.dirname(locfname))
            os.rename(remfname, locfname)

    def feed(self, agent, maxnbr=None):
        if False:
            for i in range(10):
                print('nop')
        'Send targets to scan to `agent`, depending on how many it\n        can receive.\n\n        '
        for _ in range(max(agent.may_receive(), maxnbr or 0)):
            addr = utils.int2ip(next(self.targiter))
            with open(os.path.join(agent.get_local_path('input'), '%s.%s' % (self.visiblecategory, addr)), 'w', encoding='utf8') as fdesc:
                fdesc.write('%s\n' % addr)

    def feedloop(self):
        if False:
            for i in range(10):
                print('nop')
        'Feed periodically the agents affected to the `Campaign`\n        (`self.agents`).\n\n        '
        while True:
            for agent in self.agents:
                try:
                    self.feed(agent, maxnbr=self.maxfeed)
                except StopIteration:
                    return
            time.sleep(self.sleep)

def syncloop(agents, sleep=2):
    if False:
        for i in range(10):
            print('nop')
    'Synchronize periodically the `agents`.'
    while True:
        for agent in agents:
            agent.sync()
        time.sleep(sleep)