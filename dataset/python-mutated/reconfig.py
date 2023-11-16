import os
import platform
import signal
from twisted.internet import defer
from twisted.internet import reactor
from buildbot.scripts.logwatcher import BuildmasterTimeoutError
from buildbot.scripts.logwatcher import LogWatcher
from buildbot.scripts.logwatcher import ReconfigError
from buildbot.util import in_reactor
from buildbot.util import rewrap

class Reconfigurator:

    @defer.inlineCallbacks
    def run(self, basedir, quiet, timeout=None):
        if False:
            while True:
                i = 10
        if platform.system() in ('Windows', 'Microsoft'):
            print('Reconfig (through SIGHUP) is not supported on Windows.')
            return None
        with open(os.path.join(basedir, 'twistd.pid'), 'rt', encoding='utf-8') as f:
            self.pid = int(f.read().strip())
        if quiet:
            os.kill(self.pid, signal.SIGHUP)
            return None
        self.sent_signal = False
        reactor.callLater(0.2, self.sighup)
        lw = LogWatcher(os.path.join(basedir, 'twistd.log'), timeout=timeout)
        try:
            yield lw.start()
            print('Reconfiguration appears to have completed successfully')
            return 0
        except BuildmasterTimeoutError:
            print('Never saw reconfiguration finish.')
        except ReconfigError:
            print(rewrap("                Reconfiguration failed. Please inspect the master.cfg file for\n                errors, correct them, then try 'buildbot reconfig' again.\n                "))
        except IOError:
            self.sighup()
        except Exception as e:
            print(f'Error while following twistd.log: {e}')
        return 1

    def sighup(self):
        if False:
            return 10
        if self.sent_signal:
            return
        print(f'sending SIGHUP to process {self.pid}')
        self.sent_signal = True
        os.kill(self.pid, signal.SIGHUP)

@in_reactor
def reconfig(config):
    if False:
        return 10
    basedir = config['basedir']
    quiet = config['quiet']
    timeout = config.get('progress_timeout', None)
    if timeout is not None:
        try:
            timeout = float(timeout)
        except ValueError:
            print('Progress timeout must be a number')
            return 1
    r = Reconfigurator()
    return r.run(basedir, quiet, timeout=timeout)