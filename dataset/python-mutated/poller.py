from buildbot.changes.base import ReconfigurablePollingChangeSource
from buildbot.util import bytes2unicode
from buildbot.util import unicode2bytes
from buildbot.www.hooks.base import BaseHookHandler

class PollingHandler(BaseHookHandler):

    def getChanges(self, req):
        if False:
            while True:
                i = 10
        change_svc = req.site.master.change_svc
        poll_all = b'poller' not in req.args
        allow_all = True
        allowed = []
        if isinstance(self.options, dict) and b'allowed' in self.options:
            allow_all = False
            allowed = self.options[b'allowed']
        pollers = []
        for source in change_svc:
            if not isinstance(source, ReconfigurablePollingChangeSource):
                continue
            if not hasattr(source, 'name'):
                continue
            if not poll_all and unicode2bytes(source.name) not in req.args[b'poller']:
                continue
            if not allow_all and unicode2bytes(source.name) not in allowed:
                continue
            pollers.append(source)
        if not poll_all:
            missing = set(req.args[b'poller']) - set((unicode2bytes(s.name) for s in pollers))
            if missing:
                raise ValueError(f"Could not find pollers: {bytes2unicode(b','.join(missing))}")
        for p in pollers:
            p.force()
        return ([], None)
poller = PollingHandler