import gevent
from Debug import Debug

class GreenletManager:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.greenlets = set()

    def spawnLater(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        greenlet = gevent.spawn_later(*args, **kwargs)
        greenlet.link(lambda greenlet: self.greenlets.remove(greenlet))
        self.greenlets.add(greenlet)
        return greenlet

    def spawn(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        greenlet = gevent.spawn(*args, **kwargs)
        greenlet.link(lambda greenlet: self.greenlets.remove(greenlet))
        self.greenlets.add(greenlet)
        return greenlet

    def stopGreenlets(self, reason='Stopping all greenlets'):
        if False:
            i = 10
            return i + 15
        num = len(self.greenlets)
        gevent.killall(list(self.greenlets), Debug.createNotifyType(reason), block=False)
        return num