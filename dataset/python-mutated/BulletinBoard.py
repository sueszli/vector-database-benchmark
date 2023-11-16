"""Contains the BulletinBoard class."""
__all__ = ['BulletinBoard']
from direct.directnotify import DirectNotifyGlobal
from direct.showbase.MessengerGlobal import messenger

class BulletinBoard:
    """This class implements a global location for key/value pairs to be
    stored. Intended to prevent coders from putting global variables directly
    on showbase, so that potential name collisions can be more easily
    detected."""
    notify = DirectNotifyGlobal.directNotify.newCategory('BulletinBoard')

    def __init__(self):
        if False:
            while True:
                i = 10
        self._dict = {}

    def get(self, postName, default=None):
        if False:
            i = 10
            return i + 15
        return self._dict.get(postName, default)

    def has(self, postName):
        if False:
            for i in range(10):
                print('nop')
        return postName in self._dict

    def getEvent(self, postName):
        if False:
            return 10
        return 'bboard-%s' % postName

    def getRemoveEvent(self, postName):
        if False:
            return 10
        return 'bboard-remove-%s' % postName

    def post(self, postName, value=None):
        if False:
            while True:
                i = 10
        if postName in self._dict:
            BulletinBoard.notify.warning('changing %s from %s to %s' % (postName, self._dict[postName], value))
        self.update(postName, value)

    def update(self, postName, value):
        if False:
            print('Hello World!')
        'can use this to set value the first time'
        if postName in self._dict:
            BulletinBoard.notify.info('update: posting %s' % postName)
        self._dict[postName] = value
        messenger.send(self.getEvent(postName))

    def remove(self, postName):
        if False:
            return 10
        if postName in self._dict:
            del self._dict[postName]
            messenger.send(self.getRemoveEvent(postName))

    def removeIfEqual(self, postName, value):
        if False:
            while True:
                i = 10
        if self.has(postName):
            if self.get(postName) == value:
                self.remove(postName)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        str = 'Bulletin Board Contents\n'
        str += '======================='
        for postName in sorted(self._dict):
            str += '\n%s: %s' % (postName, self._dict[postName])
        return str