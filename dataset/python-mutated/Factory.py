"""Contains the Factory class."""
__all__ = ['Factory']
from direct.directnotify.DirectNotifyGlobal import directNotify

class Factory:
    """This class manages a list of object types and their corresponding constructors.
    Objects may be created on-demand from their type. Object types may be any hashable
    piece of unique data (such as a string).

    This class is intended to be derived from. Subclasses should call self._registerTypes
    to set up type constructors."""
    notify = directNotify.newCategory('Factory')

    def __init__(self):
        if False:
            print('Hello World!')
        self._type2ctor = {}

    def create(self, type, *args, **kwArgs):
        if False:
            return 10
        return self._type2ctor[type](*args, **kwArgs)

    def _registerType(self, type, ctor):
        if False:
            i = 10
            return i + 15
        if type in self._type2ctor:
            self.notify.debug('replacing %s ctor %s with %s' % (type, self._type2ctor[type], ctor))
        self._type2ctor[type] = ctor

    def _registerTypes(self, type2ctor):
        if False:
            i = 10
            return i + 15
        for (type, ctor) in list(type2ctor.items()):
            self._registerType(type, ctor)

    def nullCtor(self, *args, **kwArgs):
        if False:
            print('Hello World!')
        return None