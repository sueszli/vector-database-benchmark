"""
Example demonstrating the emitting of events using the emit() method.

The "!" in the event name is to supress a warning for connecting to an
event that is not known beforehand (i.e. there is no corresponding
property or emitter).
"""
from flexx import event

class Basic(event.Component):

    @event.reaction('!foo')
    def on_foo(self, *events):
        if False:
            while True:
                i = 10
        print('foo reaction called with %i events' % len(events))

    @event.reaction('!bar')
    def on_bar(self, *events):
        if False:
            for i in range(10):
                print('nop')
        print('bar reaction called with %i events' % len(events))
b = Basic()
b.emit('foo', {})
b.emit('foo', {})
b.emit('bar', {})
b.emit('spam', {})
event.loop.iter()