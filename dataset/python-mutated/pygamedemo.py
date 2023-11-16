"""
Demonstration of how L{twisted.internet._threadedselect} might be used (this is
not an example showing the best way to integrate Twisted with pygame).
"""
from twisted.internet import _threadedselect
_threadedselect.install()
import pygame
from pygame.locals import *
from twisted.internet import reactor
try:
    import pygame.fastevent as eventmodule
except ImportError:
    import pygame.event as eventmodule
TWISTEDEVENT = USEREVENT

def postTwistedEvent(func):
    if False:
        print('Hello World!')
    eventmodule.post(eventmodule.Event(TWISTEDEVENT, iterateTwisted=func))

def helloWorld():
    if False:
        for i in range(10):
            print('nop')
    print('hello, world')
    reactor.callLater(1, helloWorld)
reactor.callLater(1, helloWorld)

def twoSecondsPassed():
    if False:
        print('Hello World!')
    print('two seconds passed')
reactor.callLater(2, twoSecondsPassed)

def eventIterator():
    if False:
        while True:
            i = 10
    while True:
        yield eventmodule.wait()
        while True:
            event = eventmodule.poll()
            if event.type == NOEVENT:
                break
            else:
                yield event

def main():
    if False:
        for i in range(10):
            print('nop')
    pygame.init()
    if hasattr(eventmodule, 'init'):
        eventmodule.init()
    screen = pygame.display.set_mode((300, 300))
    reactor.interleave(postTwistedEvent)
    shouldQuit = []
    reactor.addSystemEventTrigger('after', 'shutdown', shouldQuit.append, True)
    for event in eventIterator():
        if event.type == TWISTEDEVENT:
            event.iterateTwisted()
            if shouldQuit:
                break
        elif event.type == QUIT:
            reactor.stop()
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            reactor.stop()
    pygame.quit()
if __name__ == '__main__':
    main()