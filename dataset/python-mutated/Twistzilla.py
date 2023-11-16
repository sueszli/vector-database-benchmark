from AppKit import *
from Foundation import *
from PyObjCTools import AppHelper, NibClassBuilder
from twisted.internet import _threadedselect
_threadedselect.install()
import sys
import urlparse
from twisted.internet import protocol, reactor
from twisted.python import log
from twisted.web import http
NibClassBuilder.extractClasses('MainMenu')

class TwistzillaClient(http.HTTPClient):

    def __init__(self, delegate, urls):
        if False:
            i = 10
            return i + 15
        self.urls = urls
        self.delegate = delegate

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.sendCommand('GET', str(self.urls[2]))
        self.sendHeader('Host', '%s:%d' % (self.urls[0], self.urls[1]))
        self.sendHeader('User-Agent', 'CocoaTwistzilla')
        self.endHeaders()

    def handleResponse(self, data):
        if False:
            return 10
        self.delegate.gotResponse_(data)

class MyAppDelegate(NibClassBuilder.AutoBaseClass):

    def gotResponse_(self, html):
        if False:
            i = 10
            return i + 15
        s = self.resultTextField.textStorage()
        s.replaceCharactersInRange_withString_((0, s.length()), html)
        self.progressIndicator.stopAnimation_(self)

    def doTwistzillaFetch_(self, sender):
        if False:
            for i in range(10):
                print('nop')
        s = self.resultTextField.textStorage()
        s.deleteCharactersInRange_((0, s.length()))
        self.progressIndicator.startAnimation_(self)
        u = urlparse.urlparse(self.messageTextField.stringValue())
        pos = u[1].find(':')
        if pos == -1:
            (host, port) = (u[1], 80)
        else:
            (host, port) = (u[1][:pos], int(u[1][pos + 1:]))
        if u[2] == '':
            fname = '/'
        else:
            fname = u[2]
        host = host.encode('utf8')
        fname = fname.encode('utf8')
        protocol.ClientCreator(reactor, TwistzillaClient, self, (host, port, fname)).connectTCP(host, port).addErrback(lambda f: self.gotResponse_(f.getBriefTraceback()))

    def applicationDidFinishLaunching_(self, aNotification):
        if False:
            while True:
                i = 10
        '\n        Invoked by NSApplication once the app is done launching and\n        immediately before the first pass through the main event\n        loop.\n        '
        self.messageTextField.setStringValue_('http://www.twistedmatrix.com/')
        reactor.interleave(AppHelper.callAfter)

    def applicationShouldTerminate_(self, sender):
        if False:
            while True:
                i = 10
        if reactor.running:
            reactor.addSystemEventTrigger('after', 'shutdown', AppHelper.stopEventLoop)
            reactor.stop()
            return False
        return True
if __name__ == '__main__':
    log.startLogging(sys.stdout)
    AppHelper.runEventLoop()