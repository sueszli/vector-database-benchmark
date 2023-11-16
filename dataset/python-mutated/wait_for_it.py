import sys
from twisted.internet.defer import Deferred
from twisted.web.template import Element, XMLString, flatten, renderer
sample = XMLString('\n    <div xmlns:t="http://twistedmatrix.com/ns/twisted.web.template/0.1">\n    Before waiting ...\n    <span t:render="wait"></span>\n    ... after waiting.\n    </div>\n    ')

class WaitForIt(Element):

    def __init__(self):
        if False:
            return 10
        Element.__init__(self, loader=sample)
        self.deferred = Deferred()

    @renderer
    def wait(self, request, tag):
        if False:
            for i in range(10):
                print('nop')
        return self.deferred.addCallback(lambda aValue: tag('A value: ' + repr(aValue)))

def done(ignore):
    if False:
        return 10
    print('[[[Deferred fired.]]]')
print('[[[Rendering the template.]]]')
it = WaitForIt()
flatten(None, it, sys.stdout.write).addCallback(done)
print('[[[In progress... now firing the Deferred.]]]')
it.deferred.callback('<value>')
print('[[[All done.]]]')