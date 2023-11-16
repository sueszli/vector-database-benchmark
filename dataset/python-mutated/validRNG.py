import libxml2
import sys
ARG = 'test string'

class ErrorHandler:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.errors = []

    def handler(self, msg, data):
        if False:
            i = 10
            return i + 15
        if data != ARG:
            raise Exception('Error handler did not receive correct argument')
        self.errors.append(msg)
libxml2.debugMemory(1)
schema = '<?xml version="1.0"?>\n<element name="foo"\n         xmlns="http://relaxng.org/ns/structure/1.0"\n         xmlns:a="http://relaxng.org/ns/annotation/1.0"\n         xmlns:ex1="http://www.example.com/n1"\n         xmlns:ex2="http://www.example.com/n2">\n  <a:documentation>A foo element.</a:documentation>\n  <element name="ex1:bar1">\n    <empty/>\n  </element>\n  <element name="ex2:bar2">\n    <empty/>\n  </element>\n</element>\n'
valid = '<?xml version="1.0"?>\n<foo><pre1:bar1 xmlns:pre1="http://www.example.com/n1"/><pre2:bar2 xmlns:pre2="http://www.example.com/n2"/></foo>'
invalid = '<?xml version="1.0"?>\n<foo><pre1:bar1 xmlns:pre1="http://www.example.com/n1">bad</pre1:bar1><pre2:bar2 xmlns:pre2="http://www.example.com/n2"/></foo>'
rngp = libxml2.relaxNGNewMemParserCtxt(schema, len(schema))
rngs = rngp.relaxNGParse()
ctxt = rngs.relaxNGNewValidCtxt()
e = ErrorHandler()
ctxt.setValidityErrorHandler(e.handler, e.handler, ARG)
doc = libxml2.parseDoc(valid)
ret = doc.relaxNGValidateDoc(ctxt)
if ret != 0 or e.errors:
    print('error doing RelaxNG validation')
    sys.exit(1)
doc.freeDoc()
doc = libxml2.parseDoc(invalid)
ret = doc.relaxNGValidateDoc(ctxt)
if ret == 0 or not e.errors:
    print('Error: document supposed to be RelaxNG invalid')
    sys.exit(1)
doc.freeDoc()
del rngp
del rngs
del ctxt
libxml2.relaxNGCleanupTypes()
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()