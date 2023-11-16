import libxml2
import libxml2mod
import sys

def error(msg, data):
    if False:
        i = 10
        return i + 15
    pass
libxml2.debugMemory(1)
dtd = '<!ELEMENT foo EMPTY>'
instance = '<?xml version="1.0"?>\n<foo></foo>'
dtd = libxml2.parseDTD(None, 'test.dtd')
ctxt = libxml2.newValidCtxt()
libxml2mod.xmlSetValidErrors(ctxt._o, error, error)
doc = libxml2.parseDoc(instance)
ret = doc.validateDtd(ctxt, dtd)
if ret != 1:
    print('error doing DTD validation')
    sys.exit(1)
doc.freeDoc()
dtd.freeDtd()
del dtd
del ctxt
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()