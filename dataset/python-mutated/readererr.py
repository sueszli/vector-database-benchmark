import libxml2
import sys
try:
    import StringIO
    str_io = StringIO.StringIO
except:
    import io
    str_io = io.StringIO
libxml2.debugMemory(1)
expect = '--> (3) test1:1:xmlns: URI foo is not absolute\n--> (4) test1:1:Opening and ending tag mismatch: c line 1 and a\n'
err = ''

def myErrorHandler(arg, msg, severity, locator):
    if False:
        print('Hello World!')
    global err
    err = err + '%s (%d) %s:%d:%s' % (arg, severity, locator.BaseURI(), locator.LineNumber(), msg)
f = str_io('<a xmlns="foo"><b b1="b1"/><c>content of c</a>')
input = libxml2.inputBuffer(f)
reader = input.newTextReader('test1')
reader.SetErrorHandler(myErrorHandler, '-->')
while reader.Read() == 1:
    pass
if err != expect:
    print('error')
    print('received %s' % err)
    print('expected %s' % expect)
    sys.exit(1)
reader.SetErrorHandler(None, None)
if reader.GetErrorHandler() != (None, None):
    print('GetErrorHandler failed')
    sys.exit(1)
del f
del input
del reader
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()