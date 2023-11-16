import sys
import libxml2
try:
    import StringIO
    str_io = StringIO.StringIO
except:
    import io
    str_io = io.StringIO

def testSimpleBufferWrites():
    if False:
        print('Hello World!')
    f = str_io()
    buf = libxml2.createOutputBuffer(f, 'ISO-8859-1')
    buf.write(3, 'foo')
    buf.writeString('bar')
    buf.close()
    if f.getvalue() != 'foobar':
        print('Failed to save to StringIO')
        sys.exit(1)

def testSaveDocToBuffer():
    if False:
        while True:
            i = 10
    '\n    Regression test for bug #154294.\n    '
    input = '<foo>Hello</foo>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<foo>Hello</foo>\n'
    f = str_io()
    buf = libxml2.createOutputBuffer(f, 'UTF-8')
    doc = libxml2.parseDoc(input)
    doc.saveFileTo(buf, 'UTF-8')
    doc.freeDoc()
    if f.getvalue() != expected:
        print('xmlDoc.saveFileTo() call failed.')
        print('     got: %s' % repr(f.getvalue()))
        print('expected: %s' % repr(expected))
        sys.exit(1)

def testSaveFormattedDocToBuffer():
    if False:
        for i in range(10):
            print('nop')
    input = '<outer><inner>Some text</inner><inner/></outer>'
    expected = ('<?xml version="1.0" encoding="UTF-8"?>\n<outer><inner>Some text</inner><inner/></outer>\n', '<?xml version="1.0" encoding="UTF-8"?>\n<outer>\n  <inner>Some text</inner>\n  <inner/>\n</outer>\n')
    doc = libxml2.parseDoc(input)
    for i in (0, 1):
        f = str_io()
        buf = libxml2.createOutputBuffer(f, 'UTF-8')
        doc.saveFormatFileTo(buf, 'UTF-8', i)
        if f.getvalue() != expected[i]:
            print('xmlDoc.saveFormatFileTo() call failed.')
            print('     got: %s' % repr(f.getvalue()))
            print('expected: %s' % repr(expected[i]))
            sys.exit(1)
    doc.freeDoc()

def testSaveIntoOutputBuffer():
    if False:
        for i in range(10):
            print('nop')
    '\n    Similar to the previous two tests, except this time we invoke the save\n    methods on the output buffer object and pass in an XML node object.\n    '
    input = '<foo>Hello</foo>'
    expected = '<?xml version="1.0" encoding="UTF-8"?>\n<foo>Hello</foo>\n'
    f = str_io()
    doc = libxml2.parseDoc(input)
    buf = libxml2.createOutputBuffer(f, 'UTF-8')
    buf.saveFileTo(doc, 'UTF-8')
    if f.getvalue() != expected:
        print('outputBuffer.saveFileTo() call failed.')
        print('     got: %s' % repr(f.getvalue()))
        print('expected: %s' % repr(expected))
        sys.exit(1)
    f = str_io()
    buf = libxml2.createOutputBuffer(f, 'UTF-8')
    buf.saveFormatFileTo(doc, 'UTF-8', 1)
    if f.getvalue() != expected:
        print('outputBuffer.saveFormatFileTo() call failed.')
        print('     got: %s' % repr(f.getvalue()))
        print('expected: %s' % repr(expected))
        sys.exit(1)
    doc.freeDoc()
if __name__ == '__main__':
    libxml2.debugMemory(1)
    testSimpleBufferWrites()
    testSaveDocToBuffer()
    testSaveFormattedDocToBuffer()
    testSaveIntoOutputBuffer()
    libxml2.cleanupParser()
    if libxml2.debugMemory(1) == 0:
        print('OK')
    else:
        print('Memory leak %d bytes' % libxml2.debugMemory(1))
        libxml2.dumpMemory()