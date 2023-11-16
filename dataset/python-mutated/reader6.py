import sys
import libxml2
try:
    import StringIO
    str_io = StringIO.StringIO
except:
    import io
    str_io = io.StringIO
schema = '<element name="foo" xmlns="http://relaxng.org/ns/structure/1.0"\n         datatypeLibrary="http://www.w3.org/2001/XMLSchema-datatypes">\n  <oneOrMore>\n    <element name="label">\n      <text/>\n    </element>\n    <optional>\n      <element name="opt">\n        <empty/>\n      </element>\n    </optional>\n    <element name="item">\n      <data type="byte"/>\n    </element>\n  </oneOrMore>\n</element>\n'
libxml2.debugMemory(1)
rngp = libxml2.relaxNGNewMemParserCtxt(schema, len(schema))
rngs = rngp.relaxNGParse()
del rngp
docstr = '<foo>\n<label>some text</label>\n<item>100</item>\n</foo>'
f = str_io(docstr)
input = libxml2.inputBuffer(f)
reader = input.newTextReader('correct')
reader.RelaxNGSetSchema(rngs)
ret = reader.Read()
while ret == 1:
    ret = reader.Read()
if ret != 0:
    print('Error parsing the document')
    sys.exit(1)
if reader.IsValid() != 1:
    print('Document failed to validate')
    sys.exit(1)
docstr = '<foo>\n<label>some text</label>\n<item>1000</item>\n</foo>'
err = ''
expect = "Type byte doesn't allow value '1000'\nError validating datatype byte\nElement item failed to validate content\n"

def callback(ctx, str):
    if False:
        return 10
    global err
    err = err + '%s' % str
libxml2.registerErrorHandler(callback, '')
f = str_io(docstr)
input = libxml2.inputBuffer(f)
reader = input.newTextReader('error')
reader.RelaxNGSetSchema(rngs)
ret = reader.Read()
while ret == 1:
    ret = reader.Read()
if ret != 0:
    print('Error parsing the document')
    sys.exit(1)
if reader.IsValid() != 0:
    print('Document failed to detect the validation error')
    sys.exit(1)
if err != expect:
    print('Did not get the expected error message:')
    print(err)
    sys.exit(1)
del f
del input
del reader
del rngs
libxml2.relaxNGCleanupTypes()
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()