import sys
import libxml2
try:
    import StringIO
    str_io = StringIO.StringIO
except:
    import io
    str_io = io.StringIO
docstr = '<?xml version=\'1.0\'?>\n<!DOCTYPE doc [\n<!ENTITY tst "<p>test</p>">\n]>\n<doc>&tst;</doc>'
libxml2.debugMemory(1)
f = str_io(docstr)
input = libxml2.inputBuffer(f)
reader = input.newTextReader('test_noent')
ret = reader.Read()
if ret != 1:
    print('Error reading to root')
    sys.exit(1)
if reader.Name() == 'doc' or reader.NodeType() == 10:
    ret = reader.Read()
if ret != 1:
    print('Error reading to root')
    sys.exit(1)
if reader.Name() != 'doc' or reader.NodeType() != 1:
    print('test_normal: Error reading the root element')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_normal: Error reading to the entity')
    sys.exit(1)
if reader.Name() != 'tst' or reader.NodeType() != 5:
    print('test_normal: Error reading the entity')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_normal: Error reading to the end of root')
    sys.exit(1)
if reader.Name() != 'doc' or reader.NodeType() != 15:
    print('test_normal: Error reading the end of the root element')
    sys.exit(1)
ret = reader.Read()
if ret != 0:
    print('test_normal: Error detecting the end')
    sys.exit(1)
f = str_io(docstr)
input = libxml2.inputBuffer(f)
reader = input.newTextReader('test_noent')
reader.SetParserProp(libxml2.PARSER_SUBST_ENTITIES, 1)
ret = reader.Read()
if ret != 1:
    print('Error reading to root')
    sys.exit(1)
if reader.Name() == 'doc' or reader.NodeType() == 10:
    ret = reader.Read()
if ret != 1:
    print('Error reading to root')
    sys.exit(1)
if reader.Name() != 'doc' or reader.NodeType() != 1:
    print('test_noent: Error reading the root element')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_noent: Error reading to the entity content')
    sys.exit(1)
if reader.Name() != 'p' or reader.NodeType() != 1:
    print('test_noent: Error reading the p element from entity')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_noent: Error reading to the text node')
    sys.exit(1)
if reader.NodeType() != 3 or reader.Value() != 'test':
    print('test_noent: Error reading the text node')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_noent: Error reading to the end of p element')
    sys.exit(1)
if reader.Name() != 'p' or reader.NodeType() != 15:
    print('test_noent: Error reading the end of the p element')
    sys.exit(1)
ret = reader.Read()
if ret != 1:
    print('test_noent: Error reading to the end of root')
    sys.exit(1)
if reader.Name() != 'doc' or reader.NodeType() != 15:
    print('test_noent: Error reading the end of the root element')
    sys.exit(1)
ret = reader.Read()
if ret != 0:
    print('test_noent: Error detecting the end')
    sys.exit(1)
s = '<!DOCTYPE struct [\n<!ENTITY simplestruct2.ent SYSTEM "simplestruct2.ent">\n]>\n<struct>&simplestruct2.ent;</struct>\n'
expect = '10 struct 0 0\n1 struct 0 0\n1 descr 1 1\n15 struct 0 0\n'
res = ''
simplestruct2_ent = '<descr/>'

def myResolver(URL, ID, ctxt):
    if False:
        i = 10
        return i + 15
    if URL == 'simplestruct2.ent':
        return str_io(simplestruct2_ent)
    return None
libxml2.setEntityLoader(myResolver)
input = libxml2.inputBuffer(str_io(s))
reader = input.newTextReader('test3')
reader.SetParserProp(libxml2.PARSER_SUBST_ENTITIES, 1)
while reader.Read() == 1:
    res = res + '%s %s %d %d\n' % (reader.NodeType(), reader.Name(), reader.Depth(), reader.IsEmptyElement())
if res != expect:
    print('test3 failed: unexpected output')
    print(res)
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