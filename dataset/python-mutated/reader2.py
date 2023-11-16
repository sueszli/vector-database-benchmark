import sys
import glob
import string
import libxml2
try:
    import StringIO
    str_io = StringIO.StringIO
except:
    import io
    str_io = io.StringIO
libxml2.debugMemory(1)
err = ''
expect = '../../test/valid/rss.xml:177: element rss: validity error : Element rss does not carry attribute version\n</rss>\n      ^\n../../test/valid/xlink.xml:450: element termdef: validity error : ID dt-arc already defined\n\t<p><termdef id="dt-arc" term="Arc">An <ter\n\t                                  ^\n../../test/valid/xlink.xml:530: validity error : attribute def line 199 references an unknown ID "dt-xlg"\n\n^\n'

def callback(ctx, str):
    if False:
        print('Hello World!')
    global err
    err = err + '%s' % str
libxml2.registerErrorHandler(callback, '')
valid_files = glob.glob('../../test/valid/*.x*')
valid_files.sort()
for file in valid_files:
    if file.find('t8') != -1:
        continue
    if file == '../../test/valid/rss.xml':
        continue
    if file == '../../test/valid/xlink.xml':
        continue
    reader = libxml2.newTextReaderFilename(file)
    reader.SetParserProp(libxml2.PARSER_VALIDATE, 1)
    ret = reader.Read()
    while ret == 1:
        ret = reader.Read()
    if ret != 0:
        print('Error parsing and validating %s' % file)
if err != expect:
    print(err)
s = '\n<!DOCTYPE test [\n<!ELEMENT test (x,b)>\n<!ELEMENT x (c)>\n<!ELEMENT b (#PCDATA)>\n<!ELEMENT c (#PCDATA)>\n<!ENTITY x "<x><c>xxx</c></x>">\n]>\n<test>\n    &x;\n    <b>bbb</b>\n</test>\n'
expect = '10,test\n1,test\n14,#text\n1,x\n1,c\n3,#text\n15,c\n15,x\n14,#text\n1,b\n3,#text\n15,b\n14,#text\n15,test\n'
res = ''
err = ''
input = libxml2.inputBuffer(str_io(s))
reader = input.newTextReader('test2')
reader.SetParserProp(libxml2.PARSER_LOADDTD, 1)
reader.SetParserProp(libxml2.PARSER_DEFAULTATTRS, 1)
reader.SetParserProp(libxml2.PARSER_SUBST_ENTITIES, 1)
reader.SetParserProp(libxml2.PARSER_VALIDATE, 1)
while reader.Read() == 1:
    res = res + '%s,%s\n' % (reader.NodeType(), reader.Name())
if res != expect:
    print('test2 failed: unexpected output')
    print(res)
    sys.exit(1)
if err != '':
    print('test2 failed: validation error found')
    print(err)
    sys.exit(1)
s = '<!DOCTYPE test [\n<!ELEMENT test (x)>\n<!ELEMENT x (#PCDATA)>\n<!ENTITY e SYSTEM "tst.ent">\n]>\n<test>\n  &e;\n</test>\n'
tst_ent = '<x>hello</x>'
expect = '10 test\n1 test\n14 #text\n1 x\n3 #text\n15 x\n14 #text\n15 test\n'
res = ''

def myResolver(URL, ID, ctxt):
    if False:
        return 10
    if URL == 'tst.ent':
        return str_io(tst_ent)
    return None
libxml2.setEntityLoader(myResolver)
input = libxml2.inputBuffer(str_io(s))
reader = input.newTextReader('test3')
reader.SetParserProp(libxml2.PARSER_LOADDTD, 1)
reader.SetParserProp(libxml2.PARSER_DEFAULTATTRS, 1)
reader.SetParserProp(libxml2.PARSER_SUBST_ENTITIES, 1)
reader.SetParserProp(libxml2.PARSER_VALIDATE, 1)
while reader.Read() == 1:
    res = res + '%s %s\n' % (reader.NodeType(), reader.Name())
if res != expect:
    print('test3 failed: unexpected output')
    print(res)
    sys.exit(1)
if err != '':
    print('test3 failed: validation error found')
    print(err)
    sys.exit(1)
s = '<!DOCTYPE test [\n<!ELEMENT test (x, x)>\n<!ELEMENT x (y)>\n<!ELEMENT y (#PCDATA)>\n<!ENTITY x "<x>&y;</x>">\n<!ENTITY y "<y>yyy</y>">\n]>\n<test>\n  &x;\n  &x;\n</test>'
expect = '10 test 0\n1 test 0\n14 #text 1\n1 x 1\n1 y 2\n3 #text 3\n15 y 2\n15 x 1\n14 #text 1\n1 x 1\n1 y 2\n3 #text 3\n15 y 2\n15 x 1\n14 #text 1\n15 test 0\n'
res = ''
err = ''
input = libxml2.inputBuffer(str_io(s))
reader = input.newTextReader('test4')
reader.SetParserProp(libxml2.PARSER_LOADDTD, 1)
reader.SetParserProp(libxml2.PARSER_DEFAULTATTRS, 1)
reader.SetParserProp(libxml2.PARSER_SUBST_ENTITIES, 1)
reader.SetParserProp(libxml2.PARSER_VALIDATE, 1)
while reader.Read() == 1:
    res = res + '%s %s %d\n' % (reader.NodeType(), reader.Name(), reader.Depth())
if res != expect:
    print('test4 failed: unexpected output')
    print(res)
    sys.exit(1)
if err != '':
    print('test4 failed: validation error found')
    print(err)
    sys.exit(1)
s = '<!DOCTYPE test [\n<!ELEMENT test (x, x)>\n<!ELEMENT x (y)>\n<!ELEMENT y (#PCDATA)>\n<!ENTITY x "<x>&y;</x>">\n<!ENTITY y "<y>yyy</y>">\n]>\n<test>\n  &x;\n  &x;\n</test>'
expect = '10 test 0\n1 test 0\n14 #text 1\n5 x 1\n14 #text 1\n5 x 1\n14 #text 1\n15 test 0\n'
res = ''
err = ''
input = libxml2.inputBuffer(str_io(s))
reader = input.newTextReader('test5')
reader.SetParserProp(libxml2.PARSER_VALIDATE, 1)
while reader.Read() == 1:
    res = res + '%s %s %d\n' % (reader.NodeType(), reader.Name(), reader.Depth())
if res != expect:
    print('test5 failed: unexpected output')
    print(res)
if err != '':
    print('test5 failed: validation error found')
    print(err)
del input
del reader
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()