import sys
import libxml2
libxml2.debugMemory(1)
result = ''

def processNode(reader):
    if False:
        while True:
            i = 10
    global result
    result = result + '%d %d %s %d\n' % (reader.Depth(), reader.NodeType(), reader.Name(), reader.IsEmptyElement())
docstr = '<foo>\n<label>some text</label>\n<item>100</item>\n</foo>'
expect = '0 1 foo 0\n1 14 #text 0\n1 1 label 0\n2 3 #text 0\n1 15 label 0\n1 14 #text 0\n1 1 item 0\n2 3 #text 0\n1 15 item 0\n1 14 #text 0\n0 15 foo 0\n'
result = ''
reader = libxml2.readerForDoc(docstr, 'test1', None, 0)
ret = reader.Read()
while ret == 1:
    processNode(reader)
    ret = reader.Read()
if ret != 0:
    print('Error parsing the document test1')
    sys.exit(1)
if result != expect:
    print('Unexpected result for test1')
    print(result)
    sys.exit(1)
docstr = '<foo>\n<label>some text</label>\n<item>1000</item>\n</foo>'
expect = '0 1 foo 0\n1 14 #text 0\n1 1 label 0\n2 3 #text 0\n1 15 label 0\n1 14 #text 0\n1 1 item 0\n2 3 #text 0\n1 15 item 0\n1 14 #text 0\n0 15 foo 0\n'
result = ''
reader.NewDoc(docstr, 'test2', None, 0)
ret = reader.Read()
while ret == 1:
    processNode(reader)
    ret = reader.Read()
if ret != 0:
    print('Error parsing the document test2')
    sys.exit(1)
if result != expect:
    print('Unexpected result for test2')
    print(result)
    sys.exit(1)
del reader
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()