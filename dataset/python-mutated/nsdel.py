import sys
import libxml2
instance = '<?xml version="1.0"?>\n<tag xmlns:foo=\'urn:foo\' xmlns:bar=\'urn:bar\' xmlns:baz=\'urn:baz\' />'

def namespaceDefs(node):
    if False:
        for i in range(10):
            print('nop')
    n = node.nsDefs()
    while n:
        yield n
        n = n.next

def checkNamespaceDefs(node, count):
    if False:
        print('Hello World!')
    nsList = list(namespaceDefs(node))
    if len(nsList) != count:
        raise Exception('Error: saw %d namespace declarations.  Expected %d' % (len(nsList), count))
libxml2.debugMemory(1)
doc = libxml2.parseDoc(instance)
node = doc.getRootElement()
checkNamespaceDefs(node, 3)
ns = node.removeNsDef('urn:bar')
checkNamespaceDefs(node, 2)
ns.freeNsList()
doc.freeDoc()
doc = libxml2.parseDoc(instance)
node = doc.getRootElement()
checkNamespaceDefs(node, 3)
ns = node.removeNsDef(None)
checkNamespaceDefs(node, 0)
ns.freeNsList()
doc.freeDoc()
doc = libxml2.newDoc('1.0')
root = doc.newChild(None, 'root', None)
namespace = root.newNs('http://example.com/sample', 's')
child = root.newChild(namespace, 'child', None)
root.removeNsDef('http://example.com/sample')
doc.reconciliateNs(root)
namespace.freeNsList()
doc.serialize()
doc.freeDoc()
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()