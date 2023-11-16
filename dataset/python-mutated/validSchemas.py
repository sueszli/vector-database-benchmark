import libxml2
import sys
ARG = 'test string'

class ErrorHandler:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.errors = []

    def handler(self, msg, data):
        if False:
            for i in range(10):
                print('nop')
        if data != ARG:
            raise Exception('Error handler did not receive correct argument')
        self.errors.append(msg)
libxml2.debugMemory(1)
schema = '<?xml version="1.0" encoding="iso-8859-1"?>\n<schema xmlns = "http://www.w3.org/2001/XMLSchema">\n\t<element name = "Customer">\n\t\t<complexType>\n\t\t\t<sequence>\n\t\t\t\t<element name = "FirstName" type = "string" />\n\t\t\t\t<element name = "MiddleInitial" type = "string" />\n\t\t\t\t<element name = "LastName" type = "string" />\n\t\t\t</sequence>\n\t\t\t<attribute name = "customerID" type = "integer" />\n\t\t</complexType>\n\t</element>\n</schema>'
valid = '<?xml version="1.0" encoding="iso-8859-1"?>\n<Customer customerID = "24332">\n\t<FirstName>Raymond</FirstName>\n\t<MiddleInitial>G</MiddleInitial>\n\t<LastName>Bayliss</LastName>\n</Customer>\n'
invalid = '<?xml version="1.0" encoding="iso-8859-1"?>\n<Customer customerID = "24332">\n\t<MiddleInitial>G</MiddleInitial>\n\t<LastName>Bayliss</LastName>\n</Customer>\n'
e = ErrorHandler()
ctxt_parser = libxml2.schemaNewMemParserCtxt(schema, len(schema))
ctxt_schema = ctxt_parser.schemaParse()
ctxt_valid = ctxt_schema.schemaNewValidCtxt()
ctxt_valid.setValidityErrorHandler(e.handler, e.handler, ARG)
doc = libxml2.parseDoc(valid)
ret = doc.schemaValidateDoc(ctxt_valid)
if ret != 0 or e.errors:
    print('error doing schema validation')
    sys.exit(1)
doc.freeDoc()
doc = libxml2.parseDoc(invalid)
ret = doc.schemaValidateDoc(ctxt_valid)
if ret == 0 or not e.errors:
    print('Error: document supposer to be schema invalid')
    sys.exit(1)
doc.freeDoc()
del ctxt_parser
del ctxt_schema
del ctxt_valid
libxml2.schemaCleanupTypes()
libxml2.cleanupParser()
if libxml2.debugMemory(1) == 0:
    print('OK')
else:
    print('Memory leak %d bytes' % libxml2.debugMemory(1))
    libxml2.dumpMemory()