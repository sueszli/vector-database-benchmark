import xml.sax
from xml import sax
import defusedxml.sax

class ExampleContentHandler(xml.sax.ContentHandler):

    def __init__(self):
        if False:
            print('Hello World!')
        xml.sax.ContentHandler.__init__(self)

    def startElement(self, name, attrs):
        if False:
            print('Hello World!')
        print('start:', name)

    def endElement(self, name):
        if False:
            return 10
        print('end:', name)

    def characters(self, content):
        if False:
            print('Hello World!')
        print('chars:', content)

def main():
    if False:
        print('Hello World!')
    xmlString = "<note>\n<to>Tove</to>\n<from>Jani</from>\n<heading>Reminder</heading>\n<body>Don't forget me this weekend!</body>\n</note>"
    xml.sax.parseString(xmlString, ExampleContentHandler())
    xml.sax.parse('notaxmlfilethatexists.xml', ExampleContentHandler())
    sax.parseString(xmlString, ExampleContentHandler())
    sax.parse('notaxmlfilethatexists.xml', ExampleContentHandler)
    defusedxml.sax.parseString(xmlString, ExampleContentHandler())
    xml.sax.make_parser()
    sax.make_parser()
    print('nothing')
    defusedxml.sax.make_parser()
if __name__ == '__main__':
    main()