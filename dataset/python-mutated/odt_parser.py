import zipfile
import xml.etree.ElementTree as ET
from .utils import BaseParser

class Parser(BaseParser):
    """Extract text from open document files.
    """

    def extract(self, filename, **kwargs):
        if False:
            while True:
                i = 10
        with open(filename, 'rb') as stream:
            zip_stream = zipfile.ZipFile(stream)
            self.content = ET.fromstring(zip_stream.read('content.xml'))
        return self.to_string()

    def to_string(self):
        if False:
            i = 10
            return i + 15
        ' Converts the document to a string. '
        buff = u''
        for child in self.content.iter():
            if child.tag in [self.qn('text:p'), self.qn('text:h')]:
                buff += self.text_to_string(child) + '\n'
        if buff:
            buff = buff[:-1]
        return buff

    def text_to_string(self, element):
        if False:
            for i in range(10):
                print('nop')
        buff = u''
        if element.text is not None:
            buff += element.text
        for child in element:
            if child.tag == self.qn('text:tab'):
                buff += '\t'
                if child.tail is not None:
                    buff += child.tail
            elif child.tag == self.qn('text:s'):
                buff += u' '
                if child.get(self.qn('text:c')) is not None:
                    buff += u' ' * (int(child.get(self.qn('text:c'))) - 1)
                if child.tail is not None:
                    buff += child.tail
            else:
                buff += self.text_to_string(child)
        if element.tail is not None:
            buff += element.tail
        return buff

    def qn(self, namespace):
        if False:
            print('Hello World!')
        'Connect tag prefix to longer namespace'
        nsmap = {'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
        spl = namespace.split(':')
        return '{{{}}}{}'.format(nsmap[spl[0]], spl[1])