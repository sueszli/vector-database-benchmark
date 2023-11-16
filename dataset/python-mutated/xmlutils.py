"""
Utilities for XML generation/parsing.
"""
import re
from xml.sax.saxutils import XMLGenerator

class UnserializableContentError(ValueError):
    pass

class SimplerXMLGenerator(XMLGenerator):

    def addQuickElement(self, name, contents=None, attrs=None):
        if False:
            return 10
        'Convenience method for adding an element with no children'
        if attrs is None:
            attrs = {}
        self.startElement(name, attrs)
        if contents is not None:
            self.characters(contents)
        self.endElement(name)

    def characters(self, content):
        if False:
            print('Hello World!')
        if content and re.search('[\\x00-\\x08\\x0B-\\x0C\\x0E-\\x1F]', content):
            raise UnserializableContentError('Control characters are not supported in XML 1.0')
        XMLGenerator.characters(self, content)

    def startElement(self, name, attrs):
        if False:
            i = 10
            return i + 15
        sorted_attrs = dict(sorted(attrs.items())) if attrs else attrs
        super().startElement(name, sorted_attrs)