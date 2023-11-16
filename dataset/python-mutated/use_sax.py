from xml.parsers.expat import ParserCreate

class DefaultSaxHandler(object):

    def start_element(self, name, attrs):
        if False:
            while True:
                i = 10
        print('sax:start_element: %s, attrs: %s' % (name, str(attrs)))

    def end_element(self, name):
        if False:
            print('Hello World!')
        print('sax:end_element: %s' % name)

    def char_data(self, text):
        if False:
            for i in range(10):
                print('nop')
        print('sax:char_data: %s' % text)
xml = '<?xml version="1.0"?>\n<ol>\n    <li><a href="/python">Python</a></li>\n    <li><a href="/ruby">Ruby</a></li>\n</ol>\n'
handler = DefaultSaxHandler()
parser = ParserCreate()
parser.StartElementHandler = handler.start_element
parser.EndElementHandler = handler.end_element
parser.CharacterDataHandler = handler.char_data
parser.Parse(xml)