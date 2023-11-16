from html.parser import HTMLParser
from html.entities import name2codepoint

class MyHTMLParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        if False:
            while True:
                i = 10
        print('<%s>' % tag)

    def handle_endtag(self, tag):
        if False:
            return 10
        print('</%s>' % tag)

    def handle_startendtag(self, tag, attrs):
        if False:
            print('Hello World!')
        print('<%s/>' % tag)

    def handle_data(self, data):
        if False:
            print('Hello World!')
        print(data)

    def handle_comment(self, data):
        if False:
            return 10
        print('<!--', data, '-->')

    def handle_entityref(self, name):
        if False:
            print('Hello World!')
        print('&%s;' % name)

    def handle_charref(self, name):
        if False:
            return 10
        print('&#%s;' % name)
parser = MyHTMLParser()
parser.feed('<html>\n<head></head>\n<body>\n<!-- test html parser -->\n    <p>Some <a href="#">html</a> HTML&nbsp;tutorial...<br>END</p>\n</body></html>')