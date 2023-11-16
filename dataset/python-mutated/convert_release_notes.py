import sys
import mistune
import mistune.renderers
print(sys.argv[1])
with open(sys.argv[1], 'r') as source_file:
    source = source_file.read()
html = mistune.create_markdown()
print()
print('HTML')
print('=====================================')
print('<p><em>From the <a href="">GitHub release page</a>:</em></p>\n')
print(html(source))

class AdafruitBBCodeRenderer(mistune.renderers.BaseRenderer):

    def placeholder(self):
        if False:
            i = 10
            return i + 15
        return ''

    def paragraph(self, text):
        if False:
            while True:
                i = 10
        return text + '\n\n'

    def block_text(self, text):
        if False:
            for i in range(10):
                print('nop')
        return text

    def text(self, text):
        if False:
            return 10
        return text

    def link(self, link, title, text):
        if False:
            while True:
                i = 10
        return '[url={}]{}[/url]'.format(link, title)

    def autolink(self, link, is_email):
        if False:
            i = 10
            return i + 15
        if not is_email:
            return '[url={}]{}[/url]'.format(link, link)
        return link

    def heading(self, text, level):
        if False:
            print('Hello World!')
        return '[b][size=150]{}[/size][/b]\n'.format(text)

    def codespan(self, text):
        if False:
            return 10
        return '[color=#E74C3C][size=95]{}[/size][/color]'.format(text)

    def list_item(self, text, level):
        if False:
            for i in range(10):
                print('nop')
        return '[*]{}[/*]\n'.format(text.strip())

    def list(self, text, ordered, level, start=None):
        if False:
            i = 10
            return i + 15
        ordered_indicator = '=' if ordered else ''
        return '[list{}]\n{}[/list]'.format(ordered_indicator, text)

    def double_emphasis(self, text):
        if False:
            return 10
        return '[b]{}[/b]'.format(text)

    def emphasis(self, text):
        if False:
            for i in range(10):
                print('nop')
        return '[i]{}[/i]'.format(text)

    def strong(self, text):
        if False:
            i = 10
            return i + 15
        return '[b]{}[/b]'.format(text)

    def finalize(self, data):
        if False:
            while True:
                i = 10
        return ''.join(data)
bbcode = mistune.create_markdown(renderer=AdafruitBBCodeRenderer())
print()
print('BBCode')
print('=====================================')
print('[i]From the [url=]GitHub release page[/url]:[/i]\n')
print(bbcode(source))