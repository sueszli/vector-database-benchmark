__license__ = 'GPL 3'
__copyright__ = '2011, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
from calibre.ebooks.oeb.base import OEB_DOCS, XPath, barename
from calibre.utils.unsmarten import unsmarten_text

class UnsmartenPunctuation:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.html_tags = XPath('descendant::h:*')

    def unsmarten(self, root):
        if False:
            for i in range(10):
                print('nop')
        for x in self.html_tags(root):
            if not barename(x.tag) == 'pre':
                if getattr(x, 'text', None):
                    x.text = unsmarten_text(x.text)
                if getattr(x, 'tail', None) and x.tail:
                    x.tail = unsmarten_text(x.tail)

    def __call__(self, oeb, context):
        if False:
            for i in range(10):
                print('nop')
        bx = XPath('//h:body')
        for x in oeb.manifest.items:
            if x.media_type in OEB_DOCS:
                for body in bx(x.data):
                    self.unsmarten(body)