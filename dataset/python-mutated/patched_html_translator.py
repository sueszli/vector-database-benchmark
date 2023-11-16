from sphinx.util.docutils import is_html5_writer_available
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.html5 import HTML5Translator

class PatchedHTMLTranslator(HTML5Translator if is_html5_writer_available() else HTMLTranslator):

    def starttag(self, node, tagname, *args, **attrs):
        if False:
            for i in range(10):
                print('nop')
        if tagname == 'a' and 'target' not in attrs and ('external' in attrs.get('class', '') or 'external' in attrs.get('classes', [])):
            attrs['target'] = '_blank'
            attrs['ref'] = 'noopener noreferrer'
        return super().starttag(node, tagname, *args, **attrs)

def setup(app):
    if False:
        i = 10
        return i + 15
    app.set_translator('html', PatchedHTMLTranslator)