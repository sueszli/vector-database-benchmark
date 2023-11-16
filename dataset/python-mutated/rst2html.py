"""
A minimal front end to the Docutils Publisher, producing HTML.
"""
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass
import docutils
from docutils.core import publish_cmdline, default_description
if True:
    from docutils.parsers.rst.states import Body
    Body.pats['optname'] = '[a-zA-Z0-9][a-zA-Z0-9._-]*'
    Body.pats['longopt'] = '(--|/)%(optname)s([ =]%(optarg)s)?' % Body.pats
    Body.pats['option'] = '(%(shortopt)s|%(longopt)s)' % Body.pats
    Body.patterns['option_marker'] = '%(option)s(, %(option)s)*(  +| ?$)' % Body.pats
description = 'Generates (X)HTML documents from standalone reStructuredText sources.  ' + default_description
from docutils.writers import html4css1

class IESafeHtmlTranslator(html4css1.HTMLTranslator):

    def starttag(self, node, tagname, suffix='\n', empty=0, **attributes):
        if False:
            while True:
                i = 10
        x = html4css1.HTMLTranslator.starttag(self, node, tagname, suffix, empty, **attributes)
        y = x.replace('id="tags"', 'id="tags_"')
        y = y.replace('name="tags"', 'name="tags_"')
        y = y.replace('href="#tags"', 'href="#tags_"')
        return y
mywriter = html4css1.Writer()
mywriter.translator_class = IESafeHtmlTranslator
publish_cmdline(writer=mywriter, description=description)