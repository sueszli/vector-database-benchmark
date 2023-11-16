from . import render
from . import rml2pdf
from . import rml2html as htmlizer
from . import rml2txt as txtizer
from . import odt2odt as odt
from . import html2html as html
from . import makohtml2html as makohtml

class rml(render.render):

    def __init__(self, rml, localcontext=None, datas=None, path='.', title=None):
        if False:
            for i in range(10):
                print('nop')
        render.render.__init__(self, datas, path)
        self.localcontext = localcontext
        self.rml = rml
        self.output_type = 'pdf'
        self.title = title

    def _render(self):
        if False:
            while True:
                i = 10
        return rml2pdf.parseNode(self.rml, self.localcontext, images=self.bin_datas, path=self.path, title=self.title)

class rml2html(render.render):

    def __init__(self, rml, localcontext=None, datas=None):
        if False:
            while True:
                i = 10
        super(rml2html, self).__init__(datas)
        self.rml = rml
        self.localcontext = localcontext
        self.output_type = 'html'

    def _render(self):
        if False:
            print('Hello World!')
        return htmlizer.parseString(self.rml, self.localcontext)

class rml2txt(render.render):

    def __init__(self, rml, localcontext=None, datas=None):
        if False:
            i = 10
            return i + 15
        super(rml2txt, self).__init__(datas)
        self.rml = rml
        self.localcontext = localcontext
        self.output_type = 'txt'

    def _render(self):
        if False:
            while True:
                i = 10
        return txtizer.parseString(self.rml, self.localcontext)

class odt2odt(render.render):

    def __init__(self, rml, localcontext=None, datas=None):
        if False:
            i = 10
            return i + 15
        render.render.__init__(self, datas)
        self.rml_dom = rml
        self.localcontext = localcontext
        self.output_type = 'odt'

    def _render(self):
        if False:
            while True:
                i = 10
        return odt.parseNode(self.rml_dom, self.localcontext)

class html2html(render.render):

    def __init__(self, rml, localcontext=None, datas=None):
        if False:
            i = 10
            return i + 15
        render.render.__init__(self, datas)
        self.rml_dom = rml
        self.localcontext = localcontext
        self.output_type = 'html'

    def _render(self):
        if False:
            return 10
        return html.parseString(self.rml_dom, self.localcontext)

class makohtml2html(render.render):

    def __init__(self, html, localcontext=None):
        if False:
            for i in range(10):
                print('nop')
        render.render.__init__(self)
        self.html = html
        self.localcontext = localcontext
        self.output_type = 'html'

    def _render(self):
        if False:
            for i in range(10):
                print('nop')
        return makohtml.parseNode(self.html, self.localcontext)