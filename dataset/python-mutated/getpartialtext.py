"""
Basically same as
`sphinx gettext buidler <https://www.sphinx-doc.org/en/master/_modules/sphinx/builders/gettext.html>`_,
but only get texts from files in a whitelist.
"""
import re
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.builders.gettext import MessageCatalogBuilder

class PartialMessageCatalogBuilder(MessageCatalogBuilder):
    name = 'getpartialtext'

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        super().init()
        self.whitelist_docs = [re.compile(x) for x in self.config.gettext_documents]

    def write_doc(self, docname: str, doctree: nodes.document) -> None:
        if False:
            i = 10
            return i + 15
        for doc_re in self.whitelist_docs:
            if doc_re.match(docname):
                return super().write_doc(docname, doctree)

def setup(app: Sphinx):
    if False:
        print('Hello World!')
    app.add_builder(PartialMessageCatalogBuilder)
    app.add_config_value('gettext_documents', [], 'gettext')