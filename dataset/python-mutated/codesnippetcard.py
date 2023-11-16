"""
Code snippet card, used in index page.
"""
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
from sphinx.addnodes import pending_xref
CARD_TEMPLATE_HEADER = '\n.. raw:: html\n\n    <div class="codesnippet-card admonition">\n\n    <div class="codesnippet-card-body">\n\n    <div class="codesnippet-card-title-container">\n\n    <div class="codesnippet-card-icon">\n\n.. image:: {icon}\n\n.. raw:: html\n\n    </div>\n\n    <h4>{title}</h4>\n    </div>\n\n'
CARD_TEMPLATE_FOOTER = '\n.. raw:: html\n\n    </div>\n'
CARD_TEMPLATE_LINK_CONTAINER_HEADER = '\n.. raw:: html\n\n    <div class="codesnippet-card-footer">\n'
CARD_TEMPLATE_LINK = '\n.. raw:: html\n\n    <div class="codesnippet-card-link">\n    {seemore}\n    <span class="material-icons right">arrow_forward</span>\n    </div>\n'

class CodeSnippetCardDirective(Directive):
    option_spec = {'icon': directives.unchanged, 'title': directives.unchanged, 'link': directives.unchanged, 'seemore': directives.unchanged}
    has_content = True

    def run(self):
        if False:
            while True:
                i = 10
        anchor_node = nodes.paragraph()
        try:
            title = self.options['title']
            link = directives.uri(self.options['link'])
            icon = directives.uri(self.options['icon'])
            seemore = self.options.get('seemore', 'For a full tutorial, please go here.')
        except ValueError as e:
            print(e)
            raise
        card_rst = CARD_TEMPLATE_HEADER.format(title=title, icon=icon)
        card_list = StringList(card_rst.split('\n'))
        self.state.nested_parse(card_list, self.content_offset, anchor_node)
        self.state.nested_parse(self.content, self.content_offset, anchor_node)
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK_CONTAINER_HEADER.split('\n')), self.content_offset, anchor_node)
        link_node = pending_xref(CARD_TEMPLATE_LINK, reftype='doc', refdomain='std', reftarget=link, refexplicit=False, refwarn=True, refkeepformat=True)
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK.format(seemore=seemore).split('\n')), self.content_offset, link_node)
        anchor_node += link_node
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)
        return [anchor_node]

def setup(app):
    if False:
        while True:
            i = 10
    app.add_directive('codesnippetcard', CodeSnippetCardDirective)