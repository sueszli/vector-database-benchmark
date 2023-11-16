from docutils import nodes
from itertools import count
from sphinx.environment.adapters.toctree import TocTree
id_counter = count()
ID = 'sidebar-collapsible-toc'
CSS = '\nID li {\n    list-style: none;\n    margin-left: 0;\n    padding-left: 0.2em;\n    text-indent: -0.7em;\n}\n\nID li.leaf-node {\n    text-indent: 0;\n}\n\nID li input[type=checkbox] {\n    display: none;\n}\n\nID li > label {\n    cursor: pointer;\n}\n\nID li > input[type=checkbox] ~ ul > li {\n    display: none;\n}\n\nID li > input[type=checkbox]:checked ~ ul > li {\n    display: block;\n}\n\nID li > input[type=checkbox]:checked + label:before {\n    content: "\\025bf";\n}\n\nID li > input[type=checkbox]:not(:checked) + label:before {\n    content: "\\025b8";\n}\n'.replace('ID', 'ul#' + ID)

class checkbox(nodes.Element):
    pass

def visit_checkbox(self, node):
    if False:
        i = 10
        return i + 15
    cid = node['ids'][0]
    node['classes'] = []
    self.body.append('<input id="{0}" type="checkbox" /><label for="{0}">&nbsp;</label>'.format(cid))

def modify_li(li):
    if False:
        print('Hello World!')
    sublist = li.first_child_matching_class(nodes.bullet_list)
    if sublist is None or li[sublist].first_child_matching_class(nodes.list_item) is None:
        if not li.get('classes'):
            li['classes'] = []
        li['classes'].append('leaf-node')
    else:
        c = checkbox()
        c['ids'] = ['collapse-checkbox-{}'.format(next(id_counter))]
        li.insert(0, c)

def create_toc(app, pagename):
    if False:
        while True:
            i = 10
    tt = TocTree(app.env)
    toctree = tt.get_toc_for(pagename, app.builder)
    if toctree is not None:
        subtree = toctree[toctree.first_child_matching_class(nodes.list_item)]
        bl = subtree.first_child_matching_class(nodes.bullet_list)
        if bl is None:
            return
        subtree = subtree[bl]
        for li in subtree.traverse(nodes.list_item):
            modify_li(li)
        subtree['ids'] = [ID]
        return '<style>' + CSS + '</style>' + app.builder.render_partial(subtree)['fragment']

def add_html_context(app, pagename, templatename, context, *args):
    if False:
        for i in range(10):
            print('nop')
    if 'toc' in context:
        context['toc'] = create_toc(app, pagename) or context['toc']

def setup(app):
    if False:
        while True:
            i = 10
    app.add_node(checkbox, html=(visit_checkbox, lambda *x: None))
    app.connect('html-page-context', add_html_context)