"""Add the generic fixed and derived content to a Classic Pygame document"""
from ext.utils import Visitor, get_name, GetError, get_refid, as_refid, as_refuri
from ext.indexer import get_descinfo, get_descinfo_refid
from sphinx.addnodes import desc, desc_content, desc_classname, desc_name, desc_signature
from sphinx.addnodes import index as section_prelude_end_class
from sphinx.domains.python import PyClasslike
from docutils.nodes import section, literal, reference, paragraph, title, document, Text, TextElement, inline, table, tgroup, colspec, tbody, row, entry, whitespace_normalize_name, SkipNode, line
import os
import re
from collections import deque

class PyGameClasslike(PyClasslike):
    """
    No signature prefix for classes.
    """

    def get_signature_prefix(self, sig):
        if False:
            i = 10
            return i + 15
        return '' if self.objtype == 'class' else PyClasslike(self, sig)

def setup(app):
    if False:
        return 10
    app.setup_extension('ext.indexer')
    app.add_config_value('boilerplate_skip_transform', [], '')
    app.add_directive_to_domain('py', 'class', PyGameClasslike)
    app.connect('doctree-resolved', transform_document)
    app.connect('html-page-context', inject_template_globals)
    app.add_node(TocRef, html=(visit_toc_ref_html, depart_toc_ref_html), latex=(visit_toc_ref, depart_toc_ref), text=(visit_toc_ref, depart_toc_ref))
    app.add_node(TocTable, html=(visit_toc_table_html, depart_toc_table_html), latex=(visit_skip, None), text=(visit_skip, None))
    app.add_node(DocTitle, html=(visit_doc_title_html, depart_doc_title_html), latex=(visit_doc_title, depart_doc_title))

class TocRef(reference):
    pass

def visit_toc_ref(self, node):
    if False:
        while True:
            i = 10
    self.visit_reference(node)

def depart_toc_ref(self, node):
    if False:
        while True:
            i = 10
    self.depart_reference(node)

def visit_toc_ref_html(self, node):
    if False:
        print('Hello World!')
    refuri = node['refuri']
    refid = as_refid(refuri)
    docname = get_descinfo_refid(refid, self.settings.env)['docname']
    link_suffix = self.builder.link_suffix
    node['refuri'] = f'{os.path.basename(docname)}{link_suffix}{refuri}'
    visit_toc_ref(self, node)

class TocTable(table):
    pass

def visit_toc_table_html(self, node):
    if False:
        i = 10
        return i + 15
    self.visit_table(node)

def depart_toc_table_html(self, node):
    if False:
        for i in range(10):
            print('nop')
    self.depart_table(node)

def visit_skip(self, node):
    if False:
        while True:
            i = 10
    raise SkipNode()
depart_toc_ref_html = depart_toc_ref

class DocTitle(title):
    pass
visit_doc_title_html = visit_skip
depart_doc_title_html = None

def visit_doc_title(self, node):
    if False:
        i = 10
        return i + 15
    self.visit_title(node)

def depart_doc_title(self, node):
    if False:
        while True:
            i = 10
    self.depart_title(node)

def transform_document(app, doctree, docname):
    if False:
        return 10
    if docname in app.config['boilerplate_skip_transform']:
        return
    doctree.walkabout(DocumentTransformer(app, doctree))

class DocumentTransformer(Visitor):
    _key_re = '(?P<key>[a-zA-Z_][a-zA-Z0-9_]*(\\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    key_pat = re.compile(_key_re)

    def __init__(self, app, document_node):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(app, document_node)
        self.module_stack = deque()
        self.title_stack = deque()

    def visit_section(self, node):
        if False:
            print('Hello World!')
        self.title_stack.append(None)

    def depart_section(self, node):
        if False:
            while True:
                i = 10
        title_node = self.title_stack.pop()
        if node['ids'][0].startswith('module-'):
            transform_module_section(node, title_node, self.env)

    def visit_desc(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node['domain'] != 'py':
            raise self.skip_node

    def depart_desc(self, node):
        if False:
            return 10
        node['classes'].append('definition')
        node[0]['classes'].append('title')
        if not node.attributes['noindex']:
            add_toc(node, self.env)

    def visit_title(self, node):
        if False:
            i = 10
            return i + 15
        if isinstance(node.parent.parent, document):
            node.__class__ = DocTitle

    def visit_reference(self, node):
        if False:
            print('Hello World!')
        if 'toc' in node['classes']:
            return
        try:
            child = node[0]
        except IndexError:
            return
        if not isinstance(child, TextElement):
            return
        name = child.astext()
        m = self.key_pat.match(name)
        if m is None:
            return
        key = m.group('key')
        try:
            summary = get_descinfo_refid(key, self.env)['summary']
        except GetError:
            return
        if summary:
            node['reftitle'] = ''
            node['classes'].append('tooltip')
            inline_node = inline('', summary, classes=['tooltip-content'])
            node.append(inline_node)

def transform_module_section(section_node, title_node, env):
    if False:
        i = 10
        return i + 15
    fullmodname = section_node['names'][0]
    where = section_node.first_child_matching_class(section_prelude_end_class)
    content_children = list(ipop_child(section_node, where + 1))
    if title_node is None:
        signature_children = [literal('', fullmodname)]
    else:
        signature_children = list(ipop_child(title_node))
    signature_node = desc_signature('', '', *signature_children, classes=['title', 'module'], names=[fullmodname])
    content_node = desc_content('', *content_children)
    desc_node = desc('', signature_node, content_node, desctype='module', objtype='module', classes=['definition'])
    section_node.append(desc_node)
    add_toc(desc_node, env, section_node)

def ipop_child(node, start=0):
    if False:
        for i in range(10):
            print('nop')
    while len(node) > start:
        yield node.pop(start)

def get_target_summary(reference_node, env):
    if False:
        print('Hello World!')
    try:
        return get_descinfo_refid(reference_node['refid'], env)['summary']
    except KeyError:
        raise GetError('reference has no refid')

def add_toc(desc_node, env, section_node=None):
    if False:
        return 10
    'Add a table of contents to a desc node'
    if section_node is not None:
        refid = get_refid(section_node)
    else:
        refid = get_refid(desc_node)
    descinfo = get_descinfo_refid(refid, env)
    toc = build_toc(descinfo, env)
    if toc is None:
        return
    content_node = desc_node[-1]
    insert_at = 0
    if descinfo['summary']:
        insert_at += 1
    content_node.insert(insert_at, toc)

def build_toc(descinfo, env):
    if False:
        for i in range(10):
            print('nop')
    'Return a desc table of contents node tree'
    separator = '—'
    child_ids = descinfo['children']
    if not child_ids:
        return None
    max_fullname_len = 0
    max_summary_len = 0
    rows = []
    for (fullname, refid, summary) in ichild_ids(child_ids, env):
        max_fullname_len = max(max_fullname_len, len(fullname))
        max_summary_len = max(max_summary_len, len(summary))
        reference_node = toc_ref(fullname, refid)
        ref_entry_node = entry('', line('', '', reference_node))
        sep_entry_node = entry('', Text(separator, ''))
        sum_entry_node = entry('', Text(summary, ''))
        row_node = row('', ref_entry_node, sep_entry_node, sum_entry_node)
        rows.append(row_node)
    col0_len = max_fullname_len + 2
    col1_len = len(separator)
    col2_len = max_summary_len + 10
    tbody_node = tbody('', *rows)
    col0_colspec_node = colspec(colwidth=col0_len)
    col1_colspec_node = colspec(colwidth=col1_len)
    col2_colspec_node = colspec(colwidth=col2_len)
    tgroup_node = tgroup('', col0_colspec_node, col1_colspec_node, col2_colspec_node, tbody_node, cols=3)
    return TocTable('', tgroup_node, classes=['toc'])

def ichild_ids(child_ids, env):
    if False:
        for i in range(10):
            print('nop')
    for refid in child_ids:
        descinfo = env.pyg_descinfo_tbl[refid]
        yield (descinfo['fullname'], descinfo['refid'], descinfo['summary'])

def toc_ref(fullname, refid):
    if False:
        i = 10
        return i + 15
    name = (whitespace_normalize_name(fullname),)
    return TocRef('', fullname, name=name, refuri=as_refuri(refid), classes=['toc'])

def decorate_signatures(desc, classname):
    if False:
        print('Hello World!')
    prefix = classname + '.'
    for child in desc.children:
        if isinstance(child, desc_signature) and isinstance(child[0], desc_name):
            new_desc_classname = desc_classname('', prefix)
            child.insert(0, new_desc_classname)

def inject_template_globals(app, pagename, templatename, context, doctree):
    if False:
        i = 10
        return i + 15

    def lowercase_name(d):
        if False:
            return 10
        return get_name(d['fullname']).lower()
    env = app.builder.env
    try:
        sections = env.pyg_sections
    except AttributeError:
        sections = []
    else:
        sections = sorted(sections, key=lowercase_name)
    existing_order = ['Color', 'cursors', 'display', 'draw', 'event', 'font', 'image', 'joystick', 'key', 'locals', 'mask', 'mixer', 'mouse', 'music', 'pygame', 'Rect', 'Surface', 'sprite', 'time', 'transform', 'BufferProxy', 'freetype', 'gfxdraw', 'midi', 'Overlay', 'PixelArray', 'pixelcopy', 'sndarray', 'surfarray']
    existing_order = ['pygame.' + x for x in existing_order]

    def sort_by_order(sequence, existing_order):
        if False:
            while True:
                i = 10
        return existing_order + [x for x in sequence if x not in existing_order]
    full_name_section = {x['fullname']: x for x in sections}
    full_names = [x['fullname'] for x in sections]
    sorted_names = sort_by_order(full_names, existing_order)
    sections = [full_name_section[name] for name in sorted_names if name in full_name_section]
    context['pyg_sections'] = sections