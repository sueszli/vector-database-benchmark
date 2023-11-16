from sphinx import addnodes

def html_page_context(app, pagename, templatename, context, doctree):
    if False:
        print('Hello World!')
    "Event handler for the html-page-context signal.\n    Modifies the context directly.\n     - Replaces the 'toc' value created by the HTML builder with one\n       that shows all document titles and the local table of contents.\n     - Sets display_toc to True so the table of contents is always\n       displayed, even on empty pages.\n     - Replaces the 'toctree' function with one that uses the entire\n       document structure, ignores the maxdepth argument, and uses\n       only prune and collapse.\n    "
    rendered_toc = get_rendered_toctree(app.builder, pagename)
    context['toc'] = rendered_toc
    context['display_toc'] = True
    if 'toctree' not in context:
        return

    def make_toctree(collapse=True, maxdepth=-1, includehidden=True):
        if False:
            while True:
                i = 10
        return get_rendered_toctree(app.builder, pagename, prune=False, collapse=collapse)
    context['toctree'] = make_toctree

def get_rendered_toctree(builder, docname, prune=False, collapse=True):
    if False:
        print('Hello World!')
    'Build the toctree relative to the named document,\n    with the given parameters, and then return the rendered\n    HTML fragment.\n    '
    fulltoc = build_full_toctree(builder, docname, prune=prune, collapse=collapse)
    rendered_toc = builder.render_partial(fulltoc)['fragment']
    return rendered_toc

def build_full_toctree(builder, docname, prune, collapse):
    if False:
        print('Hello World!')
    'Return a single toctree starting from docname containing all\n    sub-document doctrees.\n    '
    env = builder.env
    doctree = env.get_doctree(env.config.master_doc)
    toctrees = []
    for toctreenode in doctree.traverse(addnodes.toctree):
        toctree = env.resolve_toctree(docname, builder, toctreenode, collapse=collapse, prune=prune, includehidden=True)
        if toctree is not None:
            toctrees.append(toctree)
    if not toctrees:
        return None
    result = toctrees[0]
    for toctree in toctrees[1:]:
        if toctree:
            result.extend(toctree.children)
    env.resolve_references(result, docname, builder)
    return result

def setup(app):
    if False:
        return 10
    app.connect('html-page-context', html_page_context)