"""Inspired by https://github.com/pandas-dev/pydata-sphinx-theme

BSD 3-Clause License

Copyright (c) 2018, pandas
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import docutils

def add_toctree_functions(app, pagename, templatename, context, doctree):
    if False:
        i = 10
        return i + 15
    'Add functions so Jinja templates can add toctree objects.\n\n    This converts the docutils nodes into a nested dictionary that Jinja can\n    use in our templating.\n    '
    from sphinx.environment.adapters.toctree import TocTree

    def get_nav_object(maxdepth=None, collapse=True, numbered=False, **kwargs):
        if False:
            print('Hello World!')
        'Return a list of nav links that can be accessed from Jinja.\n\n        Parameters\n        ----------\n        maxdepth: int\n            How many layers of TocTree will be returned\n        collapse: bool\n            Whether to only include sub-pages of the currently-active page,\n            instead of sub-pages of all top-level pages of the site.\n        numbered: bool\n            Whether to add section number to title\n        kwargs: key/val pairs\n            Passed to the `TocTree.get_toctree_for` Sphinx method\n        '
        toctree = TocTree(app.env).get_toctree_for(pagename, app.builder, collapse=collapse, maxdepth=maxdepth, **kwargs)
        if toctree is None:
            return []
        toc_items = [item for child in toctree.children for item in child if isinstance(item, docutils.nodes.list_item)]
        nav = [docutils_node_to_jinja(child, only_pages=True, numbered=numbered) for child in toc_items]
        return nav
    context['get_nav_object'] = get_nav_object

def docutils_node_to_jinja(list_item, only_pages=False, numbered=False):
    if False:
        while True:
            i = 10
    'Convert a docutils node to a structure that can be read by Jinja.\n\n    Parameters\n    ----------\n    list_item : docutils list_item node\n        A parent item, potentially with children, corresponding to the level\n        of a TocTree.\n    only_pages : bool\n        Only include items for full pages in the output dictionary. Exclude\n        anchor links (TOC items with a URL that starts with #)\n    numbered: bool\n        Whether to add section number to title\n\n    Returns\n    -------\n    nav : dict\n        The TocTree, converted into a dictionary with key/values that work\n        within Jinja.\n    '
    if not list_item.children:
        return None
    reference = list_item.children[0].children[0]
    title = reference.astext()
    url = reference.attributes['refuri']
    active = 'current' in list_item.attributes['classes']
    secnumber = reference.attributes.get('secnumber', None)
    if numbered and secnumber is not None:
        secnumber = '.'.join((str(n) for n in secnumber))
        title = f'{secnumber}. {title}'
    if only_pages and '#' in url:
        return None
    nav = {}
    nav['title'] = title
    nav['url'] = url
    nav['active'] = active
    nav['children'] = []
    if len(list_item.children) > 1:
        subpage_list = list_item.children[1].children
        for sub_page in subpage_list:
            child_nav = docutils_node_to_jinja(sub_page, only_pages=only_pages, numbered=numbered)
            if child_nav is not None:
                nav['children'].append(child_nav)
    return nav

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('html-page-context', add_toctree_functions)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}