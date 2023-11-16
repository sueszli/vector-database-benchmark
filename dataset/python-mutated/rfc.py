"""
Custom Sphinx documentation module to link to parts of the OAuth2 RFC.
"""
from docutils import nodes
base_url = 'https://rfc-editor.org/rfc/rfc6749.html'

def rfclink(name, rawtext, text, lineno, inliner, options={}, content=[]):
    if False:
        return 10
    'Link to the OAuth2 draft.\n\n    Returns 2 part tuple containing list of nodes to insert into the\n    document and a list of system messages.  Both are allowed to be\n    empty.\n\n    :param name: The role name used in the document.\n    :param rawtext: The entire markup snippet, with role.\n    :param text: The text marked with the role.\n    :param lineno: The line number where rawtext appears in the input.\n    :param inliner: The inliner instance that called us.\n    :param options: Directive options for customization.\n    :param content: The directive content for customization.\n    '
    node = nodes.reference(rawtext, 'RFC6749 Section ' + text, refuri='%s#section-%s' % (base_url, text))
    return ([node], [])

def setup(app):
    if False:
        while True:
            i = 10
    '\n    Install the plugin.\n\n    :param app: Sphinx application context.\n    '
    app.add_role('rfc', rfclink)