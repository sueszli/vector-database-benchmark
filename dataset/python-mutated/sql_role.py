"""
    sql role
    ~~~~~~~~

    An interpreted text role to style SQL syntax in Psycopg documentation.

    :copyright: Copyright 2010 by Daniele Varrazzo.
"""
from docutils import nodes, utils
from docutils.parsers.rst import roles

def sql_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    if False:
        print('Hello World!')
    text = utils.unescape(text)
    options['classes'] = ['sql']
    return ([nodes.literal(rawtext, text, **options)], [])

def setup(app):
    if False:
        print('Hello World!')
    roles.register_local_role('sql', sql_role)