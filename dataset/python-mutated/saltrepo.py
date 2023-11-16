"""
    saltrepo
    ~~~~~~~~

    SaltStack Repository Sphinx directives
"""

def source_read_handler(app, docname, source):
    if False:
        for i in range(10):
            print('nop')
    if '|repo_primary_branch|' in source[0]:
        source[0] = source[0].replace('|repo_primary_branch|', app.config.html_context['repo_primary_branch'])

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('source-read', source_read_handler)
    return {'version': 'builtin', 'parallel_read_safe': True, 'parallel_write_safe': True}