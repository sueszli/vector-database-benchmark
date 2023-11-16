from docutils.parsers.rst.roles import code_role

def python_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    if False:
        while True:
            i = 10
    options = {'language': 'python'}
    return code_role(role, rawtext, text, lineno, inliner, options=options, content=content)

def setup(app):
    if False:
        print('Hello World!')
    app.add_role('python', python_role)
    app.add_role('py', python_role)