from invoke import task

@task(help=dict(code='the Python code to transpile'))
def py2js(ctx, code):
    if False:
        print('Hello World!')
    'transpile given Python code to JavaScript\n    '
    from pscript import py2js
    print(py2js(code))