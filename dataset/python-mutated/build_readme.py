def _read_readme():
    if False:
        print('Hello World!')
    with open('docs/readme.rst', 'r') as readme:
        return readme.read()

def _substitute(readme):
    if False:
        return 10
    readme = readme.replace('_static', 'docs/_static').replace('.. testcode::', '.. code-block:: python').replace('.. testoutput::\n   :hide:', '')
    return readme

def _write(readme):
    if False:
        while True:
            i = 10
    with open('readme.rst', 'w') as out:
        out.write(readme)
if __name__ == '__main__':
    readme = _read_readme()
    readme = _substitute(readme)
    _write(readme)