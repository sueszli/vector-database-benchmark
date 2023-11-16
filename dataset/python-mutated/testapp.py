"""A small application that can be used to test a WSGI server and check
it for WSGI compliance.
"""
from __future__ import annotations
import os
import sys
import typing as t
from textwrap import wrap
from markupsafe import escape
from . import __version__ as _werkzeug_version
from .wrappers.request import Request
from .wrappers.response import Response
TEMPLATE = '<!doctype html>\n<html lang=en>\n<title>WSGI Information</title>\n<style type="text/css">\n  @import url(https://fonts.googleapis.com/css?family=Ubuntu);\n\n  body       { font-family: \'Lucida Grande\', \'Lucida Sans Unicode\', \'Geneva\',\n               \'Verdana\', sans-serif; background-color: white; color: #000;\n               font-size: 15px; text-align: center; }\n  div.box    { text-align: left; width: 45em; margin: auto; padding: 50px 0;\n               background-color: white; }\n  h1, h2     { font-family: \'Ubuntu\', \'Lucida Grande\', \'Lucida Sans Unicode\',\n               \'Geneva\', \'Verdana\', sans-serif; font-weight: normal; }\n  h1         { margin: 0 0 30px 0; }\n  h2         { font-size: 1.4em; margin: 1em 0 0.5em 0; }\n  table      { width: 100%%; border-collapse: collapse; border: 1px solid #AFC5C9 }\n  table th   { background-color: #AFC1C4; color: white; font-size: 0.72em;\n               font-weight: normal; width: 18em; vertical-align: top;\n               padding: 0.5em 0 0.1em 0.5em; }\n  table td   { border: 1px solid #AFC5C9; padding: 0.1em 0 0.1em 0.5em; }\n  code       { font-family: \'Consolas\', \'Monaco\', \'Bitstream Vera Sans Mono\',\n               monospace; font-size: 0.7em; }\n  ul li      { line-height: 1.5em; }\n  ul.path    { font-size: 0.7em; margin: 0 -30px; padding: 8px 30px;\n               list-style: none; background: #E8EFF0; }\n  ul.path li { line-height: 1.6em; }\n  li.virtual { color: #999; text-decoration: underline; }\n  li.exp     { background: white; }\n</style>\n<div class="box">\n  <h1>WSGI Information</h1>\n  <p>\n    This page displays all available information about the WSGI server and\n    the underlying Python interpreter.\n  <h2 id="python-interpreter">Python Interpreter</h2>\n  <table>\n    <tr>\n      <th>Python Version\n      <td>%(python_version)s\n    <tr>\n      <th>Platform\n      <td>%(platform)s [%(os)s]\n    <tr>\n      <th>API Version\n      <td>%(api_version)s\n    <tr>\n      <th>Byteorder\n      <td>%(byteorder)s\n    <tr>\n      <th>Werkzeug Version\n      <td>%(werkzeug_version)s\n  </table>\n  <h2 id="wsgi-environment">WSGI Environment</h2>\n  <table>%(wsgi_env)s</table>\n  <h2 id="installed-eggs">Installed Eggs</h2>\n  <p>\n    The following python packages were installed on the system as\n    Python eggs:\n  <ul>%(python_eggs)s</ul>\n  <h2 id="sys-path">System Path</h2>\n  <p>\n    The following paths are the current contents of the load path.  The\n    following entries are looked up for Python packages.  Note that not\n    all items in this path are folders.  Gray and underlined items are\n    entries pointing to invalid resources or used by custom import hooks\n    such as the zip importer.\n  <p>\n    Items with a bright background were expanded for display from a relative\n    path.  If you encounter such paths in the output you might want to check\n    your setup as relative paths are usually problematic in multithreaded\n    environments.\n  <ul class="path">%(sys_path)s</ul>\n</div>\n'

def iter_sys_path() -> t.Iterator[tuple[str, bool, bool]]:
    if False:
        while True:
            i = 10
    if os.name == 'posix':

        def strip(x: str) -> str:
            if False:
                return 10
            prefix = os.path.expanduser('~')
            if x.startswith(prefix):
                x = f'~{x[len(prefix):]}'
            return x
    else:

        def strip(x: str) -> str:
            if False:
                print('Hello World!')
            return x
    cwd = os.path.abspath(os.getcwd())
    for item in sys.path:
        path = os.path.join(cwd, item or os.path.curdir)
        yield (strip(os.path.normpath(path)), not os.path.isdir(path), path != item)

@Request.application
def test_app(req: Request) -> Response:
    if False:
        print('Hello World!')
    "Simple test application that dumps the environment.  You can use\n    it to check if Werkzeug is working properly:\n\n    .. sourcecode:: pycon\n\n        >>> from werkzeug.serving import run_simple\n        >>> from werkzeug.testapp import test_app\n        >>> run_simple('localhost', 3000, test_app)\n         * Running on http://localhost:3000/\n\n    The application displays important information from the WSGI environment,\n    the Python interpreter and the installed libraries.\n    "
    try:
        import pkg_resources
    except ImportError:
        eggs: t.Iterable[t.Any] = ()
    else:
        eggs = sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower())
    python_eggs = []
    for egg in eggs:
        try:
            version = egg.version
        except (ValueError, AttributeError):
            version = 'unknown'
        python_eggs.append(f'<li>{escape(egg.project_name)} <small>[{escape(version)}]</small>')
    wsgi_env = []
    sorted_environ = sorted(req.environ.items(), key=lambda x: repr(x[0]).lower())
    for (key, value) in sorted_environ:
        value = ''.join(wrap(str(escape(repr(value)))))
        wsgi_env.append(f'<tr><th>{escape(key)}<td><code>{value}</code>')
    sys_path = []
    for (item, virtual, expanded) in iter_sys_path():
        class_ = []
        if virtual:
            class_.append('virtual')
        if expanded:
            class_.append('exp')
        class_ = f''' class="{' '.join(class_)}"''' if class_ else ''
        sys_path.append(f'<li{class_}>{escape(item)}')
    context = {'python_version': '<br>'.join(escape(sys.version).splitlines()), 'platform': escape(sys.platform), 'os': escape(os.name), 'api_version': sys.api_version, 'byteorder': sys.byteorder, 'werkzeug_version': _werkzeug_version, 'python_eggs': '\n'.join(python_eggs), 'wsgi_env': '\n'.join(wsgi_env), 'sys_path': '\n'.join(sys_path)}
    return Response(TEMPLATE % context, mimetype='text/html')
if __name__ == '__main__':
    from .serving import run_simple
    run_simple('localhost', 5000, test_app, use_reloader=True)