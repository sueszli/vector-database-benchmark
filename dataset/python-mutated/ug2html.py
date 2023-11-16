"""ug2html.py -- Creates HTML version of Robot Framework User Guide

Usage:  ug2html.py [ cr(eate) | dist | zip ]

create .. Creates the user guide so that it has relative links to images,
          library docs, etc. Mainly used to test how changes look in HTML.

dist .... Creates the user guide under 'robotframework-userguide-<version>'
          directory and also copies all needed images and other link targets
          there. The created output directory can thus be distributed
          independently.
          Note: Before running this command, you must generate documents at
          project root, for example, with: invoke library-docs all

zip ..... Uses 'dist' to create a stand-alone distribution and then packages
          it into 'robotframework-userguide-<version>.zip'

Version number to use is got automatically from 'src/robot/version.py' file.
"""
import os
import sys
import shutil
from translations import update_translations
'\n    The Pygments MoinMoin Parser\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    This fragment is a Docutils_ 0.4 directive that renders source code\n    (to HTML only, currently) via Pygments.\n\n    To use it, adjust the options below and copy the code into a module\n    that you import on initialization.  The code then automatically\n    registers a ``sourcecode`` directive that you can use instead of\n    normal code blocks like this::\n\n        .. sourcecode:: python\n\n            My code goes here.\n\n    If you want to have different code styles, e.g. one with line numbers\n    and one without, add formatters with their names in the VARIANTS dict\n    below.  You can invoke them instead of the DEFAULT one by using a\n    directive option::\n\n        .. sourcecode:: python\n            :linenos:\n\n            My code goes here.\n\n    Look at the `directive documentation`_ to get all the gory details.\n\n    .. _Docutils: http://docutils.sf.net/\n    .. _directive documentation:\n       http://docutils.sourceforge.net/docs/howto/rst-directives.html\n\n    :copyright: 2007 by Georg Brandl.\n    :license: BSD, see LICENSE for more details.\n'
INLINESTYLES = False
from pygments.formatters import HtmlFormatter
DEFAULT = HtmlFormatter(noclasses=INLINESTYLES)
VARIANTS = {}
from docutils import nodes
from docutils.parsers.rst import directives
from pygments import highlight, __version__ as pygments_version
from pygments.lexers import get_lexer_by_name

def too_old(version_string, minimum):
    if False:
        while True:
            i = 10
    version = tuple((int(v) for v in version_string.split('.')[:2]))
    return version < minimum
if too_old(pygments_version, (2, 8)):
    sys.exit('Pygments >= 2.8 is required.')

def pygments_directive(name, arguments, options, content, lineno, content_offset, block_text, state, state_machine):
    if False:
        print('Hello World!')
    try:
        lexer = get_lexer_by_name(arguments[0])
    except ValueError as err:
        raise ValueError(f'Invalid syntax highlighting language "{arguments[0]}".')
    formatter = options and VARIANTS[options.keys()[0]] or DEFAULT
    filtered = [line for line in content if line.strip()]
    if len(filtered) == 1:
        path = filtered[0].replace('/', os.sep)
        if os.path.isfile(path):
            content = open(path, encoding='utf-8').read().splitlines()
    parsed = highlight(u'\n'.join(content), lexer, formatter)
    return [nodes.raw('', parsed, format='html')]
pygments_directive.arguments = (1, 0, 1)
pygments_directive.content = 1
pygments_directive.options = dict([(key, directives.flag) for key in VARIANTS])
directives.register_directive('sourcecode', pygments_directive)
CURDIR = os.path.dirname(os.path.abspath(__file__))
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except ImportError:
    pass

def create_userguide():
    if False:
        i = 10
        return i + 15
    from docutils.core import publish_cmdline
    print('Creating user guide ...')
    print('Updating translations')
    update_translations()
    (version, version_file) = _update_version()
    install_file = _copy_installation_instructions()
    description = 'HTML generator for Robot Framework User Guide.'
    arguments = ['--stylesheet-path', ['src/userguide.css'], 'src/RobotFrameworkUserGuide.rst', 'RobotFrameworkUserGuide.html']
    os.chdir(CURDIR)
    publish_cmdline(writer_name='html', description=description, argv=arguments)
    os.unlink(version_file)
    os.unlink(install_file)
    ugpath = os.path.abspath(arguments[-1])
    print(ugpath)
    return (ugpath, version)

def _update_version():
    if False:
        return 10
    version = _get_version()
    print(f'Version: {version}')
    with open(os.path.join(CURDIR, 'src', 'version.rst'), 'w', encoding='utf-8') as vfile:
        vfile.write(f'.. |version| replace:: {version}\n')
    return (version, vfile.name)

def _get_version():
    if False:
        print('Hello World!')
    namespace = {}
    versionfile = os.path.join(CURDIR, '..', '..', 'src', 'robot', 'version.py')
    with open(versionfile, encoding='utf-8') as f:
        code = compile(f.read(), versionfile, 'exec')
        exec(code, namespace)
    return namespace['get_version']()

def _copy_installation_instructions():
    if False:
        while True:
            i = 10
    source = os.path.join(CURDIR, '..', '..', 'INSTALL.rst')
    target = os.path.join(CURDIR, 'src', 'GettingStarted', 'INSTALL.rst')
    include = True
    with open(source, encoding='utf-8') as source_file:
        with open(target, 'w', encoding='utf-8') as target_file:
            for line in source_file:
                if 'START USER GUIDE IGNORE' in line:
                    include = False
                if include:
                    target_file.write(line)
                if 'END USER GUIDE IGNORE' in line:
                    include = True
    return target

def create_distribution():
    if False:
        for i in range(10):
            print('nop')
    import re
    from urllib.parse import urlparse
    dist = os.path.normpath(os.path.join(CURDIR, '..', '..', 'dist'))
    (ugpath, version) = create_userguide()
    outdir = os.path.join(dist, f'robotframework-userguide-{version}')
    libraries = os.path.join(outdir, 'libraries')
    images = os.path.join(outdir, 'images')
    print('Creating distribution directory ...')
    if os.path.exists(outdir):
        print('Removing previous user guide distribution')
        shutil.rmtree(outdir)
    elif not os.path.exists(dist):
        os.mkdir(dist)
    for dirname in [outdir, libraries, images]:
        print(f'Creating output directory {dirname!r}')
        os.mkdir(dirname)

    def replace_links(res):
        if False:
            for i in range(10):
                print('nop')
        if not res.group(5):
            return res.group(0)
        (scheme, _, path, _, _, fragment) = urlparse(res.group(5))
        if scheme or (fragment and (not path)):
            return res.group(0)
        replaced_link = f'{res.group(1)} {res.group(4)}="%s/{os.path.basename(path)}"'
        if path.startswith('../libraries'):
            copy(path, libraries)
            replaced_link = replaced_link % 'libraries'
        elif path.startswith('src/'):
            copy(path, images)
            replaced_link = replaced_link % 'images'
        else:
            raise ValueError(f'Invalid link target: {path} (context: {res.group(0)})')
        print(f'Modified link {res.group(0)!r} -> {replaced_link!r}')
        return replaced_link

    def copy(source, dest):
        if False:
            print('Hello World!')
        print(f'Copying {source!r} -> {dest!r}')
        shutil.copy(source, dest)
    link_regexp = re.compile('\n(<(a|img)\\s+.*?)\n(\\s+(href|src)="(.*?)"|>)\n', re.VERBOSE | re.DOTALL | re.IGNORECASE)
    with open(ugpath, encoding='utf-8') as infile:
        content = link_regexp.sub(replace_links, infile.read())
    with open(os.path.join(outdir, os.path.basename(ugpath)), 'w', encoding='utf-8') as outfile:
        outfile.write(content)
    print(os.path.abspath(outfile.name))
    return outdir

def create_zip():
    if False:
        i = 10
        return i + 15
    ugdir = create_distribution()
    print('Creating zip package ...')
    zip_path = zip_distribution(ugdir)
    print(f'Removing distribution directory {ugdir!r}')
    shutil.rmtree(ugdir)
    print(zip_path)

def zip_distribution(dirpath):
    if False:
        while True:
            i = 10
    from zipfile import ZipFile, ZIP_DEFLATED
    zippath = os.path.normpath(dirpath) + '.zip'
    arcroot = os.path.dirname(dirpath)
    with ZipFile(zippath, 'w', compression=ZIP_DEFLATED) as zipfile:
        for (root, _, files) in os.walk(dirpath):
            for name in files:
                path = os.path.join(root, name)
                arcpath = os.path.relpath(path, arcroot)
                print(f'Adding {arcpath!r}')
                zipfile.write(path, arcpath)
    return os.path.abspath(zippath)
if __name__ == '__main__':
    actions = {'create': create_userguide, 'cr': create_userguide, 'dist': create_distribution, 'zip': create_zip}
    try:
        actions[sys.argv[1]](*sys.argv[2:])
    except (KeyError, IndexError, TypeError):
        print(__doc__)