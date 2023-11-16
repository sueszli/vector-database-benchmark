"""
Script to generate Kivy API from source code.

Code is messy, but working.
Be careful if you change anything in !

"""
ignore_list = ('kivy._clock', 'kivy._event', 'kivy.factory_registers', 'kivy.graphics.buffer', 'kivy.graphics.vbo', 'kivy.graphics.vertex', 'kivy.uix.recycleview.__init__', 'kivy.setupconfig', 'kivy.version', 'kivy._version')
import os
import sys
from glob import glob
import kivy
import kivy.app
import kivy.metrics
import kivy.atlas
import kivy.context
import kivy.core.audio
import kivy.core.camera
import kivy.core.clipboard
import kivy.core.gl
import kivy.core.image
import kivy.core.spelling
import kivy.core.text
import kivy.core.text.markup
import kivy.core.video
import kivy.core.window
import kivy.geometry
import kivy.graphics
import kivy.graphics.shader
import kivy.graphics.tesselator
import kivy.animation
import kivy.modules.console
import kivy.modules.keybinding
import kivy.modules.monitor
import kivy.modules.touchring
import kivy.modules.inspector
import kivy.modules.recorder
import kivy.modules.screen
import kivy.modules.joycursor
import kivy.storage
import kivy.storage.dictstore
import kivy.storage.jsonstore
import kivy.storage.redisstore
import kivy.network.urlrequest
import kivy.modules.webdebugger
import kivy.support
try:
    import kivy.tools.packaging.pyinstaller_hooks
except ImportError:
    pass
import kivy.input.recorder
import kivy.interactive
import kivy.garden
from kivy.factory import Factory
from kivy.lib import ddsfile, mtdev
BE_QUIET = True
if os.environ.get('BE_QUIET') == 'False':
    BE_QUIET = False
for x in list(Factory.classes.keys())[:]:
    getattr(Factory, x)
base_dir = os.path.dirname(__file__)
dest_dir = os.path.join(base_dir, 'sources')
examples_framework_dir = os.path.join(base_dir, '..', 'examples', 'framework')
base = 'autobuild.py-done'
with open(os.path.join(base_dir, base), 'w') as f:
    f.write('')

def writefile(filename, data):
    if False:
        print('Hello World!')
    global dest_dir
    f = os.path.join(dest_dir, filename)
    if not BE_QUIET:
        print('write', filename)
    if os.path.exists(f):
        with open(f) as fd:
            if fd.read() == data:
                return
    h = open(f, 'w')
    h.write(data)
    h.close()
'\nfor k in kivy.kivy_modules.list().keys():\n    kivy.kivy_modules.import_module(k)\n'
l = [(x, sys.modules[x], os.path.basename(sys.modules[x].__file__).rsplit('.', 1)[0]) for x in sys.modules if x.startswith('kivy') and sys.modules[x]]
packages = []
modules = {}
api_modules = []
for (name, module, filename) in l:
    if name in ignore_list:
        continue
    if not any([name.startswith(x) for x in ignore_list]):
        api_modules.append(name)
    if filename == '__init__':
        packages.append(name)
    elif hasattr(module, '__all__'):
        modules[name] = module.__all__
    else:
        modules[name] = [x for x in dir(module) if not x.startswith('__')]
packages.sort()
api_index = 'API Reference\n-------------\n\nThe API reference is a lexicographic list of all the different classes,\nmethods and features that Kivy offers.\n\n.. toctree::\n    :maxdepth: 1\n\n'
api_modules.sort()
for package in api_modules:
    api_index += '    api-%s.rst\n' % package
writefile('api-index.rst', api_index)
template = '\n'.join(('=' * 100, '$SUMMARY', '=' * 100, '\n$EXAMPLES_REF\n\n.. automodule:: $PACKAGE\n    :members:\n    :show-inheritance:\n\n.. toctree::\n\n$EXAMPLES\n'))
template_examples = '.. _example-reference%d:\n\nExamples\n--------\n\n%s\n'
template_examples_ref = '# :ref:`Jump directly to Examples <example-reference%d>`'

def extract_summary_line(doc):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param doc: the __doc__ field of a module\n    :return: a doc string suitable for a header or empty string\n    '
    if doc is None:
        return ''
    for line in doc.split('\n'):
        line = line.strip()
        if len(line) < 1:
            continue
        if line.startswith('.. _'):
            continue
        return line
for package in packages:
    summary = extract_summary_line(sys.modules[package].__doc__)
    if summary is None or summary == '':
        summary = 'NO DOCUMENTATION (package %s)' % package
    t = template.replace('$SUMMARY', summary)
    t = t.replace('$PACKAGE', package)
    t = t.replace('$EXAMPLES_REF', '')
    t = t.replace('$EXAMPLES', '')
    for subpackage in packages:
        packagemodule = subpackage.rsplit('.', 1)[0]
        if packagemodule != package or len(subpackage.split('.')) <= 2:
            continue
        t += '    api-%s.rst\n' % subpackage
    m = list(modules.keys())
    m.sort(key=lambda x: extract_summary_line(sys.modules[x].__doc__).upper())
    for module in m:
        packagemodule = module.rsplit('.', 1)[0]
        if packagemodule != package:
            continue
        t += '    api-%s.rst\n' % module
    writefile('api-%s.rst' % package, t)
m = list(modules.keys())
m.sort()
refid = 0
for module in m:
    summary = extract_summary_line(sys.modules[module].__doc__)
    if summary is None or summary == '':
        summary = 'NO DOCUMENTATION (module %s)' % module
    example_output = []
    example_prefix = module
    if module.startswith('kivy.'):
        example_prefix = module[5:]
    example_prefix = example_prefix.replace('.', '_')
    list_examples = glob('%s*.py' % os.path.join(examples_framework_dir, example_prefix))
    for x in list_examples:
        xb = os.path.basename(x)
        example_output.append('File :download:`%s <%s>` ::' % (xb, os.path.join('..', x)))
        with open(x, 'r') as fd:
            d = fd.read().strip()
            d = '\t' + '\n\t'.join(d.split('\n'))
            example_output.append(d)
    t = template.replace('$SUMMARY', summary)
    t = t.replace('$PACKAGE', module)
    if len(example_output):
        refid += 1
        example_output = template_examples % (refid, '\n\n\n'.join(example_output))
        t = t.replace('$EXAMPLES_REF', template_examples_ref % refid)
        t = t.replace('$EXAMPLES', example_output)
    else:
        t = t.replace('$EXAMPLES_REF', '')
        t = t.replace('$EXAMPLES', '')
    writefile('api-%s.rst' % module, t)
print('Auto-generation finished')