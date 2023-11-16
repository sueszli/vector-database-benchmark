"""
Helper script for generating namespace packages for test-cases.
"""
import os
import shutil
declare_namespace_template = '\nimport pkg_resources\npkg_resources.declare_namespace(__name__)\n'
pkgutil_extend_path_template = '\nfrom pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n'
module_template = "\nprint ('this is module %s' % __name__)\n"
setup_template = "\nfrom setuptools import setup, find_packages\n\nsetup(\n    name='%(pkgname)s',\n    version='0.1',\n    description='A test package for name-spaces',\n    zip_safe=%(zip_safe)r,\n    packages=find_packages(),\n    namespace_packages = %(namespace_packages)r\n    )\n"
workdir = os.getcwd()
OLDPWD = os.getcwd()

def make_package(pkgname, namespace_packages, modules, zip_safe=False, declare_namespace_template=declare_namespace_template):
    if False:
        print('Hello World!')
    base = os.path.join(workdir, pkgname)
    if os.path.exists(base):
        shutil.rmtree(base)
    os.mkdir(base)
    os.chdir(base)
    for ns in namespace_packages:
        ns = os.path.join(*ns.split('.'))
        if not os.path.exists(ns):
            os.mkdir(ns)
        ns = os.path.join(ns, '__init__.py')
        with open(ns, 'w') as outfh:
            outfh.write(declare_namespace_template)
    for mod in modules:
        mod = os.path.join(*mod.split('/'))
        ns = os.path.dirname(mod)
        if not os.path.exists(ns):
            os.mkdir(ns)
        with open(mod, 'w') as outfh:
            outfh.write(module_template)
    with open('setup.py', 'w') as outfh:
        outfh.write(setup_template % locals())
    os.chdir(OLDPWD)
make_package('nspkg1-aaa', ['nspkg1'], ['nspkg1/aaa/__init__.py'])
make_package('nspkg1-bbb', ['nspkg1', 'nspkg1.bbb'], ['nspkg1/bbb/zzz/__init__.py'], zip_safe=True)
make_package('nspkg1-ccc', ['nspkg1'], ['nspkg1/ccc.py'])
make_package('nspkg1-empty', ['nspkg1'], [], zip_safe=True)
make_package('nspkg2-aaa', ['nspkg2'], ['nspkg2/aaa/__init__.py'])
make_package('nspkg2-bbb', ['nspkg2', 'nspkg2.bbb'], ['nspkg2/bbb/zzz/__init__.py'], zip_safe=True)
make_package('nspkg2-ccc', ['nspkg2'], ['nspkg2/ccc.py'])
make_package('nspkg2-empty', ['nspkg2'], [], zip_safe=True)
make_package('nspkg3-a', ['nspkg3', 'nspkg3.a'], ['nspkg3/a/__init__.py'], zip_safe=True, declare_namespace_template=pkgutil_extend_path_template)
make_package('nspkg3-aaa', ['nspkg3'], ['nspkg3/aaa/__init__.py'], declare_namespace_template=pkgutil_extend_path_template)
make_package('nspkg3-bbb', ['nspkg3', 'nspkg3.bbb'], ['nspkg3/bbb/zzz/__init__.py'], zip_safe=True, declare_namespace_template=pkgutil_extend_path_template)
make_package('nspkg3-ccc', ['nspkg3'], ['nspkg3/ccc.py'], declare_namespace_template=pkgutil_extend_path_template)
make_package('nspkg3-empty', ['nspkg3'], [], zip_safe=True, declare_namespace_template=pkgutil_extend_path_template)