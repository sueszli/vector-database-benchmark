import inspect
import os
import pkgutil
import shutil
import sys

def _obj_name(obj):
    if False:
        while True:
            i = 10
    if hasattr(obj, '__name__'):
        return obj.__name__

def make_markdown_url(line_string, s):
    if False:
        print('Hello World!')
    '\n    Turns an URL starting with s into\n    a markdown link\n    '
    new_line = []
    old_line = line_string.split(' ')
    for token in old_line:
        if not token.startswith(s):
            new_line.append(token)
        else:
            new_line.append('[%s](%s)' % (token, token))
    return ' '.join(new_line)

def docstring_to_markdown(docstring):
    if False:
        print('Hello World!')
    "Convert a Python object's docstring to markdown\n\n    Parameters\n    ----------\n    docstring : str\n        The docstring body.\n\n    Returns\n    ----------\n    clean_lst : list\n        The markdown formatted docstring as lines (str) in a Python list.\n\n    "
    new_docstring_lst = []
    encountered_examples = False
    for (idx, line) in enumerate(docstring.split('\n')):
        line = line.strip()
        if set(line) in ({'-'}, {'='}):
            new_docstring_lst[idx - 1] = '**%s**' % new_docstring_lst[idx - 1]
        elif line.startswith('>>>'):
            if not encountered_examples:
                new_docstring_lst.append('```')
                encountered_examples = True
        new_docstring_lst.append(line)
    for (idx, line) in enumerate(new_docstring_lst[1:]):
        if line:
            if line.startswith('Description : '):
                new_docstring_lst[idx + 1] = new_docstring_lst[idx + 1].replace('Description : ', '')
            elif ' : ' in line:
                line = line.replace(' : ', '` : ')
                new_docstring_lst[idx + 1] = '\n- `%s\n' % line
            elif '**' in new_docstring_lst[idx - 1] and '**' not in line:
                new_docstring_lst[idx + 1] = '\n%s' % line.lstrip()
            elif '**' not in line:
                new_docstring_lst[idx + 1] = '    %s' % line.lstrip()
    clean_lst = []
    for line in new_docstring_lst:
        if set(line.strip()) not in ({'-'}, {'='}):
            clean_lst.append(line)
    if encountered_examples:
        clean_lst.append('```')
    return clean_lst

def object_to_markdownpage(obj_name, obj, s=''):
    if False:
        for i in range(10):
            print('nop')
    "Generate the markdown documentation of a Python object.\n\n    Parameters\n    ----------\n    obj_name : str\n        Name of the Python object.\n    obj : object\n        Python object (class, method, function, ...)\n    s : str (default: '')\n        A string to which the documentation will be appended to.\n\n    Returns\n    ---------\n    s : str\n        The markdown page.\n\n    "
    s += '## %s\n' % obj_name
    sig = str(inspect.signature(obj)).replace('(self, ', '(')
    s += '\n*%s%s*\n\n' % (obj_name, sig)
    doc = str(inspect.getdoc(obj))
    ds = docstring_to_markdown(doc)
    s += '\n'.join(ds)
    if inspect.isclass(obj):
        (methods, properties) = ('\n\n### Methods', '\n\n### Properties')
        members = inspect.getmembers(obj)
        for m in members:
            if not m[0].startswith('_') and len(m) >= 2:
                if isinstance(m[1], property):
                    properties += '\n\n<hr>\n\n*%s*\n\n' % m[0]
                    m_doc = docstring_to_markdown(str(inspect.getdoc(m[1])))
                    properties += '\n'.join(m_doc)
                else:
                    sig = str(inspect.signature(m[1]))
                    sig = sig.replace('(self, ', '(').replace('(self)', '()')
                    sig = sig.replace('(self)', '()')
                    methods += '\n\n<hr>\n\n*%s%s*\n\n' % (m[0], sig)
                    m_doc = docstring_to_markdown(str(inspect.getdoc(m[1])))
                    methods += '\n'.join(m_doc)
        if len(methods) > len('\n\n### Methods'):
            s += methods
        if len(properties) > len('\n\n### Properties'):
            s += properties
    return s + '\n\n'

def import_package(rel_path_to_package, package_name):
    if False:
        while True:
            i = 10
    "Imports a Python package into the current namespace.\n\n    Parameters\n    ----------\n    rel_path_to_package : str\n        Path to the package containing director relative from this script's\n        directory.\n    package_name : str\n        The name of the package to be imported.\n\n    Returns\n    ---------\n    package : The imported package object.\n\n    "
    try:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        curr_dir = os.path.dirname(os.path.realpath(os.getcwd()))
    package_path = os.path.join(curr_dir, rel_path_to_package)
    if package_path not in sys.path:
        sys.path = [package_path] + sys.path
    package = __import__(package_name)
    return package

def get_subpackages(package):
    if False:
        for i in range(10):
            print('nop')
    'Return subpackages of a package.\n\n    Parameters\n    ----------\n    package : Python package object\n\n    Returns\n    --------\n    list : list containing (importer, subpackage_name) tuples\n\n    '
    return [i for i in pkgutil.iter_modules(package.__path__) if i[2]]

def get_modules(package):
    if False:
        print('Hello World!')
    'Return modules of a package.\n\n    Parameters\n    ----------\n    package : Python package object\n\n    Returns\n    --------\n    list : list containing (importer, subpackage_name) tuples\n\n    '
    return [i for i in pkgutil.iter_modules(package.__path__)]

def get_functions_and_classes(package):
    if False:
        i = 10
        return i + 15
    'Retun lists of functions and classes from a package.\n\n    Parameters\n    ----------\n    package : Python package object\n\n    Returns\n    --------\n    list, list : list of classes and functions\n        Each sublist consists of [name, member] sublists.\n\n    '
    (classes, functions) = ([], [])
    for (name, member) in inspect.getmembers(package):
        if not name.startswith('_'):
            if inspect.isclass(member):
                classes.append([name, member])
            elif inspect.isfunction(member):
                functions.append([name, member])
    return (classes, functions)

def generate_api_docs(package, api_dir, clean=False, printlog=True, ignore_packages=None):
    if False:
        i = 10
        return i + 15
    'Generate a module level API documentation of a python package.\n\n    Description\n    -----------\n    Generates markdown API files for each module in a Python package whereas\n    the structure is as follows:\n    `package/package.subpackage/package.subpackage.module.md`\n\n    Parameters\n    -----------\n    package : Python package object\n    api_dir : str\n        Output directory path for the top-level package directory\n    clean : bool (default: False)\n        Removes previously existing API directory if True.\n    printlog : bool (default: True)\n        Prints a progress log to the standard output screen if True.\n    ignore_packages : iterable or None (default: None)\n        Iterable (list, set, tuple) that contains the names of packages\n        and subpackages to ignore or skip. For instance, if the\n        images subpackage in mlxtend is supposed to be split, provide the\n        argument `{mlxtend.image}`.\n\n    '
    if printlog:
        print('\n\nGenerating Module Files\n%s\n' % (50 * '='))
    prefix = package.__name__ + '.'
    if clean:
        if os.path.isdir(api_dir):
            shutil.rmtree(api_dir)
    api_docs = {}
    for (importer, pkg_name, is_pkg) in pkgutil.iter_modules(package.__path__, prefix):
        if ignore_packages is not None and pkg_name in ignore_packages:
            if printlog:
                print('ignored %s' % pkg_name)
            continue
        if is_pkg:
            subpackage = __import__(pkg_name, fromlist='dummy')
            prefix = subpackage.__name__ + '.'
            (classes, functions) = get_functions_and_classes(subpackage)
            target_dir = os.path.join(api_dir, subpackage.__name__)
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
                if printlog:
                    print('created %s' % target_dir)
            for obj in classes + functions:
                md_path = os.path.join(target_dir, obj[0]) + '.md'
                if md_path not in api_docs:
                    api_docs[md_path] = object_to_markdownpage(obj_name=obj[0], obj=obj[1], s='')
                else:
                    api_docs[md_path] += object_to_markdownpage(obj_name=obj[0], obj=obj[1], s='')
    for d in sorted(api_docs):
        prev = ''
        if os.path.isfile(d):
            with open(d, 'r') as f:
                prev = f.read()
            if prev == api_docs[d]:
                msg = 'skipped'
            else:
                msg = 'updated'
        else:
            msg = 'created'
        if msg != 'skipped':
            with open(d, 'w') as f:
                f.write(api_docs[d])
        if printlog:
            print('%s %s' % (msg, d))

def summarize_methdods_and_functions(api_modules, out_dir, printlog=False, clean=True, str_above_header=''):
    if False:
        return 10
    "Generates subpacke-level summary files.\n\n    Description\n    -----------\n    A function to generate subpacke-level summary markdown API files from\n    a module-level API documentation previously created via the\n    `generate_api_docs` function.\n    The output structure is:\n        package/package.subpackage.md\n\n    Parameters\n    ----------\n    api_modules : str\n        Path to the API documentation crated via `generate_api_docs`\n    out_dir : str\n        Path to the desired output directory for the new markdown files.\n    clean : bool (default: False)\n        Removes previously existing API directory if True.\n    printlog : bool (default: True)\n        Prints a progress log to the standard output screen if True.\n    str_above_header : str (default: '')\n        Places a string just above the header.\n\n    "
    if printlog:
        print('\n\nGenerating Subpackage Files\n%s\n' % (50 * '='))
    if clean:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        if printlog:
            print('created %s' % out_dir)
    subdir_paths = [os.path.join(api_modules, d) for d in os.listdir(api_modules) if not d.startswith('.')]
    out_files = [os.path.join(out_dir, os.path.basename(d)) + '.md' for d in subdir_paths]
    for (sub_p, out_f) in zip(subdir_paths, out_files):
        module_paths = (os.path.join(sub_p, m) for m in os.listdir(sub_p) if not m.startswith('.'))
        new_output = []
        if str_above_header:
            new_output.append(str_above_header)
        for p in sorted(module_paths):
            with open(p, 'r') as r:
                new_output.extend(r.readlines())
                new_output.extend(['\n', '\n', '\n'])
        msg = ''
        if not os.path.isfile(out_f):
            msg = 'created'
        if msg != 'created':
            with open(out_f, 'r') as f:
                prev = f.readlines()
            if prev != new_output:
                msg = 'updated'
            else:
                msg = 'skipped'
        if msg != 'skipped':
            with open(out_f, 'w') as f:
                f.write(''.join(new_output))
        if printlog:
            print('%s %s' % (msg, out_f))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert docstring into a markdown API documentation.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-n', '--package_name', default='mlxtend', help='Name of the package')
    parser.add_argument('-d', '--package_dir', default='../../mlxtend/', help="Path to the package's enclosing directory")
    parser.add_argument('-o1', '--output_module_api', default='../docs/sources/api_modules', help='Target directory for the module-level API Markdown files')
    parser.add_argument('-o2', '--output_subpackage_api', default='../docs/sources/api_subpackages', help='Target directory for the subpackage-level API Markdown files')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove previous API files')
    parser.add_argument('-s', '--silent', action='store_true', help='Suppress log printed to the screen')
    parser.add_argument('-v', '--version', action='version', version='v. 0.1')
    parser.add_argument('--ignore_packages', default='', help='Ignores subpackages listed via this option. For example, to ignore mlxtend.image, type "mlxtend.image". For multiple subpackages, separate them via, commas. For example, "mlxtend.image,mlxtend.plotting".')
    args = parser.parse_args()
    ignore_packages_set = set(args.ignore_packages.split(','))
    package = import_package(args.package_dir, args.package_name)
    generate_api_docs(package=package, api_dir=args.output_module_api, clean=args.clean, ignore_packages=ignore_packages_set, printlog=not args.silent)
    summarize_methdods_and_functions(api_modules=args.output_module_api, out_dir=args.output_subpackage_api, printlog=not args.silent, clean=args.clean, str_above_header='mlxtend version: %s \n' % package.__version__)