from pathlib import Path

def _parse_entry(entry):
    if False:
        i = 10
        return i + 15
    'Wrap in DocEntry object if it the entry was just a string'
    if isinstance(entry, str):
        return doc_entry(entry)
    else:
        return entry

class Doc:

    def __init__(self, title, underline_char, options, entries):
        if False:
            return 10
        self.title = title
        self.underline_char = underline_char
        if self.underline_char is not None and len(self.underline_char) != 1:
            raise ValueError(f'Expected only 1 character for `underline_char`, got {self.underline_char}.')
        if not isinstance(options, list):
            self.options = [options]
        else:
            self.options = options
        self.entries = entries
        self.entries = [_parse_entry(entry) for entry in entries]

    def get_title(self):
        if False:
            while True:
                i = 10
        if self.underline_char is None:
            return f'.. title:: {self.title}\n'
        else:
            return f'{self.title}\n{self.underline_char * len(self.title)}\n'

class DocEntry:

    def __init__(self, name, operator_refs):
        if False:
            i = 10
            return i + 15
        self.name = name
        if operator_refs is not None:
            if isinstance(operator_refs, list):
                for elem in operator_refs:
                    if not isinstance(elem, OpReference):
                        raise TypeError('Expected a single op_reference or a list of them to be provided')
                self.operator_refs = operator_refs
            elif not isinstance(operator_refs, OpReference):
                raise TypeError('Expected a single op_reference or a list of them to be provided')
            else:
                self.operator_refs = [operator_refs]
        else:
            self.operator_refs = None
        self.python_index = True if name.endswith('.py') else False

    def name_to_sphinx(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name.endswith('.py'):
            return str(Path(self.name).with_suffix('.rst'))
        return self.name

class OpReference:

    def __init__(self, operator, docstring, order=None):
        if False:
            print('Hello World!')
        self.operator = operator
        self.docstring = docstring
        self.order = 1000000 if order is None else order

def doc(title, underline_char=None, options=':maxdepth: 2', entries=[]):
    if False:
        print('Hello World!')
    'Main entry point for index.py file that replaces a standard index.rst file.\n\n    The contents of this doc will be used to generate corresponding index.rst\n\n    Parameters\n    ----------\n    title : str\n        Either a title used within `..title::` directive or if underline_char is present,\n        the underline_char will be used to do the sphinx header by placing\n        it len(title) times under the title.\n    underline_char : str, optional\n        If provided, do not generate a `..title::` section but a header with specified underline\n    options : str or list[str]\n        List of options like `:maxdepth:` for the toctree.\n    entries : list[str or doc_entry(...)]\n        Toctree of subpages, can be either represented by regular strings or by\n        `doc_entry()` that allows to put the reference from operator to given notebook.\n\n        Entries come in three form:\n          * a path to Python index file, for example: "operations/index.py" must lead to another\n            file with `doc()` section to be processed recursively.\n          * any other string representing path that doesn\'t end with `.py` - they will be inserted\n            as is. No extension also supported with the same behaviour as regular Sphinx.\n            Python processing stops here.\n          * an doc_entry() - allows to provide optional reference.\n\n    '
    global doc_return_value
    doc_return_value = Doc(title, underline_char, options, entries)

def doc_entry(name, operator_refs=None):
    if False:
        print('Hello World!')
    'Place given notebook or doc page in the toctree and optionally add a reference from operator\n    documentation to that notebook or page.\n\n    Parameters\n    ----------\n    name : str\n        Name of jupyter notebook or rst file, must contain proper extension.\n    operator_refs : OpReference or List[OpReference], optional\n        Optional reference, defined by `op_reference()` call, by default None\n    '
    return DocEntry(name, operator_refs)

def op_reference(operator, docstring, order=None):
    if False:
        for i in range(10):
            print('nop')
    'Add a reference from operator to this notebook with specified docstring.\n\n    Parameters\n    ----------\n    operator : str\n        Name of operator without nvidia.dali prefix, for example fn.resize or fn.gaussian_blur\n    docstring : str\n        Text that would appear in the see also block for given link.\n    order : int, optional\n        The order in which this entry should appear - lower values appear on top\n    '
    return OpReference(operator, docstring, order)

def _obtain_doc(py_file):
    if False:
        i = 10
        return i + 15
    'Extract the doc() definition from index.py file'
    with open(py_file, 'r') as f:
        doc_file = f.read()
        exec(doc_file)
        return doc_return_value

def _collect_references(base_path, entry_name, operator_refs, result_dict):
    if False:
        print('Hello World!')
    if operator_refs is None:
        return
    for op_ref in operator_refs:
        if not op_ref.operator in result_dict:
            result_dict[op_ref.operator] = []
        result_dict[op_ref.operator].append((op_ref.docstring, str((base_path / entry_name).with_suffix('.html')), op_ref))

def _document_examples(path, result_dict={}):
    if False:
        return 10
    if not path.endswith('.py'):
        raise ValueError(f"Expected a path to Python index file (ending with '.py'), got {path}")
    rst_file = Path(path).with_suffix('.rst')
    doc_contents = _obtain_doc(path)
    tab = ' ' * 3
    with open(rst_file, 'w') as f:
        f.write(doc_contents.get_title())
        f.write('\n')
        f.write(f'.. toctree::\n')
        for option in doc_contents.options:
            f.write(f'{tab}{option}\n')
        f.write('\n')
        for entry in doc_contents.entries:
            f.write(f'{tab}{entry.name_to_sphinx()}\n')
    canonical_path = Path(path)
    base_path = canonical_path.parent
    for entry in doc_contents.entries:
        _collect_references(base_path, entry.name_to_sphinx(), entry.operator_refs, result_dict)
        if entry.python_index:
            _document_examples(str(base_path / entry.name), result_dict)
    return result_dict

def document_examples(path):
    if False:
        return 10
    'Main api entry point, for given path to top-level index.py file containing doc() defintion\n    will generate a dictionary mapping operator/module to the list of referenced examples.\n\n    Parameters\n    ----------\n    path : str\n        Path to Python index file (with .py extension)\n\n    Returns\n    -------\n    Dict\n        Mapping from fn.operator or fn.module to list of example references\n    '
    dict = _document_examples(path)
    for key in dict:
        entries = sorted(dict[key], key=lambda entry: entry[2].order)
        dict[key] = [(str, url) for (str, url, _) in entries]
    return dict