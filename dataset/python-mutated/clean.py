import nbformat
from nbdev.export import *
from nbdev.clean import *
from fastcore.all import *
_re_header = re.compile('^#+\\s+\\S+')
_re_clean = re.compile('^\\s*#\\s*clean\\s*')

def is_header_cell(cell):
    if False:
        while True:
            i = 10
    return _re_header.search(cell['source']) is not None

def is_clean_cell(cell):
    if False:
        while True:
            i = 10
    return _re_clean.search(cell['source']) is not None
_re_questionnaire = re.compile('^#+\\s+Questionnaire')

def get_stop_idx(cells):
    if False:
        for i in range(10):
            print('nop')
    i = 0
    while i < len(cells) and _re_questionnaire.search(cells[i]['source']) is None:
        i += 1
    return i

def clean_tags(cell):
    if False:
        while True:
            i = 10
    if is_header_cell(cell):
        return cell
    for attr in ['id', 'caption', 'alt', 'width', 'hide_input', 'hide_output', 'clean']:
        cell['source'] = re.sub('#\\s*' + attr + '.*?($|\\n)', '', cell['source'])
    return cell

def proc_nb(fname, dest):
    if False:
        while True:
            i = 10
    nb = read_nb(fname)
    i = get_stop_idx(nb['cells'])
    nb['cells'] = [clean_tags(c) for (j, c) in enumerate(nb['cells']) if c['cell_type'] == 'code' or is_header_cell(c) or is_clean_cell(c) or (j >= i)]
    clean_nb(nb, clear_all=True)
    with open(dest / fname.name, 'w') as f:
        nbformat.write(nb, f, version=4)

def proc_all(path='.', dest_path='clean'):
    if False:
        for i in range(10):
            print('nop')
    (path, dest_path) = (Path(path), Path(dest_path))
    fns = [f for f in path.iterdir() if f.suffix == '.ipynb' and (not f.name.startswith('_'))]
    for fn in fns:
        proc_nb(fn, dest=dest_path)
if __name__ == '__main__':
    proc_all()