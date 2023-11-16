import os
from execnb.nbio import read_nb, write_nb
from io import StringIO
from contextlib import redirect_stdout

def exec_scr(src, dst, md):
    if False:
        for i in range(10):
            print('nop')
    f = StringIO()
    g = {}
    with redirect_stdout(f):
        exec(compile(src.read_text(), src, 'exec'), g)
    res = ''
    if md:
        res += '---\n' + md + '\n---\n\n'
    dst.write_text(res + f.getvalue())

def exec_nb(src, dst, cb):
    if False:
        i = 10
        return i + 15
    nb = read_nb(src)
    cb()(nb)
    write_nb(nb, dst)

def main(o):
    if False:
        return 10
    (src, dst, x) = o
    os.environ['IN_TEST'] = '1'
    if src.suffix == '.ipynb':
        exec_nb(src, dst, x)
    elif src.suffix == '.py':
        exec_scr(src, dst, x)
    else:
        raise Exception(src)
    del os.environ['IN_TEST']