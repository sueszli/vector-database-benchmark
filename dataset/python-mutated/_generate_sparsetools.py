"""
python generate_sparsetools.py

Generate manual wrappers for C++ sparsetools code.

Type codes used:

    'i':  integer scalar
    'I':  integer array
    'T':  data array
    'B':  boolean array
    'V':  std::vector<integer>*
    'W':  std::vector<data>*
    '*':  indicates that the next argument is an output argument
    'v':  void
    'l':  64-bit integer scalar

See sparsetools.cxx for more details.

"""
import argparse
import os
from stat import ST_MTIME
BSR_ROUTINES = '\nbsr_diagonal        v iiiiiIIT*T\nbsr_tocsr           v iiiiIIT*I*I*T\nbsr_scale_rows      v iiiiII*TT\nbsr_scale_columns   v iiiiII*TT\nbsr_sort_indices    v iiii*I*I*T\nbsr_transpose       v iiiiIIT*I*I*T\nbsr_matmat          v iiiiiiIITIIT*I*I*T\nbsr_matvec          v iiiiIITT*T\nbsr_matvecs         v iiiiiIITT*T\nbsr_elmul_bsr       v iiiiIITIIT*I*I*T\nbsr_eldiv_bsr       v iiiiIITIIT*I*I*T\nbsr_plus_bsr        v iiiiIITIIT*I*I*T\nbsr_minus_bsr       v iiiiIITIIT*I*I*T\nbsr_maximum_bsr     v iiiiIITIIT*I*I*T\nbsr_minimum_bsr     v iiiiIITIIT*I*I*T\nbsr_ne_bsr          v iiiiIITIIT*I*I*B\nbsr_lt_bsr          v iiiiIITIIT*I*I*B\nbsr_gt_bsr          v iiiiIITIIT*I*I*B\nbsr_le_bsr          v iiiiIITIIT*I*I*B\nbsr_ge_bsr          v iiiiIITIIT*I*I*B\n'
CSC_ROUTINES = '\ncsc_diagonal        v iiiIIT*T\ncsc_tocsr           v iiIIT*I*I*T\ncsc_matmat_maxnnz   l iiIIII\ncsc_matmat          v iiIITIIT*I*I*T\ncsc_matvec          v iiIITT*T\ncsc_matvecs         v iiiIITT*T\ncsc_elmul_csc       v iiIITIIT*I*I*T\ncsc_eldiv_csc       v iiIITIIT*I*I*T\ncsc_plus_csc        v iiIITIIT*I*I*T\ncsc_minus_csc       v iiIITIIT*I*I*T\ncsc_maximum_csc     v iiIITIIT*I*I*T\ncsc_minimum_csc     v iiIITIIT*I*I*T\ncsc_ne_csc          v iiIITIIT*I*I*B\ncsc_lt_csc          v iiIITIIT*I*I*B\ncsc_gt_csc          v iiIITIIT*I*I*B\ncsc_le_csc          v iiIITIIT*I*I*B\ncsc_ge_csc          v iiIITIIT*I*I*B\n'
CSR_ROUTINES = '\ncsr_matmat_maxnnz   l iiIIII\ncsr_matmat          v iiIITIIT*I*I*T\ncsr_diagonal        v iiiIIT*T\ncsr_tocsc           v iiIIT*I*I*T\ncsr_tobsr           v iiiiIIT*I*I*T\ncsr_todense         v iiIIT*T\ncsr_matvec          v iiIITT*T\ncsr_matvecs         v iiiIITT*T\ncsr_elmul_csr       v iiIITIIT*I*I*T\ncsr_eldiv_csr       v iiIITIIT*I*I*T\ncsr_plus_csr        v iiIITIIT*I*I*T\ncsr_minus_csr       v iiIITIIT*I*I*T\ncsr_maximum_csr     v iiIITIIT*I*I*T\ncsr_minimum_csr     v iiIITIIT*I*I*T\ncsr_ne_csr          v iiIITIIT*I*I*B\ncsr_lt_csr          v iiIITIIT*I*I*B\ncsr_gt_csr          v iiIITIIT*I*I*B\ncsr_le_csr          v iiIITIIT*I*I*B\ncsr_ge_csr          v iiIITIIT*I*I*B\ncsr_scale_rows      v iiII*TT\ncsr_scale_columns   v iiII*TT\ncsr_sort_indices    v iI*I*T\ncsr_eliminate_zeros v ii*I*I*T\ncsr_sum_duplicates  v ii*I*I*T\nget_csr_submatrix   v iiIITiiii*V*V*W\ncsr_row_index       v iIIIT*I*T\ncsr_row_slice       v iiiIIT*I*T\ncsr_column_index1   v iIiiII*I*I\ncsr_column_index2   v IIiIT*I*T\ncsr_sample_values   v iiIITiII*T\ncsr_count_blocks    i iiiiII\ncsr_sample_offsets  i iiIIiII*I\ncsr_hstack          v iiIIIT*I*I*T\nexpandptr           v iI*I\ntest_throw_error    i\ncsr_has_sorted_indices    i iII\ncsr_has_canonical_format  i iII\n'
OTHER_ROUTINES = '\ncoo_tocsr           v iiiIIT*I*I*T\ncoo_todense         v iilIIT*Ti\ncoo_matvec          v lIITT*T\ndia_matvec          v iiiiITT*T\ncs_graph_components i iII*I\n'
COMPILATION_UNITS = [('bsr', BSR_ROUTINES), ('csr', CSR_ROUTINES), ('csc', CSC_ROUTINES), ('other', OTHER_ROUTINES)]
I_TYPES = [('NPY_INT32', 'npy_int32'), ('NPY_INT64', 'npy_int64')]
T_TYPES = [('NPY_BOOL', 'npy_bool_wrapper'), ('NPY_BYTE', 'npy_byte'), ('NPY_UBYTE', 'npy_ubyte'), ('NPY_SHORT', 'npy_short'), ('NPY_USHORT', 'npy_ushort'), ('NPY_INT', 'npy_int'), ('NPY_UINT', 'npy_uint'), ('NPY_LONG', 'npy_long'), ('NPY_ULONG', 'npy_ulong'), ('NPY_LONGLONG', 'npy_longlong'), ('NPY_ULONGLONG', 'npy_ulonglong'), ('NPY_FLOAT', 'npy_float'), ('NPY_DOUBLE', 'npy_double'), ('NPY_LONGDOUBLE', 'npy_longdouble'), ('NPY_CFLOAT', 'npy_cfloat_wrapper'), ('NPY_CDOUBLE', 'npy_cdouble_wrapper'), ('NPY_CLONGDOUBLE', 'npy_clongdouble_wrapper')]
THUNK_TEMPLATE = '\nstatic PY_LONG_LONG %(name)s_thunk(int I_typenum, int T_typenum, void **a)\n{\n    %(thunk_content)s\n}\n'
METHOD_TEMPLATE = '\nNPY_VISIBILITY_HIDDEN PyObject *\n%(name)s_method(PyObject *self, PyObject *args)\n{\n    return call_thunk(\'%(ret_spec)s\', "%(arg_spec)s", %(name)s_thunk, args);\n}\n'
GET_THUNK_CASE_TEMPLATE = '\nstatic int get_thunk_case(int I_typenum, int T_typenum)\n{\n    %(content)s;\n    return -1;\n}\n'

def newer(source, target):
    if False:
        i = 10
        return i + 15
    "\n    Return true if 'source' exists and is more recently modified than\n    'target', or if 'source' exists and 'target' doesn't.  Return false if\n    both exist and 'target' is the same age or younger than 'source'.\n    "
    if not os.path.exists(source):
        raise ValueError("file '%s' does not exist" % os.path.abspath(source))
    if not os.path.exists(target):
        return 1
    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]
    return mtime1 > mtime2

def get_thunk_type_set():
    if False:
        return 10
    '\n    Get a list containing cartesian product of data types, plus a getter routine.\n\n    Returns\n    -------\n    i_types : list [(j, I_typenum, None, I_type, None), ...]\n         Pairing of index type numbers and the corresponding C++ types,\n         and an unique index `j`. This is for routines that are parameterized\n         only by I but not by T.\n    it_types : list [(j, I_typenum, T_typenum, I_type, T_type), ...]\n         Same as `i_types`, but for routines parameterized both by T and I.\n    getter_code : str\n         C++ code for a function that takes I_typenum, T_typenum and returns\n         the unique index corresponding to the lists, or -1 if no match was\n         found.\n\n    '
    it_types = []
    i_types = []
    j = 0
    getter_code = '    if (0) {}'
    for (I_typenum, I_type) in I_TYPES:
        piece = '\n        else if (I_typenum == %(I_typenum)s) {\n            if (T_typenum == -1) { return %(j)s; }'
        getter_code += piece % dict(I_typenum=I_typenum, j=j)
        i_types.append((j, I_typenum, None, I_type, None))
        j += 1
        for (T_typenum, T_type) in T_TYPES:
            piece = '\n            else if (T_typenum == %(T_typenum)s) { return %(j)s; }'
            getter_code += piece % dict(T_typenum=T_typenum, j=j)
            it_types.append((j, I_typenum, T_typenum, I_type, T_type))
            j += 1
        getter_code += '\n        }'
    return (i_types, it_types, GET_THUNK_CASE_TEMPLATE % dict(content=getter_code))

def parse_routine(name, args, types):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate thunk and method code for a given routine.\n\n    Parameters\n    ----------\n    name : str\n        Name of the C++ routine\n    args : str\n        Argument list specification (in format explained above)\n    types : list\n        List of types to instantiate, as returned `get_thunk_type_set`\n\n    '
    ret_spec = args[0]
    arg_spec = args[1:]

    def get_arglist(I_type, T_type):
        if False:
            return 10
        '\n        Generate argument list for calling the C++ function\n        '
        args = []
        next_is_writeable = False
        j = 0
        for t in arg_spec:
            const = '' if next_is_writeable else 'const '
            next_is_writeable = False
            if t == '*':
                next_is_writeable = True
                continue
            elif t == 'i':
                args.append('*(%s*)a[%d]' % (const + I_type, j))
            elif t == 'I':
                args.append('(%s*)a[%d]' % (const + I_type, j))
            elif t == 'T':
                args.append('(%s*)a[%d]' % (const + T_type, j))
            elif t == 'B':
                args.append('(npy_bool_wrapper*)a[%d]' % (j,))
            elif t == 'V':
                if const:
                    raise ValueError("'V' argument must be an output arg")
                args.append('(std::vector<%s>*)a[%d]' % (I_type, j))
            elif t == 'W':
                if const:
                    raise ValueError("'W' argument must be an output arg")
                args.append('(std::vector<%s>*)a[%d]' % (T_type, j))
            elif t == 'l':
                args.append('*(%snpy_int64*)a[%d]' % (const, j))
            else:
                raise ValueError(f'Invalid spec character {t!r}')
            j += 1
        return ', '.join(args)
    thunk_content = 'int j = get_thunk_case(I_typenum, T_typenum);\n    switch (j) {'
    for (j, I_typenum, T_typenum, I_type, T_type) in types:
        arglist = get_arglist(I_type, T_type)
        piece = '\n        case %(j)s:'
        if ret_spec == 'v':
            piece += '\n            (void)%(name)s(%(arglist)s);\n            return 0;'
        else:
            piece += '\n            return %(name)s(%(arglist)s);'
        thunk_content += piece % dict(j=j, I_type=I_type, T_type=T_type, I_typenum=I_typenum, T_typenum=T_typenum, arglist=arglist, name=name)
    thunk_content += '\n    default:\n        throw std::runtime_error("internal error: invalid argument typenums");\n    }'
    thunk_code = THUNK_TEMPLATE % dict(name=name, thunk_content=thunk_content)
    method_code = METHOD_TEMPLATE % dict(name=name, ret_spec=ret_spec, arg_spec=arg_spec)
    return (thunk_code, method_code)

def main():
    if False:
        return 10
    p = argparse.ArgumentParser(usage=(__doc__ or '').strip())
    p.add_argument('--no-force', action='store_false', dest='force', default=True)
    p.add_argument('-o', '--outdir', type=str, help='Relative path to the output directory')
    options = p.parse_args()
    names = []
    (i_types, it_types, getter_code) = get_thunk_type_set()
    for (unit_name, routines) in COMPILATION_UNITS:
        thunks = []
        methods = []
        for line in routines.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                (name, args) = line.split(None, 1)
            except ValueError as e:
                raise ValueError(f'Malformed line: {line!r}') from e
            args = ''.join(args.split())
            if 't' in args or 'T' in args:
                (thunk, method) = parse_routine(name, args, it_types)
            else:
                (thunk, method) = parse_routine(name, args, i_types)
            if name in names:
                raise ValueError(f'Duplicate routine {name!r}')
            names.append(name)
            thunks.append(thunk)
            methods.append(method)
        if options.outdir:
            outdir = os.path.join(os.getcwd(), options.outdir)
        dst = os.path.join(outdir, unit_name + '_impl.h')
        if newer(__file__, dst) or options.force:
            if not options.outdir:
                print(f'[generate_sparsetools] generating {dst!r}')
            with open(dst, 'w') as f:
                write_autogen_blurb(f)
                f.write(getter_code)
                for thunk in thunks:
                    f.write(thunk)
                for method in methods:
                    f.write(method)
        elif not options.outdir:
            print(f'[generate_sparsetools] {dst!r} already up-to-date')
    method_defs = ''
    for name in names:
        method_defs += f'NPY_VISIBILITY_HIDDEN PyObject *{name}_method(PyObject *, PyObject *);\n'
    method_struct = '\nstatic struct PyMethodDef sparsetools_methods[] = {'
    for name in names:
        method_struct += '\n        {"%(name)s", (PyCFunction)%(name)s_method, METH_VARARGS, NULL},' % dict(name=name)
    method_struct += '\n        {NULL, NULL, 0, NULL}\n    };'
    dst = os.path.join(outdir, 'sparsetools_impl.h')
    if newer(__file__, dst) or options.force:
        if not options.outdir:
            print(f'[generate_sparsetools] generating {dst!r}')
        with open(dst, 'w') as f:
            write_autogen_blurb(f)
            f.write(method_defs)
            f.write(method_struct)
    elif not options.outdir:
        print(f'[generate_sparsetools] {dst!r} already up-to-date')

def write_autogen_blurb(stream):
    if False:
        i = 10
        return i + 15
    stream.write('/* This file is autogenerated by generate_sparsetools.py\n * Do not edit manually or check into VCS.\n */\n')
if __name__ == '__main__':
    main()