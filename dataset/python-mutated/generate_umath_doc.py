import sys
import os
import textwrap
import argparse
sys.path.insert(0, os.path.dirname(__file__))
import ufunc_docstrings as docstrings
sys.path.pop(0)

def normalize_doc(docstring):
    if False:
        while True:
            i = 10
    docstring = textwrap.dedent(docstring).strip()
    docstring = docstring.encode('unicode-escape').decode('ascii')
    docstring = docstring.replace('"', '\\"')
    docstring = docstring.replace("'", "\\'")
    docstring = '\\n""'.join(docstring.split('\\n'))
    return docstring

def write_code(target):
    if False:
        i = 10
        return i + 15
    with open(target, 'w') as fid:
        fid.write('#ifndef NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n#define NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H_\n')
        for (place, string) in docstrings.docdict.items():
            cdef_name = f"DOC_{place.upper().replace('.', '_')}"
            cdef_str = normalize_doc(string)
            fid.write(f'#define {cdef_name} "{cdef_str}"\n')
        fid.write('#endif //NUMPY_CORE_INCLUDE__UMATH_DOC_GENERATED_H\n')

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', type=str, help='Path to the output directory')
    args = parser.parse_args()
    outfile = os.path.join(os.getcwd(), args.outfile)
    write_code(outfile)
if __name__ == '__main__':
    main()