"""
Look for escape sequences deprecated in Python 3.6.

Python 3.6 deprecates a number of non-escape sequences starting with '\\' that
were accepted before. For instance, '\\(' was previously accepted but must now
be written as '\\\\(' or r'\\('.

"""

def main(root):
    if False:
        return 10
    'Find deprecated escape sequences.\n\n    Checks for deprecated escape sequences in ``*.py files``. If `root` is a\n    file, that file is checked, if `root` is a directory all ``*.py`` files\n    found in a recursive descent are checked.\n\n    If a deprecated escape sequence is found, the file and line where found is\n    printed. Note that for multiline strings the line where the string ends is\n    printed and the error(s) are somewhere in the body of the string.\n\n    Parameters\n    ----------\n    root : str\n        File or directory to check.\n    Returns\n    -------\n    None\n\n    '
    import ast
    import tokenize
    import warnings
    from pathlib import Path
    count = 0
    base = Path(root)
    paths = base.rglob('*.py') if base.is_dir() else [base]
    for path in paths:
        with tokenize.open(str(path)) as f:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                tree = ast.parse(f.read())
            if w:
                print('file: ', str(path))
                for e in w:
                    print('line: ', e.lineno, ': ', e.message)
                print()
                count += len(w)
    print('Errors Found', count)
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Find deprecated escaped characters')
    parser.add_argument('root', help='directory or file to be checked')
    args = parser.parse_args()
    main(args.root)