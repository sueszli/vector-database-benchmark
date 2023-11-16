from argparse import ArgumentParser
import re

class PybindHeaderParser:

    def __init__(self, pathname):
        if False:
            return 10
        with open(pathname, 'r') as f:
            self.file_txt = f.read()

    def get_flag_automatic(self):
        if False:
            i = 10
            return i + 15
        m = re.search('BINDTOOL_GEN_AUTOMATIC\\(([^\\s])\\)', self.file_txt)
        if m and m.group(1) == '1':
            return True
        else:
            return False

    def get_flag_pygccxml(self):
        if False:
            i = 10
            return i + 15
        m = re.search('BINDTOOL_USE_PYGCCXML\\(([^\\s])\\)', self.file_txt)
        if m and m.group(1) == '1':
            return True
        else:
            return False

    def get_header_filename(self):
        if False:
            for i in range(10):
                print('nop')
        m = re.search('BINDTOOL_HEADER_FILE\\(([^\\s]*)\\)', self.file_txt)
        if m:
            return m.group(1)
        else:
            return None

    def get_header_file_hash(self):
        if False:
            return 10
        m = re.search('BINDTOOL_HEADER_FILE_HASH\\(([^\\s]*)\\)', self.file_txt)
        if m:
            return m.group(1)
        else:
            return None

    def get_flags(self):
        if False:
            return 10
        return f'{self.get_flag_automatic()};{self.get_flag_pygccxml()};{self.get_header_filename()};{self.get_header_file_hash()};'

def argParse():
    if False:
        i = 10
        return i + 15
    'Parses commandline args.'
    desc = 'Reads the parameters from the comment block in the pybind files'
    parser = ArgumentParser(description=desc)
    parser.add_argument('function', help='Operation to perform on comment block of pybind file', choices=['flag_auto', 'flag_pygccxml', 'header_filename', 'header_file_hash', 'all'])
    parser.add_argument('pathname', help='Pathname of pybind c++ file to read, e.g. blockname_python.cc')
    return parser.parse_args()
if __name__ == '__main__':
    args = argParse()
    pbhp = PybindHeaderParser(args.pathname)
    if args.function == 'flag_auto':
        print(pbhp.get_flag_automatic())
    elif args.function == 'flag_pygccxml':
        print(pbhp.get_flag_pygccxml())
    elif args.function == 'header_filename':
        print(pbhp.get_header_filename())
    elif args.function == 'header_file_hash':
        print(pbhp.get_header_file_hash())
    elif args.function == 'all':
        print(pbhp.get_flags())