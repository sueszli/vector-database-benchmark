import os
import re
import sys
from os.path import join, split, normpath, exists
import subprocess
import argparse
from collections import defaultdict
parser = argparse.ArgumentParser(description="\nScript to fix invalid headers in the repository source files after file moves.\n\nMust be run from the root of the repository.\n\nExamples:\n\nTo fix one header file in all source files.\n\n./scripts/fix_headers.py <header_name>\n\nTo fix all references to a collection of header files (note regex match):\n\n./scripts/fix_headers.py --all-match='src/<subdir/.*\\.h.*'\n\n")
parser.add_argument('--header-root', dest='header_root', default='src', type=str, help='Base source of header info.')
parser.add_argument('--src-match', dest='src_match', default='src/* test/*', type=str, help='Match pattern of files to fix.')
parser.add_argument('headers', metavar='header', type=str, nargs='*', help='List of headers to fix.')
parser.add_argument('--all-match', dest='fix_all', help='Fix all headers below --header-root matching given regex pattern.', type=str)
args = parser.parse_args()
header_root = args.header_root.replace('//', '/')
while header_root.endswith('/'):
    header_root = header_root[:-1]
root_path = os.getcwd()

def error_out(*messages):
    if False:
        return 10
    print(''.join(messages), '\n')
    sys.exit(1)
if not exists(header_root + '/'):
    error_out(header_root + ' is not a valid directory.')
if not exists(join(root_path, '.git/config')):
    error_out('Script must be run in root of repository.')

def _run_git(cmd):
    if False:
        for i in range(10):
            print('nop')
    output = subprocess.check_output(cmd, shell=True)
    if type(output) is not str:
        output = output.decode('ascii')
    return [x.strip() for x in output.split('\n') if len(x.strip()) != 0]
raw_header_list = _run_git("git ls-files '%s/*.h' '%s/*.hpp'" % (header_root, header_root))

def all_repl_cmd(filter_name, *sed_commands):
    if False:
        print('Hello World!')
    proc = subprocess.check_call(cmd, shell=True)

def fix_headers_cmd(header_name, true_header):
    if False:
        while True:
            i = 10
    sed_commands = ['s|\\#include[ ]*\\<boost|\\#include_boost_|g', 's|\\#include[ ]*\\<[^\\>]*/%s\\>|#include <%s>|g' % (header_name, true_header), 's|\\#include[ ]*\\<%s\\>|#include <%s>|g' % (header_name, true_header), 's|\\#include[ ]*\\"[^\\"]*/%s\\"|#include <%s>|g' % (header_name, true_header), 's|\\#include[ ]*\\"[^\\"]*/%s\\"|#include <%s>|g' % (header_name, true_header), 's|cdef extern from \\"\\<[^\\>]*%s\\>\\"|cdef extern from "<%s>"|g' % (header_name, true_header), 's|\\#include_boost_|\\#include \\<boost|g']
    return "git grep -l '%s' %s | grep -v -E '^src/external/.+/.+/.*' | xargs sed -i '' -e %s " % (header_name, args.src_match, ' -e '.join(("'%s'" % s for s in sed_commands)))

def fix_headers(filenames):
    if False:
        i = 10
        return i + 15
    repls = []
    print('Locating true header paths.')
    for filename in filenames:
        header_files = [fn for fn in raw_header_list if fn.endswith('/' + filename) and 'boost/' not in fn]
        if len(header_files) == 0:
            error_out('File ', filename, ' not found in repository.')
        if len(header_files) > 1:
            error_out('Multiple matches for file ', filename, ' found. Please disambiguate by providing part of the path.\nFound: \n' + '\n'.join(header_files))
        new_file = header_files[0]
        assert new_file.startswith(('%s/' % header_root).replace('//', '/')), new_file
        new_file = new_file[len(header_root) + 1:]
        repls.append((filename, new_file))
    if len(repls) > 100:
        print('Fixing header locations for %d headers.' % len(repls))
    else:
        print('Fixing header locations for headers: \n' + '\n'.join(('  %s -> %s' % (h, fl) for (h, fl) in repls)))
    shell_cmd = '\n'.join(("{2} || echo 'ERROR fixing {0}; ignoring.' && echo 'Fixed {0} (True = {1}). ' \n".format(header, new_file, fix_headers_cmd(header, new_file)) for (header, new_file) in repls))
    open('run_all.out', 'w').write(shell_cmd)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
        temp.write(shell_cmd)
        temp.flush()
        subprocess.check_call(['bash', temp.name])

def print_usage_and_exit():
    if False:
        for i in range(10):
            print('nop')
    print('')
    print('Usage: %s <command> [args...]' % sys.argv[0])
    print('Commands: ')
    print('  --fix-headers <header_name>        Fixes all import paths of a unique header.')
    sys.exit(1)

def all_headers_matching(match_regex):
    if False:
        i = 10
        return i + 15
    rhl = [h for h in raw_header_list if match_regex.match(h)]

    def separate_unique(raw_headers, n):
        if False:
            for i in range(10):
                print('nop')
        lookup = defaultdict(lambda : [])
        for h in raw_headers:
            fn = '/'.join(h.split('/')[-n:])
            lookup[fn].append(h)
        header_list = []
        for (fn, hl) in lookup.items():
            if len(hl) == 1:
                header_list.append((fn, hl[0]))
            else:
                header_list += separate_unique(hl, n + 1)
        return header_list
    header_list = separate_unique(raw_header_list, 1)
    assert len(raw_header_list) == len(header_list)
    assert len(header_list) == len(set(header_list))
    ret = [h for (h, full_h) in header_list if match_regex.match(full_h)]
    print('Located %s headers matching given pattern.' % len(ret))
    return ret
if __name__ == '__main__':
    if args.fix_all:
        filter_regex = re.compile(args.fix_all)
        fix_headers(all_headers_matching(filter_regex))
    else:
        fix_headers(args.headers)