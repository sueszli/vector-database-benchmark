"""
Generate the regression database db.zip from the files in the <root>/test/models
directory. Older databases are overwritten with no prompt but can be restored
using Git as needed.

Use --help for usage.

On Windows, use ``py run.py <arguments>`` to make sure command line parameters
are forwarded to the script.
"""
import sys
import os
import subprocess
import zipfile
import settings
import utils
usage = "gen_db [assimp_binary] [-i=...] [-e=...] [-p] [-n]\n\nThe assimp_cmd (or assimp) binary to use is specified by the first\ncommand line argument and defaults to ``assimp``.\n\nTo build, set ``ASSIMP_BUILD_ASSIMP_TOOLS=ON`` in CMake. If generating\nconfigs for an IDE, make sure to build the assimp_cmd project.\n\n-i,--include: List of file extensions to update dumps for. If omitted,\n         all file extensions are updated except those in `exclude`.\n         Example: -ixyz,abc\n                  -i.xyz,.abc\n                  --include=xyz,abc\n\n-e,--exclude: Merged with settings.exclude_extensions to produce a\n         list of all file extensions to ignore. If dumps exist,\n         they are not altered. If not, theu are not created.\n\n-p,--preview: Preview list of file extensions touched by the update.\n         Dont' change anything.\n\n-n,--nozip: Don't pack to ZIP archive. Keep all dumps in individual files.\n"

def process_dir(d, outfile, file_filter):
    if False:
        print('Hello World!')
    " Generate small dump records for all files in 'd' "
    print('Processing directory ' + d)
    num = 0
    for f in os.listdir(d):
        fullp = os.path.join(d, f)
        if os.path.isdir(fullp) and (not f == '.svn'):
            num += process_dir(fullp, outfile, file_filter)
            continue
        if file_filter(f):
            for pp in settings.pp_configs_to_test:
                num += 1
                print('DUMP ' + fullp + '\n post-processing: ' + pp)
                outf = os.path.join(os.getcwd(), settings.database_name, utils.hashing(fullp, pp))
                cmd = [assimp_bin_path, 'dump', fullp, outf, '-b', '-s', '-l'] + pp.split()
                outfile.write('assimp dump ' + '-' * 80 + '\n')
                outfile.flush()
                if subprocess.call(cmd, stdout=outfile, stderr=outfile, shell=False):
                    print('Failure processing ' + fullp)
                    with open(outf, 'wb') as f:
                        pass
    return num

def make_zip():
    if False:
        while True:
            i = 10
    'Zip the contents of ./<settings.database_name>\n    to <settings.database_name>.zip using DEFLATE\n    compression to minimize the file size. '
    num = 0
    zipout = zipfile.ZipFile(settings.database_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    for f in os.listdir(settings.database_name):
        p = os.path.join(settings.database_name, f)
        zipout.write(p, f)
        if settings.remove_old:
            os.remove(p)
        num += 1
    if settings.remove_old:
        os.rmdir(settings.database_name)
    bad = zipout.testzip()
    assert bad is None
    print('=' * 60)
    print('Database contains {0} entries'.format(num))

def extract_zip():
    if False:
        print('Hello World!')
    'Unzip <settings.database_name>.zip to\n    ./<settings.database_name>'
    try:
        zipout = zipfile.ZipFile(settings.database_name + '.zip', 'r', 0)
        zipout.extractall(path=settings.database_name)
    except (RuntimeError, IOError) as r:
        print(r)
        print('failed to extract previous ZIP contents. DB is generated from scratch.')

def gen_db(ext_list, outfile):
    if False:
        print('Hello World!')
    'Generate the crash dump database in\n    ./<settings.database_name>'
    try:
        os.mkdir(settings.database_name)
    except OSError:
        pass
    num = 0
    for tp in settings.model_directories:
        num += process_dir(tp, outfile, lambda x: os.path.splitext(x)[1].lower() in ext_list and (not x in settings.files_to_ignore))
    print('=' * 60)
    print('Updated {0} entries'.format(num))
if __name__ == '__main__':

    def clean(f):
        if False:
            while True:
                i = 10
        f = f.strip("* '")
        return '.' + f if f[:1] != '.' else f
    if len(sys.argv) <= 1 or sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print(usage)
        sys.exit(0)
    assimp_bin_path = sys.argv[1]
    (ext_list, preview, nozip) = (None, False, False)
    for m in sys.argv[2:]:
        if m[:10] == '--exclude=':
            settings.exclude_extensions += map(clean, m[10:].split(','))
        elif m[:2] == '-e':
            settings.exclude_extensions += map(clean, m[2:].split(','))
        elif m[:10] == '--include=':
            ext_list = m[10:].split(',')
        elif m[:2] == '-i':
            ext_list = m[2:].split(',')
        elif m == '-p' or m == '--preview':
            preview = True
        elif m == '-n' or m == '--nozip':
            nozip = True
        else:
            print('Unrecognized parameter: ' + m)
            sys.exit(-1)
    outfile = open(os.path.join('..', 'results', 'gen_regression_db_output.txt'), 'w')
    if ext_list is None:
        (ext_list, err) = subprocess.Popen([assimp_bin_path, 'listext'], stdout=subprocess.PIPE).communicate()
        ext_list = str(ext_list.strip()).lower().split(';')
    ext_list = list(filter(lambda f: not f in settings.exclude_extensions, map(clean, ext_list)))
    print('File extensions processed: ' + ', '.join(ext_list))
    if preview:
        sys.exit(1)
    extract_zip()
    gen_db(ext_list, outfile)
    make_zip()
    print('=' * 60)
    input('Press any key to continue')
    sys.exit(0)