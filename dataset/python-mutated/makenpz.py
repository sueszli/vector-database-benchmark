"""
python makenpz.py DIRECTORY

Build a npz containing all data files in the directory.

"""
import os
import numpy as np
import argparse
from stat import ST_MTIME

def newer(source, target):
    if False:
        return 10
    "\n    Return true if 'source' exists and is more recently modified than\n    'target', or if 'source' exists and 'target' doesn't.  Return false if\n    both exist and 'target' is the same age or younger than 'source'.\n    "
    if not os.path.exists(source):
        raise ValueError("file '%s' does not exist" % os.path.abspath(source))
    if not os.path.exists(target):
        return 1
    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]
    return mtime1 > mtime2

def main():
    if False:
        print('Hello World!')
    p = argparse.ArgumentParser(usage=(__doc__ or '').strip())
    p.add_argument('--use-timestamp', action='store_true', default=False, help="don't rewrite npz file if it is newer than sources")
    p.add_argument('dirname')
    p.add_argument('-o', '--outdir', type=str, help='Relative path to the output directory')
    args = p.parse_args()
    if not args.outdir:
        raise ValueError('Missing `--outdir` argument to makenpz.py')
    else:
        inp = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'tests', 'data', args.dirname)
        outdir_abs = os.path.join(os.getcwd(), args.outdir)
        outp = os.path.join(outdir_abs, args.dirname + '.npz')
    if os.path.isfile(outp) and (not os.path.isdir(inp)):
        return
    files = []
    for (dirpath, dirnames, filenames) in os.walk(inp):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            if fn.endswith('.txt'):
                key = dirpath[len(inp) + 1:] + '-' + fn[:-4]
                key = key.strip('-')
                files.append((key, os.path.join(dirpath, fn)))
    if args.use_timestamp and os.path.isfile(outp):
        try:
            old_data = np.load(outp)
            try:
                changed = set(old_data.keys()) != {key for (key, _) in files}
            finally:
                old_data.close()
        except OSError:
            changed = True
        changed = changed or any((newer(fn, outp) for (key, fn) in files))
        changed = changed or newer(__file__, outp)
        if not changed:
            return
    data = {}
    for (key, fn) in files:
        data[key] = np.loadtxt(fn)
    np.savez_compressed(outp, **data)
if __name__ == '__main__':
    main()