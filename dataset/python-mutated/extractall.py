import os.path as osp
import tarfile
import zipfile

def extractall(path, to=None):
    if False:
        for i in range(10):
            print('nop')
    'Extract archive file.\n\n    Parameters\n    ----------\n    path: str\n        Path of archive file to be extracted.\n    to: str, optional\n        Directory to which the archive file will be extracted.\n        If None, it will be set to the parent directory of the archive file.\n    '
    if to is None:
        to = osp.dirname(path)
    if path.endswith('.zip'):
        (opener, mode) = (zipfile.ZipFile, 'r')
    elif path.endswith('.tar'):
        (opener, mode) = (tarfile.open, 'r')
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        (opener, mode) = (tarfile.open, 'r:gz')
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        (opener, mode) = (tarfile.open, 'r:bz2')
    else:
        raise ValueError("Could not extract '%s' as no appropriate extractor is found" % path)

    def namelist(f):
        if False:
            return 10
        if isinstance(f, zipfile.ZipFile):
            return f.namelist()
        return [m.path for m in f.members]

    def filelist(f):
        if False:
            i = 10
            return i + 15
        files = []
        for fname in namelist(f):
            fname = osp.join(to, fname)
            files.append(fname)
        return files
    with opener(path, mode) as f:
        f.extractall(path=to)
    return filelist(f)