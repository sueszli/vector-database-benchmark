try:
    import os
    os.VfsPosix
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

def test(testdir):
    if False:
        return 10
    curdir = os.getcwd()
    vfs = os.VfsPosix(testdir)
    os.chdir(testdir)
    dirs = ['.a', '..a', '...a', 'a.b', 'a..b']
    for dir in dirs:
        vfs.mkdir(dir)
    dirs = []
    for entry in vfs.ilistdir('/'):
        dirs.append(entry[0])
    dirs.sort()
    print(dirs)
    os.chdir(curdir)
temp_dir = 'vfs_posix_ilistdir_filter_test_dir'
try:
    os.stat(temp_dir)
    print('SKIP')
    raise SystemExit
except OSError:
    pass
os.mkdir(temp_dir)
try:
    test(temp_dir)
finally:
    for td in os.listdir(temp_dir):
        os.rmdir('/'.join((temp_dir, td)))
    os.rmdir(temp_dir)