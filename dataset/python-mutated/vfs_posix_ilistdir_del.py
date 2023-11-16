import gc
try:
    import os
    os.VfsPosix
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

def test(testdir):
    if False:
        print('Hello World!')
    curdir = os.getcwd()
    vfs = os.VfsPosix(testdir)
    os.chdir(testdir)
    vfs.mkdir('/test_d1')
    vfs.mkdir('/test_d2')
    vfs.mkdir('/test_d3')
    for i in range(10):
        print(i)
        idir = vfs.ilistdir('/')
        print(any(idir))
        for (dname, *_) in vfs.ilistdir('/'):
            vfs.rmdir(dname)
            break
        vfs.mkdir(dname)
        idir_emptied = vfs.ilistdir('/')
        l = list(idir_emptied)
        print(len(l))
        try:
            next(idir_emptied)
        except StopIteration:
            pass
        gc.collect()
        vfs.open('/test', 'w').close()
        vfs.remove('/test')
    os.chdir(curdir)
temp_dir = 'vfs_posix_ilistdir_del_test_dir'
try:
    os.stat(temp_dir)
    print('SKIP')
    raise SystemExit
except OSError:
    pass
os.mkdir(temp_dir)
test(temp_dir)
for td in os.listdir(temp_dir):
    os.rmdir('/'.join((temp_dir, td)))
os.rmdir(temp_dir)