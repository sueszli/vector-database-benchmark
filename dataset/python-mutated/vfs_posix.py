try:
    import gc
    import os
    os.VfsPosix
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit
temp_dir = 'micropy_test_dir'
try:
    import os
    os.stat(temp_dir)
    print('SKIP')
    raise SystemExit
except OSError:
    pass
curdir = os.getcwd()
os.chdir('/')
print(os.getcwd())
os.chdir(curdir)
print(os.getcwd() == curdir)
print(type(os.stat('/')))
print(type(os.listdir('/')))
os.mkdir(temp_dir)
f = open(temp_dir + '/test', 'w')
f.write('hello')
f.close()
f.close()
f = open(2)
print(f)
f = open(temp_dir + '/test', 'r')
print(f.read())
f.close()
names = [temp_dir + '/x%d' % i for i in range(4)]
basefd = temp_dir + '/nextfd1'
nextfd = temp_dir + '/nextfd2'
with open(basefd, 'w') as f:
    base_file_no = f.fileno()
for i in range(1024):
    []

def write_files_without_closing():
    if False:
        for i in range(10):
            print('nop')
    for n in names:
        open(n, 'w').write(n)
    sorted(list(range(128)), key=lambda x: x)
write_files_without_closing()
gc.collect()
with open(nextfd, 'w') as f:
    next_file_no = f.fileno()
    print('next_file_no <= base_file_no', next_file_no <= base_file_no)
for n in names + [basefd, nextfd]:
    os.remove(n)
os.rename(temp_dir + '/test', temp_dir + '/test2')
print(os.listdir(temp_dir))
vfs = os.VfsPosix(temp_dir)
os.chdir(temp_dir)
print(list((i[0] for i in vfs.ilistdir('.'))))
print(type(vfs.stat('.')))
if hasattr(vfs, 'statvfs'):
    assert type(vfs.statvfs('.')) is tuple
print(type(list(vfs.ilistdir('.'))[0][0]))
print(type(list(vfs.ilistdir(b'.'))[0][0]))
vfs.mkdir('/subdir')
vfs.mkdir('/subdir/micropy_test_dir')
with vfs.open('/subdir/micropy_test_dir/test2', 'w') as f:
    f.write('wrong')
vfs.chdir('/subdir')
with vfs.open('/test2', 'r') as f:
    print(f.read())
os.chdir(curdir)
vfs.remove('/subdir/micropy_test_dir/test2')
vfs.rmdir('/subdir/micropy_test_dir')
vfs.rmdir('/subdir')
os.chdir(curdir)
os.remove(temp_dir + '/test2')
print(os.listdir(temp_dir))
try:
    import os
    os.remove(temp_dir + '/test2')
except OSError:
    print('remove OSError')
os.rmdir(temp_dir)
print(temp_dir in os.listdir())
try:
    import os
    os.rmdir(temp_dir)
except OSError:
    print('rmdir OSError')