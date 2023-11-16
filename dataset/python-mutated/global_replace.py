import os
import stat
import sys

def update(file, pattern, replacement):
    if False:
        for i in range(10):
            print('nop')
    try:
        old_perm = os.stat(file)[0]
        if not os.access(file, os.W_OK):
            os.chmod(file, old_perm | stat.S_IWRITE)
        s = open(file, 'rb').read().decode('utf-8')
        t = s.replace(pattern, replacement)
        out = open(file, 'wb')
        out.write(t.encode('utf-8'))
        out.close()
        os.chmod(file, old_perm)
        return s != t
    except Exception:
        (exc_type, exc_obj, exc_tb) = sys.exc_info()
        print(f'Unable to check {file:s} {str(exc_type):s}')
        return 0
if __name__ == '__main__':
    if len(sys.argv) != 3:
        exit('Usage: %s <pattern> <replacement>' % sys.argv[0])
    pattern = sys.argv[1]
    replacement = sys.argv[2]
    count = 0
    for (root, dirs, files) in os.walk('.'):
        if not ('/.git' in root or '/.tox' in root):
            for file in files:
                path = os.path.join(root, file)
                if update(path, pattern, replacement):
                    print('Updated:', path)
                    count += 1
    print(f'Updated {count} files')