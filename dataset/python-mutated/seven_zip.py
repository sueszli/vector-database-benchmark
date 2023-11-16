import os
import re
from calibre.constants import iswindows

def open_archive(path_or_stream, mode='r'):
    if False:
        for i in range(10):
            print('nop')
    from py7zr import SevenZipFile
    return SevenZipFile(path_or_stream, mode=mode)

def names(path_or_stream):
    if False:
        while True:
            i = 10
    with open_archive(path_or_stream) as zf:
        return tuple(zf.getnames())

def extract_member(path_or_stream, match=None, name=None):
    if False:
        while True:
            i = 10
    if iswindows and name is not None:
        name = name.replace(os.sep, '/')
    if match is None:
        match = re.compile('\\.(jpg|jpeg|gif|png)\\s*$', re.I)

    def is_match(fname):
        if False:
            return 10
        if iswindows:
            fname = fname.replace(os.sep, '/')
        return name is not None and fname == name or (match is not None and match.search(fname) is not None)
    with open_archive(path_or_stream) as ar:
        all_names = list(filter(is_match, ar.getnames()))
        if all_names:
            return (all_names[0], ar.read(all_names[:1])[all_names[0]].read())

def extract_cover_image(stream):
    if False:
        i = 10
        return i + 15
    pos = stream.tell()
    from calibre.libunzip import name_ok, sort_key
    all_names = sorted(names(stream), key=sort_key)
    stream.seek(pos)
    for name in all_names:
        if name_ok(name):
            return extract_member(stream, name=name, match=None)

def extract(path_or_stream, location):
    if False:
        for i in range(10):
            print('nop')
    with open_archive(path_or_stream) as f:
        f.extract(location)

def test_basic():
    if False:
        while True:
            i = 10
    from tempfile import TemporaryDirectory
    from calibre import CurrentDir
    tdata = {'1/sub-one': b'sub-one\n', '2/sub-two.txt': b'sub-two\n', 'Füße.txt': b'unicode\n', 'max-compressed': b'max\n', 'one.txt': b'one\n', 'symlink': b'2/sub-two.txt', 'uncompressed': b'uncompressed\n', '诶比屁.txt': b'chinese unicode\n'}

    def do_test():
        if False:
            for i in range(10):
                print('nop')
        for (name, data) in tdata.items():
            if '/' in name:
                os.makedirs(os.path.dirname(name), exist_ok=True)
            with open(name, 'wb') as f:
                f.write(data)
        with open_archive(os.path.join('a.7z'), mode='w') as zf:
            for name in tdata:
                zf.write(name)
        with open_archive(os.path.join('a.7z')) as zf:
            if set(zf.getnames()) != set(tdata):
                raise ValueError('names not equal')
            read_data = {name: af.read() for (name, af) in zf.readall().items()}
            if read_data != tdata:
                raise ValueError('data not equal')
        for name in tdata:
            if name not in '1 2 symlink'.split():
                with open(os.path.join(tdir, name), 'rb') as s:
                    if s.read() != tdata[name]:
                        raise ValueError('Did not extract %s properly' % name)
    with TemporaryDirectory('test-7z') as tdir, CurrentDir(tdir):
        do_test()
if __name__ == '__main__':
    test_basic()