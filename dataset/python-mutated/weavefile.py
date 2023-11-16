"""Store and retrieve weaves in files.

There is one format marker followed by a blank line, followed by a
series of version headers, followed by the weave itself.

Each version marker has

 'i'   parent version indexes
 '1'   SHA-1 of text
 'n'   name

The inclusions do not need to list versions included by a parent.

The weave is bracketed by 'w' and 'W' lines, and includes the '{}[]'
processing instructions.  Lines of text are prefixed by '.' if the
line contains a newline, or ',' if not.
"""
from __future__ import absolute_import
FORMAT_1 = '# bzr weave file v5\n'

def write_weave(weave, f, format=None):
    if False:
        while True:
            i = 10
    if format is None or format == 1:
        return write_weave_v5(weave, f)
    else:
        raise ValueError('unknown weave format %r' % format)

def write_weave_v5(weave, f):
    if False:
        i = 10
        return i + 15
    'Write weave to file f.'
    f.write(FORMAT_1)
    for (version, included) in enumerate(weave._parents):
        if included:
            mininc = included
            f.write('i ')
            f.write(' '.join((str(i) for i in mininc)))
            f.write('\n')
        else:
            f.write('i\n')
        f.write('1 ' + weave._sha1s[version] + '\n')
        f.write('n ' + weave._names[version] + '\n')
        f.write('\n')
    f.write('w\n')
    for l in weave._weave:
        if isinstance(l, tuple):
            if l[0] == '}':
                f.write('}\n')
            else:
                f.write('%s %d\n' % l)
        elif not l:
            f.write(', \n')
        elif l[-1] == '\n':
            f.write('. ' + l)
        else:
            f.write(', ' + l + '\n')
    f.write('W\n')

def read_weave(f):
    if False:
        while True:
            i = 10
    from bzrlib.weave import Weave
    w = Weave(getattr(f, 'name', None))
    _read_weave_v5(f, w)
    return w

def _read_weave_v5(f, w):
    if False:
        i = 10
        return i + 15
    'Private helper routine to read a weave format 5 file into memory.\n\n    This is only to be used by read_weave and WeaveFile.__init__.\n    '
    from bzrlib.weave import WeaveFormatError
    try:
        lines = iter(f.readlines())
    finally:
        f.close()
    try:
        l = lines.next()
    except StopIteration:
        raise WeaveFormatError('invalid weave file: no header')
    if l != FORMAT_1:
        raise WeaveFormatError('invalid weave file header: %r' % l)
    ver = 0
    while True:
        l = lines.next()
        if l[0] == 'i':
            if len(l) > 2:
                w._parents.append(map(int, l[2:].split(' ')))
            else:
                w._parents.append([])
            l = lines.next()[:-1]
            w._sha1s.append(l[2:])
            l = lines.next()
            name = l[2:-1]
            w._names.append(name)
            w._name_map[name] = ver
            l = lines.next()
            ver += 1
        elif l == 'w\n':
            break
        else:
            raise WeaveFormatError('unexpected line %r' % l)
    while True:
        l = lines.next()
        if l == 'W\n':
            break
        elif '. ' == l[0:2]:
            w._weave.append(l[2:])
        elif ', ' == l[0:2]:
            w._weave.append(l[2:-1])
        elif l == '}\n':
            w._weave.append(('}', None))
        else:
            w._weave.append((intern(l[0]), int(l[2:])))
    return w