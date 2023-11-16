from __future__ import absolute_import
from bzrlib.errors import BzrError
from bzrlib.inventory import Inventory
START_MARK = '# bzr inventory format 3\n'
END_MARK = '# end of inventory\n'

def escape(s):
    if False:
        return 10
    "Very simple URL-like escaping.\n\n    (Why not just use backslashes?  Because then we couldn't parse\n    lines just by splitting on spaces.)"
    return s.replace('\\', '\\x5c').replace(' ', '\\x20').replace('\t', '\\x09').replace('\n', '\\x0a')

def unescape(s):
    if False:
        while True:
            i = 10
    if s.find(' ') != -1:
        raise AssertionError()
    s = s.replace('\\x20', ' ').replace('\\x09', '\t').replace('\\x0a', '\n').replace('\\x5c', '\\')
    return s

def write_text_inventory(inv, outf):
    if False:
        print('Hello World!')
    'Write out inv in a simple trad-unix text format.'
    outf.write(START_MARK)
    for (path, ie) in inv.iter_entries():
        if inv.is_root(ie.file_id):
            continue
        outf.write(ie.file_id + ' ')
        outf.write(escape(ie.name) + ' ')
        outf.write(ie.kind + ' ')
        outf.write(ie.parent_id + ' ')
        if ie.kind == 'file':
            outf.write(ie.text_id)
            outf.write(' ' + ie.text_sha1)
            outf.write(' ' + str(ie.text_size))
        outf.write('\n')
    outf.write(END_MARK)

def read_text_inventory(tf):
    if False:
        for i in range(10):
            print('nop')
    'Return an inventory read in from tf'
    if tf.readline() != START_MARK:
        raise BzrError('missing start mark')
    inv = Inventory()
    for l in tf:
        fields = l.split(' ')
        if fields[0] == '#':
            break
        ie = {'file_id': fields[0], 'name': unescape(fields[1]), 'kind': fields[2], 'parent_id': fields[3]}
    if l != END_MARK:
        raise BzrError('missing end mark')
    return inv