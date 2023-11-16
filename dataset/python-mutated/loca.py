__license__ = 'GPL v3'
__copyright__ = '2012, Kovid Goyal <kovid at kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import array, sys
from operator import itemgetter
from itertools import repeat
from calibre.utils.fonts.sfnt import UnknownTable
from polyglot.builtins import iteritems

def four_byte_type_code():
    if False:
        print('Hello World!')
    for c in 'IL':
        a = array.array(c)
        if a.itemsize == 4:
            return c

def read_array(data, fmt='H'):
    if False:
        for i in range(10):
            print('nop')
    ans = array.array(fmt, data)
    if sys.byteorder != 'big':
        ans.byteswap()
    return ans

class LocaTable(UnknownTable):

    def load_offsets(self, head_table, maxp_table):
        if False:
            for i in range(10):
                print('nop')
        fmt = 'H' if head_table.index_to_loc_format == 0 else four_byte_type_code()
        locs = read_array(self.raw, fmt)
        self.offset_map = locs.tolist()
        if fmt == 'H':
            self.offset_map = [2 * i for i in self.offset_map]
        self.fmt = fmt

    def glyph_location(self, glyph_id):
        if False:
            for i in range(10):
                print('nop')
        offset = self.offset_map[glyph_id]
        next_offset = self.offset_map[glyph_id + 1]
        return (offset, next_offset - offset)

    def update(self, resolved_glyph_map):
        if False:
            print('Hello World!')
        '\n        Update this table to contain pointers only to the glyphs in\n        resolved_glyph_map which must be a map of glyph_ids to (offset, sz)\n        Note that the loca table is generated for all glyphs from 0 to the\n        largest glyph that is either in resolved_glyph_map or was present\n        originally. The pointers to glyphs that have no data will be set to\n        zero. This preserves glyph ids.\n        '
        current_max_glyph_id = len(self.offset_map) - 2
        max_glyph_id = max(resolved_glyph_map or (0,))
        max_glyph_id = max(max_glyph_id, current_max_glyph_id)
        self.offset_map = list(repeat(0, max_glyph_id + 2))
        glyphs = [(glyph_id, x[0], x[1]) for (glyph_id, x) in iteritems(resolved_glyph_map)]
        glyphs.sort(key=itemgetter(1))
        for (glyph_id, offset, sz) in glyphs:
            self.offset_map[glyph_id] = offset
            self.offset_map[glyph_id + 1] = offset + sz
        for i in range(1, len(self.offset_map)):
            if self.offset_map[i] == 0:
                self.offset_map[i] = self.offset_map[i - 1]
        vals = self.offset_map
        max_offset = max(vals) if vals else 0
        if max_offset < 131072 and all((l % 2 == 0 for l in vals)):
            self.fmt = 'H'
            vals = array.array(self.fmt, (i // 2 for i in vals))
        else:
            self.fmt = four_byte_type_code()
            vals = array.array(self.fmt, vals)
        if sys.byteorder != 'big':
            vals.byteswap()
        self.raw = vals.tobytes()
    subset = update

    def dump_glyphs(self, sfnt):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(self, 'offset_map'):
            self.load_offsets(sfnt[b'head'], sfnt[b'maxp'])
        for i in range(len(self.offset_map) - 1):
            (off, noff) = (self.offset_map[i], self.offset_map[i + 1])
            if noff != off:
                print('Glyph id:', i, 'size:', noff - off)