__license__ = 'GPL v3'
__copyright__ = '2012, Kovid Goyal <kovid at kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from struct import unpack_from
from collections import OrderedDict
from calibre.utils.fonts.sfnt import UnknownTable
from polyglot.builtins import iteritems
ARG_1_AND_2_ARE_WORDS = 1
ARGS_ARE_XY_VALUES = 2
ROUND_XY_TO_GRID = 4
WE_HAVE_A_SCALE = 8
NON_OVERLAPPING = 16
MORE_COMPONENTS = 32
WE_HAVE_AN_X_AND_Y_SCALE = 64
WE_HAVE_A_TWO_BY_TWO = 128
WE_HAVE_INSTRUCTIONS = 256
USE_MY_METRICS = 512
OVERLAP_COMPOUND = 1024
SCALED_COMPONENT_OFFSET = 2048
UNSCALED_COMPONENT_OFFSET = 4096

class SimpleGlyph:

    def __init__(self, num_of_countours, raw):
        if False:
            print('Hello World!')
        self.num_of_countours = num_of_countours
        self.raw = raw
        self.glyph_indices = []
        self.is_composite = False

    def __len__(self):
        if False:
            return 10
        return len(self.raw)

    def __call__(self):
        if False:
            while True:
                i = 10
        return self.raw

class CompositeGlyph(SimpleGlyph):

    def __init__(self, num_of_countours, raw):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(num_of_countours, raw)
        self.is_composite = True
        flags = MORE_COMPONENTS
        offset = 10
        while flags & MORE_COMPONENTS:
            (flags, glyph_index) = unpack_from(b'>HH', raw, offset)
            self.glyph_indices.append(glyph_index)
            offset += 4
            if flags & ARG_1_AND_2_ARE_WORDS:
                offset += 4
            else:
                offset += 2
            if flags & WE_HAVE_A_SCALE:
                offset += 2
            elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
                offset += 4
            elif flags & WE_HAVE_A_TWO_BY_TWO:
                offset += 8

class GlyfTable(UnknownTable):

    def glyph_data(self, offset, length, as_raw=False):
        if False:
            for i in range(10):
                print('nop')
        raw = self.raw[offset:offset + length]
        if as_raw:
            return raw
        num_of_countours = unpack_from(b'>h', raw)[0] if raw else 0
        if num_of_countours >= 0:
            return SimpleGlyph(num_of_countours, raw)
        return CompositeGlyph(num_of_countours, raw)

    def update(self, sorted_glyph_map):
        if False:
            while True:
                i = 10
        ans = OrderedDict()
        offset = 0
        block = []
        for (glyph_id, glyph) in iteritems(sorted_glyph_map):
            raw = glyph()
            pad = 4 - len(raw) % 4
            if pad < 4:
                raw += b'\x00' * pad
            ans[glyph_id] = (offset, len(raw))
            offset += len(raw)
            block.append(raw)
        self.raw = b''.join(block)
        return ans