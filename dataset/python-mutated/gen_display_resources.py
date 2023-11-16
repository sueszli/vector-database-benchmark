import argparse
import os
import struct
import sys
sys.path.insert(0, 'bitmap_font')
sys.path.insert(0, '../../tools/bitmap_font')
from adafruit_bitmap_font import bitmap_font
parser = argparse.ArgumentParser(description='Generate displayio resources.')
parser.add_argument('--font', type=str, help='Font path', required=True)
parser.add_argument('--extra_characters', type=str, help='Unicode string of extra characters')
parser.add_argument('--sample_file', type=argparse.FileType('r', encoding='utf-8'), help='Text file that includes strings to support.')
parser.add_argument('--output_c_file', type=argparse.FileType('w'), required=True)
args = parser.parse_args()

class BitmapStub:

    def __init__(self, width, height, color_depth):
        if False:
            while True:
                i = 10
        self.width = width
        self.rows = [b''] * height

    def _load_row(self, y, row):
        if False:
            print('Hello World!')
        self.rows[y] = bytes(row)
f = bitmap_font.load_font(args.font, BitmapStub)
sample_characters = set()
if args.sample_file:
    for line in args.sample_file:
        if line.startswith('//'):
            continue
        for c in line.strip():
            sample_characters.add(c)
visible_ascii = bytes(range(32, 127)).decode('utf-8')
all_characters = list(visible_ascii)
for c in sample_characters:
    if c not in all_characters:
        all_characters += c
if args.extra_characters:
    all_characters.extend(args.extra_characters)
all_characters = ''.join(sorted(set(all_characters)))
filtered_characters = all_characters
f.load_glyphs(set((ord(c) for c in all_characters)))
missing = 0
for c in set(all_characters):
    if ord(c) not in f._glyphs:
        missing += 1
        filtered_characters = filtered_characters.replace(c, '')
        continue
    g = f.get_glyph(ord(c))
    if g['shift'][1] != 0:
        raise RuntimeError('y shift')
if missing > 0:
    print('Font missing', missing, 'characters', file=sys.stderr)
(tile_x, tile_y, dx, dy) = f.get_bounding_box()
total_bits = tile_x * len(all_characters)
total_bits += 32 - total_bits % 32
bytes_per_row = total_bits // 8
b = bytearray(bytes_per_row * tile_y)
for (x, c) in enumerate(filtered_characters):
    g = f.get_glyph(ord(c))
    start_bit = x * tile_x + g['bounds'][2]
    start_y = tile_y - 2 - (g['bounds'][1] + g['bounds'][3])
    for (y, row) in enumerate(g['bitmap'].rows):
        for i in range(g['bounds'][0]):
            byte = i // 8
            bit = i % 8
            if row[byte] & 1 << 7 - bit != 0:
                overall_bit = start_bit + (start_y + y) * bytes_per_row * 8 + i
                b[overall_bit // 8] |= 1 << 7 - overall_bit % 8
extra_characters = ''
for c in filtered_characters:
    if c not in visible_ascii:
        extra_characters += c
c_file = args.output_c_file
c_file.write('\n#include "shared-bindings/displayio/Bitmap.h"\n#include "shared-bindings/displayio/Palette.h"\n#include "supervisor/shared/display.h"\n\n')
c_file.write('#if CIRCUITPY_REPL_LOGO\n')
if tile_y == 16:
    blinka_size = 16
    c_file.write('const uint32_t blinka_bitmap_data[32] = {\n    0x00000011, 0x11000000,\n    0x00000111, 0x53100000,\n    0x00000111, 0x56110000,\n    0x00000111, 0x11140000,\n    0x00000111, 0x20002000,\n    0x00000011, 0x13000000,\n    0x00000001, 0x11200000,\n    0x00000000, 0x11330000,\n    0x00000000, 0x01122000,\n    0x00001111, 0x44133000,\n    0x00032323, 0x24112200,\n    0x00111114, 0x44113300,\n    0x00323232, 0x34112200,\n    0x11111144, 0x44443300,\n    0x11111111, 0x11144401,\n    0x23232323, 0x21111110\n};\n')
else:
    blinka_size = 12
    c_file.write('const uint32_t blinka_bitmap_data[28] = {\n    0x00000111, 0x00000000,\n    0x00001153, 0x10000000,\n    0x00001156, 0x11000000,\n    0x00001111, 0x14000000,\n    0x00000112, 0x00200000,\n    0x00000011, 0x30000000,\n    0x00000011, 0x20000000,\n    0x00011144, 0x13000000,\n    0x00232324, 0x12000000,\n    0x01111444, 0x13000000,\n    0x32323234, 0x12010000,\n    0x11111144, 0x44100000\n};\n')
c_file.write('const displayio_bitmap_t blinka_bitmap = {{\n    .base = {{.type = &displayio_bitmap_type }},\n    .width = {0},\n    .height = {0},\n    .data = (uint32_t*) blinka_bitmap_data,\n    .stride = 2,\n    .bits_per_value = 4,\n    .x_shift = 3,\n    .x_mask = 0x7,\n    .bitmask = 0xf,\n    .read_only = true\n}};\n\n_displayio_color_t blinka_colors[7] = {{\n    {{\n        .rgb888 = 0x000000,\n        .transparent = true\n    }},\n    {{ // Purple\n        .rgb888 = 0x8428bc\n    }},\n    {{ // Pink\n        .rgb888 = 0xff89bc\n    }},\n    {{ // Light blue\n        .rgb888 = 0x7beffe\n    }},\n    {{ // Dark purple\n        .rgb888 = 0x51395f\n    }},\n    {{ // White\n        .rgb888 = 0xffffff\n    }},\n    {{ // Dark Blue\n        .rgb888 = 0x0736a0\n    }},\n}};\n\ndisplayio_palette_t blinka_palette = {{\n    .base = {{.type = &displayio_palette_type }},\n    .colors = blinka_colors,\n    .color_count = 7,\n    .needs_refresh = false\n}};\n\ndisplayio_tilegrid_t supervisor_blinka_sprite = {{\n    .base = {{.type = &displayio_tilegrid_type }},\n    .bitmap = (displayio_bitmap_t*) &blinka_bitmap,\n    .pixel_shader = &blinka_palette,\n    .x = 0,\n    .y = 0,\n    .pixel_width = {0},\n    .pixel_height = {0},\n    .bitmap_width_in_tiles = 1,\n    .width_in_tiles = 1,\n    .height_in_tiles = 1,\n    .tile_width = {0},\n    .tile_height = {0},\n    .top_left_x = {0},\n    .top_left_y = {0},\n    .tiles = 0,\n    .partial_change = false,\n    .full_change = false,\n    .hidden = false,\n    .hidden_by_parent = false,\n    .moved = false,\n    .inline_tiles = true,\n    .in_group = true\n}};\n#endif\n'.format(blinka_size))
c_file.write('#if CIRCUITPY_TERMINALIO\n_displayio_color_t terminal_colors[2] = {\n    {\n        .rgb888 = 0x000000\n    },\n    {\n        .rgb888 = 0xffffff\n    },\n};\n\ndisplayio_palette_t supervisor_terminal_color = {\n    .base = {.type = &displayio_palette_type },\n    .colors = terminal_colors,\n    .color_count = 2,\n    .needs_refresh = false\n};\n')
c_file.write('displayio_tilegrid_t supervisor_terminal_scroll_area_text_grid = {{\n    .base = {{ .type = &displayio_tilegrid_type }},\n    .bitmap = (displayio_bitmap_t*) &supervisor_terminal_font_bitmap,\n    .pixel_shader = &supervisor_terminal_color,\n    .x = 0,\n    .y = 0,\n    .pixel_width = {1},\n    .pixel_height = {2},\n    .bitmap_width_in_tiles = {0},\n    .tiles_in_bitmap = {0},\n    .width_in_tiles = 1,\n    .height_in_tiles = 1,\n    .tile_width = {1},\n    .tile_height = {2},\n    .tiles = NULL,\n    .partial_change = false,\n    .full_change = false,\n    .hidden = false,\n    .hidden_by_parent = false,\n    .moved = false,\n    .inline_tiles = false,\n    .in_group = true\n}};\n'.format(len(all_characters), tile_x, tile_y))
c_file.write('displayio_tilegrid_t supervisor_terminal_status_bar_text_grid = {{\n    .base = {{ .type = &displayio_tilegrid_type }},\n    .bitmap = (displayio_bitmap_t*) &supervisor_terminal_font_bitmap,\n    .pixel_shader = &supervisor_terminal_color,\n    .x = 0,\n    .y = 0,\n    .pixel_width = {1},\n    .pixel_height = {2},\n    .bitmap_width_in_tiles = {0},\n    .tiles_in_bitmap = {0},\n    .width_in_tiles = 1,\n    .height_in_tiles = 1,\n    .tile_width = {1},\n    .tile_height = {2},\n    .tiles = NULL,\n    .partial_change = false,\n    .full_change = false,\n    .hidden = false,\n    .hidden_by_parent = false,\n    .moved = false,\n    .inline_tiles = false,\n    .in_group = true\n}};\n'.format(len(all_characters), tile_x, tile_y))
c_file.write('const uint32_t font_bitmap_data[{}] = {{\n'.format(bytes_per_row * tile_y // 4))
for (i, word) in enumerate(struct.iter_unpack('>I', b)):
    c_file.write('0x{:08x}, '.format(word[0]))
    if (i + 1) % (bytes_per_row // 4) == 0:
        c_file.write('\n')
c_file.write('};\n')
c_file.write('displayio_bitmap_t supervisor_terminal_font_bitmap = {{\n    .base = {{.type = &displayio_bitmap_type }},\n    .width = {},\n    .height = {},\n    .data = (uint32_t*) font_bitmap_data,\n    .stride = {},\n    .bits_per_value = 1,\n    .x_shift = 5,\n    .x_mask = 0x1f,\n    .bitmask = 0x1,\n    .read_only = true\n}};\n'.format(len(all_characters) * tile_x, tile_y, bytes_per_row / 4))
c_file.write('const fontio_builtinfont_t supervisor_terminal_font = {{\n    .base = {{.type = &fontio_builtinfont_type }},\n    .bitmap = &supervisor_terminal_font_bitmap,\n    .width = {},\n    .height = {},\n    .unicode_characters = (const uint8_t*) "{}",\n    .unicode_characters_len = {}\n}};\n'.format(tile_x, tile_y, extra_characters, len(extra_characters.encode('utf-8'))))
c_file.write('terminalio_terminal_obj_t supervisor_terminal = {\n    .base = { .type = &terminalio_terminal_type },\n    .font = &supervisor_terminal_font,\n    .cursor_x = 0,\n    .cursor_y = 0,\n    .scroll_area = NULL,\n    .status_bar = NULL\n};\n\n#endif\n')