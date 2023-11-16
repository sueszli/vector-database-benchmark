import logging
from array import array
from copy import deepcopy
from ctypes import c_void_p
from random import getrandbits
from pyboy import utils
logger = logging.getLogger(__name__)
VIDEO_RAM = 8 * 1024
OBJECT_ATTRIBUTE_MEMORY = 160
(INTR_VBLANK, INTR_LCDC, INTR_TIMER, INTR_SERIAL, INTR_HIGHTOLOW) = [1 << x for x in range(5)]
(ROWS, COLS) = (144, 160)
TILES = 384
FRAME_CYCLES = 70224
try:
    from cython import compiled
    cythonmode = compiled
except ImportError:
    cythonmode = False

class LCD:

    def __init__(self, cgb, cartridge_cgb, disable_renderer, color_palette, cgb_color_palette, randomize=False):
        if False:
            return 10
        self.VRAM0 = array('B', [0] * VIDEO_RAM)
        self.OAM = array('B', [0] * OBJECT_ATTRIBUTE_MEMORY)
        self.disable_renderer = disable_renderer
        if randomize:
            for i in range(VIDEO_RAM):
                self.VRAM0[i] = getrandbits(8)
            for i in range(OBJECT_ATTRIBUTE_MEMORY):
                self.OAM[i] = getrandbits(8)
        self._LCDC = LCDCRegister(0)
        self._STAT = STATRegister()
        self.next_stat_mode = 2
        self.SCY = 0
        self.SCX = 0
        self.LY = 0
        self.LYC = 0
        self.BGP = PaletteRegister(252)
        self.OBP0 = PaletteRegister(255)
        self.OBP1 = PaletteRegister(255)
        self.WY = 0
        self.WX = 0
        self.clock = 0
        self.clock_target = 0
        self.frame_done = False
        self.double_speed = False
        self.cgb = cgb
        if self.cgb:
            if cartridge_cgb:
                logger.debug('Starting CGB renderer')
                self.renderer = CGBRenderer()
            else:
                logger.debug('Starting CGB renderer in DMG-mode')
                (bg_pal, obj0_pal, obj1_pal) = cgb_color_palette
                self.BGP.palette_mem_rgb = [c << 8 for c in bg_pal]
                self.OBP0.palette_mem_rgb = [c << 8 for c in obj0_pal]
                self.OBP1.palette_mem_rgb = [c << 8 for c in obj1_pal]
                self.renderer = Renderer(False)
        else:
            logger.debug('Starting DMG renderer')
            self.BGP.palette_mem_rgb = [c << 8 for c in color_palette]
            self.OBP0.palette_mem_rgb = [c << 8 for c in color_palette]
            self.OBP1.palette_mem_rgb = [c << 8 for c in color_palette]
            self.renderer = Renderer(False)
        self.BGP.palette_mem_rgb[0] |= COL0_FLAG
        self.OBP0.palette_mem_rgb[0] |= COL0_FLAG
        self.OBP1.palette_mem_rgb[0] |= COL0_FLAG

    def get_lcdc(self):
        if False:
            i = 10
            return i + 15
        return self._LCDC.value

    def set_lcdc(self, value):
        if False:
            print('Hello World!')
        self._LCDC.set(value)
        if not self._LCDC.lcd_enable:
            self.clock = 0
            self.clock_target = FRAME_CYCLES
            self._STAT.set_mode(0)
            self.next_stat_mode = 2
            self.LY = 0

    def get_stat(self):
        if False:
            while True:
                i = 10
        return self._STAT.value

    def set_stat(self, value):
        if False:
            while True:
                i = 10
        self._STAT.set(value)

    def cycles_to_interrupt(self):
        if False:
            i = 10
            return i + 15
        return self.clock_target - self.clock

    def cycles_to_mode0(self):
        if False:
            return 10
        multiplier = 2 if self.double_speed else 1
        mode2 = 80 * multiplier
        mode3 = 170 * multiplier
        mode1 = 456 * multiplier
        mode = self._STAT._mode
        remainder = self.clock_target - self.clock
        mode &= 3
        if mode == 2:
            return remainder + mode3
        elif mode == 3:
            return remainder
        elif mode == 0:
            return 0
        elif mode == 1:
            remaining_ly = 153 - self.LY
            return remainder + mode1 * remaining_ly + mode2 + mode3

    def tick(self, cycles):
        if False:
            print('Hello World!')
        interrupt_flag = 0
        self.clock += cycles
        if self._LCDC.lcd_enable:
            if self.clock >= self.clock_target:
                interrupt_flag |= self._STAT.set_mode(self.next_stat_mode)
                multiplier = 2 if self.double_speed else 1
                if self._STAT._mode == 2:
                    if self.LY == 153:
                        self.LY = 0
                        self.clock %= FRAME_CYCLES
                        self.clock_target %= FRAME_CYCLES
                    else:
                        self.LY += 1
                    self.clock_target += 80 * multiplier
                    self.next_stat_mode = 3
                    interrupt_flag |= self._STAT.update_LYC(self.LYC, self.LY)
                elif self._STAT._mode == 3:
                    self.clock_target += 170 * multiplier
                    self.next_stat_mode = 0
                elif self._STAT._mode == 0:
                    self.clock_target += 206 * multiplier
                    self.renderer.scanline(self, self.LY)
                    self.renderer.scanline_sprites(self, self.LY, self.renderer._screenbuffer, False)
                    if self.LY < 143:
                        self.next_stat_mode = 2
                    else:
                        self.next_stat_mode = 1
                elif self._STAT._mode == 1:
                    self.clock_target += 456 * multiplier
                    self.next_stat_mode = 1
                    self.LY += 1
                    interrupt_flag |= self._STAT.update_LYC(self.LYC, self.LY)
                    if self.LY == 144:
                        interrupt_flag |= INTR_VBLANK
                        self.frame_done = True
                    if self.LY == 153:
                        self.next_stat_mode = 2
        elif self.clock >= FRAME_CYCLES:
            self.frame_done = True
            self.clock %= FRAME_CYCLES
            self.renderer.blank_screen(self)
        return interrupt_flag

    def save_state(self, f):
        if False:
            while True:
                i = 10
        for n in range(VIDEO_RAM):
            f.write(self.VRAM0[n])
        for n in range(OBJECT_ATTRIBUTE_MEMORY):
            f.write(self.OAM[n])
        f.write(self._LCDC.value)
        f.write(self.BGP.value)
        f.write(self.OBP0.value)
        f.write(self.OBP1.value)
        f.write(self._STAT.value)
        f.write(self.LY)
        f.write(self.LYC)
        f.write(self.SCY)
        f.write(self.SCX)
        f.write(self.WY)
        f.write(self.WX)
        f.write(self.cgb)
        f.write(self.double_speed)
        f.write_64bit(self.clock)
        f.write_64bit(self.clock_target)
        f.write(self.next_stat_mode)
        if self.cgb:
            for n in range(VIDEO_RAM):
                f.write(self.VRAM1[n])
            f.write(self.vbk.active_bank)
            self.bcps.save_state(f)
            self.bcpd.save_state(f)
            self.ocps.save_state(f)
            self.ocpd.save_state(f)

    def load_state(self, f, state_version):
        if False:
            for i in range(10):
                print('nop')
        for n in range(VIDEO_RAM):
            self.VRAM0[n] = f.read()
        for n in range(OBJECT_ATTRIBUTE_MEMORY):
            self.OAM[n] = f.read()
        self.set_lcdc(f.read())
        self.BGP.set(f.read())
        self.OBP0.set(f.read())
        self.OBP1.set(f.read())
        if state_version >= 5:
            self.set_stat(f.read())
            self.LY = f.read()
            self.LYC = f.read()
        self.SCY = f.read()
        self.SCX = f.read()
        self.WY = f.read()
        self.WX = f.read()
        if state_version >= 8:
            _cgb = f.read()
            if self.cgb != _cgb:
                logger.critical(f'Loading state which is not CGB, but PyBoy is loaded in CGB mode!')
                return
            self.cgb = _cgb
            self.double_speed = f.read()
            self.clock = f.read_64bit()
            self.clock_target = f.read_64bit()
            self.next_stat_mode = f.read()
            if self.cgb:
                for n in range(VIDEO_RAM):
                    self.VRAM1[n] = f.read()
                self.vbk.active_bank = f.read()
                self.bcps.load_state(f, state_version)
                self.bcpd.load_state(f, state_version)
                self.ocps.load_state(f, state_version)
                self.ocpd.load_state(f, state_version)

    def getwindowpos(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.WX - 7, self.WY)

    def getviewport(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.SCX, self.SCY)

class PaletteRegister:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = 0
        self.lookup = [0] * 4
        self.set(value)
        self.palette_mem_rgb = [0] * 4

    def set(self, value):
        if False:
            print('Hello World!')
        if self.value == value:
            return False
        self.value = value
        for x in range(4):
            self.lookup[x] = value >> x * 2 & 3
        return True

    def get(self):
        if False:
            return 10
        return self.value

    def getcolor(self, i):
        if False:
            print('Hello World!')
        return self.palette_mem_rgb[self.lookup[i]]

class STATRegister:

    def __init__(self):
        if False:
            return 10
        self.value = 128
        self._mode = 0

    def set(self, value):
        if False:
            i = 10
            return i + 15
        value &= 120
        self.value &= 135
        self.value |= value

    def update_LYC(self, LYC, LY):
        if False:
            for i in range(10):
                print('nop')
        if LYC == LY:
            self.value |= 4
            if self.value & 64:
                return INTR_LCDC
        else:
            self.value &= 251
        return 0

    def set_mode(self, mode):
        if False:
            print('Hello World!')
        if self._mode == mode:
            return 0
        self._mode = mode
        self.value &= 252
        self.value |= mode
        if mode != 3 and self.value & 1 << mode + 3:
            return INTR_LCDC
        return 0

class LCDCRegister:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.set(value)

    def set(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.lcd_enable = value & 1 << 7
        self.windowmap_select = value & 1 << 6
        self.window_enable = value & 1 << 5
        self.tiledata_select = value & 1 << 4
        self.backgroundmap_select = value & 1 << 3
        self.sprite_height = value & 1 << 2
        self.sprite_enable = value & 1 << 1
        self.background_enable = value & 1 << 0
        self.cgb_master_priority = self.background_enable
COL0_FLAG = 1
BG_PRIORITY_FLAG = 2

class Renderer:

    def __init__(self, cgb):
        if False:
            i = 10
            return i + 15
        self.cgb = cgb
        self.color_format = 'RGBA'
        self.buffer_dims = (ROWS, COLS)
        self._screenbuffer_raw = array('B', [0] * (ROWS * COLS * 4))
        self._tilecache0_raw = array('B', [0] * (TILES * 8 * 8 * 4))
        self._spritecache0_raw = array('B', [0] * (TILES * 8 * 8 * 4))
        self._spritecache1_raw = array('B', [0] * (TILES * 8 * 8 * 4))
        self.sprites_to_render = array('i', [0] * 10)
        self._tilecache0_state = array('B', [0] * TILES)
        self._spritecache0_state = array('B', [0] * TILES)
        self._spritecache1_state = array('B', [0] * TILES)
        self.clear_cache()
        self._screenbuffer = memoryview(self._screenbuffer_raw).cast('I', shape=(ROWS, COLS))
        self._tilecache0 = memoryview(self._tilecache0_raw).cast('I', shape=(TILES * 8, 8))
        self._spritecache0 = memoryview(self._spritecache0_raw).cast('I', shape=(TILES * 8, 8))
        self._spritecache1 = memoryview(self._spritecache1_raw).cast('I', shape=(TILES * 8, 8))
        self._screenbuffer_ptr = c_void_p(self._screenbuffer_raw.buffer_info()[0])
        self._scanlineparameters = [[0, 0, 0, 0, 0] for _ in range(ROWS)]
        self.ly_window = 0

    def _cgb_get_background_map_attributes(self, lcd, i):
        if False:
            i = 10
            return i + 15
        tile_num = lcd.VRAM1[i]
        palette = tile_num & 7
        vbank = tile_num >> 3 & 1
        horiflip = tile_num >> 5 & 1
        vertflip = tile_num >> 6 & 1
        bg_priority = tile_num >> 7 & 1
        return (palette, vbank, horiflip, vertflip, bg_priority)

    def scanline(self, lcd, y):
        if False:
            while True:
                i = 10
        (bx, by) = lcd.getviewport()
        (wx, wy) = lcd.getwindowpos()
        self._scanlineparameters[y][0] = bx
        self._scanlineparameters[y][1] = by
        self._scanlineparameters[y][2] = wx
        self._scanlineparameters[y][3] = wy
        self._scanlineparameters[y][4] = lcd._LCDC.tiledata_select
        if lcd.disable_renderer:
            return
        background_offset = 6144 if lcd._LCDC.backgroundmap_select == 0 else 7168
        wmap = 6144 if lcd._LCDC.windowmap_select == 0 else 7168
        offset = bx & 7
        if lcd._LCDC.window_enable and wy <= y and (wx < COLS):
            self.ly_window += 1
        for x in range(COLS):
            if lcd._LCDC.window_enable and wy <= y and (wx <= x):
                tile_addr = wmap + self.ly_window // 8 * 32 % 1024 + (x - wx) // 8 % 32
                wt = lcd.VRAM0[tile_addr]
                if not lcd._LCDC.tiledata_select:
                    wt = (wt ^ 128) + 128
                bg_priority_apply = 0
                if self.cgb:
                    (palette, vbank, horiflip, vertflip, bg_priority) = self._cgb_get_background_map_attributes(lcd, tile_addr)
                    if vbank:
                        self.update_tilecache1(lcd, wt, vbank)
                        tilecache = self._tilecache1
                    else:
                        self.update_tilecache0(lcd, wt, vbank)
                        tilecache = self._tilecache0
                    xx = 7 - (x - wx) % 8 if horiflip else (x - wx) % 8
                    yy = 8 * wt + (7 - self.ly_window % 8) if vertflip else 8 * wt + self.ly_window % 8
                    pixel = lcd.bcpd.getcolor(palette, tilecache[yy, xx])
                    if bg_priority:
                        bg_priority_apply = BG_PRIORITY_FLAG
                else:
                    self.update_tilecache0(lcd, wt, 0)
                    xx = (x - wx) % 8
                    yy = 8 * wt + self.ly_window % 8
                    pixel = lcd.BGP.getcolor(self._tilecache0[yy, xx])
                self._screenbuffer[y, x] = pixel | bg_priority_apply
            elif not self.cgb and lcd._LCDC.background_enable or self.cgb:
                tile_addr = background_offset + (y + by) // 8 * 32 % 1024 + (x + bx) // 8 % 32
                bt = lcd.VRAM0[tile_addr]
                if not lcd._LCDC.tiledata_select:
                    bt = (bt ^ 128) + 128
                bg_priority_apply = 0
                if self.cgb:
                    (palette, vbank, horiflip, vertflip, bg_priority) = self._cgb_get_background_map_attributes(lcd, tile_addr)
                    if vbank:
                        self.update_tilecache1(lcd, bt, vbank)
                        tilecache = self._tilecache1
                    else:
                        self.update_tilecache0(lcd, bt, vbank)
                        tilecache = self._tilecache0
                    xx = 7 - (x + offset) % 8 if horiflip else (x + offset) % 8
                    yy = 8 * bt + (7 - (y + by) % 8) if vertflip else 8 * bt + (y + by) % 8
                    pixel = lcd.bcpd.getcolor(palette, tilecache[yy, xx])
                    if bg_priority:
                        bg_priority_apply = BG_PRIORITY_FLAG
                else:
                    self.update_tilecache0(lcd, bt, 0)
                    xx = (x + offset) % 8
                    yy = 8 * bt + (y + by) % 8
                    pixel = lcd.BGP.getcolor(self._tilecache0[yy, xx])
                self._screenbuffer[y, x] = pixel | bg_priority_apply
            else:
                self._screenbuffer[y, x] = lcd.BGP.getcolor(0)
        if y == 143:
            self.ly_window = -1

    def sort_sprites(self, sprite_count):
        if False:
            return 10
        for i in range(1, sprite_count):
            key = self.sprites_to_render[i]
            j = i - 1
            while j >= 0 and key > self.sprites_to_render[j]:
                self.sprites_to_render[j + 1] = self.sprites_to_render[j]
                j -= 1
            self.sprites_to_render[j + 1] = key

    def scanline_sprites(self, lcd, ly, buffer, ignore_priority):
        if False:
            print('Hello World!')
        if not lcd._LCDC.sprite_enable or lcd.disable_renderer:
            return
        spriteheight = 16 if lcd._LCDC.sprite_height else 8
        sprite_count = 0
        for n in range(0, 160, 4):
            y = lcd.OAM[n] - 16
            x = lcd.OAM[n + 1] - 8
            if y <= ly < y + spriteheight:
                if self.cgb:
                    self.sprites_to_render[sprite_count] = n
                else:
                    self.sprites_to_render[sprite_count] = x << 16 | n
                sprite_count += 1
            if sprite_count == 10:
                break
        self.sort_sprites(sprite_count)
        for _n in self.sprites_to_render[:sprite_count]:
            if self.cgb:
                n = _n
            else:
                n = _n & 255
            y = lcd.OAM[n] - 16
            x = lcd.OAM[n + 1] - 8
            tileindex = lcd.OAM[n + 2]
            if spriteheight == 16:
                tileindex &= 254
            attributes = lcd.OAM[n + 3]
            xflip = attributes & 32
            yflip = attributes & 64
            spritepriority = attributes & 128 and (not ignore_priority)
            if self.cgb:
                palette = attributes & 7
                if attributes & 8:
                    self.update_spritecache1(lcd, tileindex, 1)
                    if lcd._LCDC.sprite_height:
                        self.update_spritecache1(lcd, tileindex + 1, 1)
                    spritecache = self._spritecache1
                else:
                    self.update_spritecache0(lcd, tileindex, 0)
                    if lcd._LCDC.sprite_height:
                        self.update_spritecache0(lcd, tileindex + 1, 0)
                    spritecache = self._spritecache0
            else:
                palette = 0
                if attributes & 16:
                    self.update_spritecache1(lcd, tileindex, 0)
                    if lcd._LCDC.sprite_height:
                        self.update_spritecache1(lcd, tileindex + 1, 0)
                    spritecache = self._spritecache1
                else:
                    self.update_spritecache0(lcd, tileindex, 0)
                    if lcd._LCDC.sprite_height:
                        self.update_spritecache0(lcd, tileindex + 1, 0)
                    spritecache = self._spritecache0
            dy = ly - y
            yy = spriteheight - dy - 1 if yflip else dy
            for dx in range(8):
                xx = 7 - dx if xflip else dx
                color_code = spritecache[8 * tileindex + yy, xx]
                if 0 <= x < COLS and (not color_code == 0):
                    if self.cgb:
                        pixel = lcd.ocpd.getcolor(palette, color_code)
                        bgmappriority = buffer[ly, x] & BG_PRIORITY_FLAG
                        if lcd._LCDC.cgb_master_priority:
                            if bgmappriority:
                                if buffer[ly, x] & COL0_FLAG:
                                    buffer[ly, x] = pixel
                            elif spritepriority:
                                if buffer[ly, x] & COL0_FLAG:
                                    buffer[ly, x] = pixel
                            else:
                                buffer[ly, x] = pixel
                        else:
                            buffer[ly, x] = pixel
                    else:
                        if attributes & 16:
                            pixel = lcd.OBP1.getcolor(color_code)
                        else:
                            pixel = lcd.OBP0.getcolor(color_code)
                        if spritepriority:
                            if buffer[ly, x] & COL0_FLAG:
                                buffer[ly, x] = pixel
                        else:
                            buffer[ly, x] = pixel
                x += 1
            x -= 8

    def clear_cache(self):
        if False:
            return 10
        self.clear_tilecache0()
        self.clear_spritecache0()
        self.clear_spritecache1()

    def invalidate_tile(self, tile, vbank):
        if False:
            while True:
                i = 10
        if vbank and self.cgb:
            self._tilecache0_state[tile] = 0
            self._tilecache1_state[tile] = 0
            self._spritecache0_state[tile] = 0
            self._spritecache1_state[tile] = 0
        else:
            self._tilecache0_state[tile] = 0
            if self.cgb:
                self._tilecache1_state[tile] = 0
            self._spritecache0_state[tile] = 0
            self._spritecache1_state[tile] = 0

    def clear_tilecache0(self):
        if False:
            while True:
                i = 10
        for i in range(TILES):
            self._tilecache0_state[i] = 0

    def clear_tilecache1(self):
        if False:
            print('Hello World!')
        pass

    def clear_spritecache0(self):
        if False:
            return 10
        for i in range(TILES):
            self._spritecache0_state[i] = 0

    def clear_spritecache1(self):
        if False:
            return 10
        for i in range(TILES):
            self._spritecache1_state[i] = 0

    def update_tilecache0(self, lcd, t, bank):
        if False:
            while True:
                i = 10
        if self._tilecache0_state[t]:
            return
        for k in range(0, 16, 2):
            byte1 = lcd.VRAM0[t * 16 + k]
            byte2 = lcd.VRAM0[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                colorcode = utils.color_code(byte1, byte2, 7 - x)
                self._tilecache0[y, x] = colorcode
        self._tilecache0_state[t] = 1

    def update_tilecache1(self, lcd, t, bank):
        if False:
            while True:
                i = 10
        pass

    def update_spritecache0(self, lcd, t, bank):
        if False:
            print('Hello World!')
        if self._spritecache0_state[t]:
            return
        for k in range(0, 16, 2):
            byte1 = lcd.VRAM0[t * 16 + k]
            byte2 = lcd.VRAM0[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                colorcode = utils.color_code(byte1, byte2, 7 - x)
                self._spritecache0[y, x] = colorcode
        self._spritecache0_state[t] = 1

    def update_spritecache1(self, lcd, t, bank):
        if False:
            while True:
                i = 10
        if self._spritecache1_state[t]:
            return
        for k in range(0, 16, 2):
            byte1 = lcd.VRAM0[t * 16 + k]
            byte2 = lcd.VRAM0[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                colorcode = utils.color_code(byte1, byte2, 7 - x)
                self._spritecache1[y, x] = colorcode
        self._spritecache1_state[t] = 1

    def blank_screen(self, lcd):
        if False:
            while True:
                i = 10
        for y in range(ROWS):
            for x in range(COLS):
                self._screenbuffer[y, x] = lcd.BGP.getcolor(0)

    def save_state(self, f):
        if False:
            print('Hello World!')
        for y in range(ROWS):
            f.write(self._scanlineparameters[y][0])
            f.write(self._scanlineparameters[y][1])
            f.write(self._scanlineparameters[y][2] + 7 & 255)
            f.write(self._scanlineparameters[y][3])
            f.write(self._scanlineparameters[y][4])
        for y in range(ROWS):
            for x in range(COLS):
                f.write_32bit(self._screenbuffer[y, x])

    def load_state(self, f, state_version):
        if False:
            return 10
        if state_version >= 2:
            for y in range(ROWS):
                self._scanlineparameters[y][0] = f.read()
                self._scanlineparameters[y][1] = f.read()
                self._scanlineparameters[y][2] = f.read() - 7 & 255
                self._scanlineparameters[y][3] = f.read()
                if state_version > 3:
                    self._scanlineparameters[y][4] = f.read()
        if state_version >= 6:
            for y in range(ROWS):
                for x in range(COLS):
                    self._screenbuffer[y, x] = f.read_32bit()
        self.clear_cache()

class CGBLCD(LCD):

    def __init__(self, cgb, cartridge_cgb, disable_renderer, color_palette, cgb_color_palette, randomize=False):
        if False:
            for i in range(10):
                print('nop')
        LCD.__init__(self, cgb, cartridge_cgb, disable_renderer, color_palette, cgb_color_palette, randomize=False)
        self.VRAM1 = array('B', [0] * VIDEO_RAM)
        self.vbk = VBKregister()
        self.bcps = PaletteIndexRegister()
        self.bcpd = PaletteColorRegister(self.bcps)
        self.ocps = PaletteIndexRegister()
        self.ocpd = PaletteColorRegister(self.ocps)

class CGBRenderer(Renderer):

    def __init__(self):
        if False:
            print('Hello World!')
        self._tilecache1_state = array('B', [0] * TILES)
        Renderer.__init__(self, True)
        self._tilecache1_raw = array('B', [255] * (TILES * 8 * 8 * 4))
        self._tilecache1 = memoryview(self._tilecache1_raw).cast('I', shape=(TILES * 8, 8))
        self._tilecache1_state = array('B', [0] * TILES)
        self.clear_cache()

    def clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        self.clear_tilecache0()
        self.clear_tilecache1()
        self.clear_spritecache0()
        self.clear_spritecache1()

    def clear_tilecache1(self):
        if False:
            print('Hello World!')
        for i in range(TILES):
            self._tilecache1_state[i] = 0

    def update_tilecache0(self, lcd, t, bank):
        if False:
            i = 10
            return i + 15
        if self._tilecache0_state[t]:
            return
        if bank:
            vram_bank = lcd.VRAM1
        else:
            vram_bank = lcd.VRAM0
        for k in range(0, 16, 2):
            byte1 = vram_bank[t * 16 + k]
            byte2 = vram_bank[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                self._tilecache0[y, x] = utils.color_code(byte1, byte2, 7 - x)
        self._tilecache0_state[t] = 1

    def update_tilecache1(self, lcd, t, bank):
        if False:
            i = 10
            return i + 15
        if self._tilecache1_state[t]:
            return
        if bank:
            vram_bank = lcd.VRAM1
        else:
            vram_bank = lcd.VRAM0
        for k in range(0, 16, 2):
            byte1 = vram_bank[t * 16 + k]
            byte2 = vram_bank[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                self._tilecache1[y, x] = utils.color_code(byte1, byte2, 7 - x)
        self._tilecache1_state[t] = 1

    def update_spritecache0(self, lcd, t, bank):
        if False:
            while True:
                i = 10
        if self._spritecache0_state[t]:
            return
        if bank:
            vram_bank = lcd.VRAM1
        else:
            vram_bank = lcd.VRAM0
        for k in range(0, 16, 2):
            byte1 = vram_bank[t * 16 + k]
            byte2 = vram_bank[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                self._spritecache0[y, x] = utils.color_code(byte1, byte2, 7 - x)
        self._spritecache0_state[t] = 1

    def update_spritecache1(self, lcd, t, bank):
        if False:
            i = 10
            return i + 15
        if self._spritecache1_state[t]:
            return
        if bank:
            vram_bank = lcd.VRAM1
        else:
            vram_bank = lcd.VRAM0
        for k in range(0, 16, 2):
            byte1 = vram_bank[t * 16 + k]
            byte2 = vram_bank[t * 16 + k + 1]
            y = (t * 16 + k) // 2
            for x in range(8):
                self._spritecache1[y, x] = utils.color_code(byte1, byte2, 7 - x)
        self._spritecache1_state[t] = 1

class VBKregister:

    def __init__(self, value=0):
        if False:
            while True:
                i = 10
        self.active_bank = value

    def set(self, value):
        if False:
            print('Hello World!')
        bank = value & 1
        self.active_bank = bank

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.active_bank | 254

class PaletteIndexRegister:

    def __init__(self, val=0):
        if False:
            print('Hello World!')
        self.value = val
        self.auto_inc = 0
        self.index = 0
        self.hl = 0

    def set(self, val):
        if False:
            return 10
        if self.value == val:
            return
        self.value = val
        self.hl = val & 1
        self.index = val >> 1 & 31
        self.auto_inc = val >> 7 & 1

    def get(self):
        if False:
            return 10
        return self.value

    def getindex(self):
        if False:
            return 10
        return self.index

    def shouldincrement(self):
        if False:
            while True:
                i = 10
        if self.auto_inc:
            new_val = 128 | self.value + 1
            self.set(new_val)

    def save_state(self, f):
        if False:
            i = 10
            return i + 15
        f.write(self.value)
        f.write(self.auto_inc)
        f.write(self.index)
        f.write(self.hl)

    def load_state(self, f, state_version):
        if False:
            return 10
        self.value = f.read()
        self.auto_inc = f.read()
        self.index = f.read()
        self.hl = f.read()
CGB_NUM_PALETTES = 8

class PaletteColorRegister:

    def __init__(self, i_reg):
        if False:
            print('Hello World!')
        self.palette_mem = array('I', [65535] * CGB_NUM_PALETTES * 4)
        self.palette_mem_rgb = array('L', [0] * CGB_NUM_PALETTES * 4)
        self.index_reg = i_reg
        for n in range(0, len(self.palette_mem), 4):
            c = [7399, 7705, 32305, 8571]
            for m in range(4):
                self.palette_mem[n + m] = c[m]
                self.palette_mem_rgb[n + m] = self.cgb_to_rgb(c[m], m)

    def cgb_to_rgb(self, cgb_color, index):
        if False:
            return 10
        red = (cgb_color & 31) << 3
        green = (cgb_color >> 5 & 31) << 3
        blue = (cgb_color >> 10 & 31) << 3
        rgb_color = (red << 16 | green << 8 | blue) << 8
        if index % 4 == 0:
            rgb_color |= COL0_FLAG
        return rgb_color

    def set(self, val):
        if False:
            while True:
                i = 10
        i_val = self.palette_mem[self.index_reg.getindex()]
        if self.index_reg.hl:
            self.palette_mem[self.index_reg.getindex()] = i_val & 255 | val << 8
        else:
            self.palette_mem[self.index_reg.getindex()] = i_val & 65280 | val
        cgb_color = self.palette_mem[self.index_reg.getindex()] & 32767
        self.palette_mem_rgb[self.index_reg.getindex()] = self.cgb_to_rgb(cgb_color, self.index_reg.getindex())
        self.index_reg.shouldincrement()

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.palette_mem[self.index_reg.getindex()]

    def getcolor(self, paletteindex, colorindex):
        if False:
            while True:
                i = 10
        return self.palette_mem_rgb[paletteindex * 4 + colorindex]

    def save_state(self, f):
        if False:
            while True:
                i = 10
        for n in range(CGB_NUM_PALETTES * 4):
            f.write_16bit(self.palette_mem[n])

    def load_state(self, f, state_version):
        if False:
            for i in range(10):
                print('nop')
        for n in range(CGB_NUM_PALETTES * 4):
            self.palette_mem[n] = f.read_16bit()
            self.palette_mem_rgb[n] = self.cgb_to_rgb(self.palette_mem[n], n % 4)