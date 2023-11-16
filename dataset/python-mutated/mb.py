import logging
from pyboy.core.opcodes import CPU_COMMANDS
from pyboy.utils import STATE_VERSION
from . import bootrom, cartridge, cpu, interaction, lcd, ram, sound, timer
(INTR_VBLANK, INTR_LCDC, INTR_TIMER, INTR_SERIAL, INTR_HIGHTOLOW) = [1 << x for x in range(5)]
logger = logging.getLogger(__name__)

class Motherboard:

    def __init__(self, gamerom_file, bootrom_file, color_palette, cgb_color_palette, disable_renderer, sound_enabled, sound_emulated, cgb, randomize=False):
        if False:
            i = 10
            return i + 15
        if bootrom_file is not None:
            logger.info('Boot-ROM file provided')
        self.cartridge = cartridge.load_cartridge(gamerom_file)
        if cgb is None:
            cgb = self.cartridge.cgb
            logger.debug(f"Cartridge type auto-detected to {('CGB' if cgb else 'DMG')}")
        self.timer = timer.Timer()
        self.interaction = interaction.Interaction()
        self.bootrom = bootrom.BootROM(bootrom_file, cgb)
        self.ram = ram.RAM(cgb, randomize=randomize)
        self.cpu = cpu.CPU(self)
        if cgb:
            self.lcd = lcd.CGBLCD(cgb, self.cartridge.cgb, disable_renderer, color_palette, cgb_color_palette, randomize=randomize)
        else:
            self.lcd = lcd.LCD(cgb, self.cartridge.cgb, disable_renderer, color_palette, cgb_color_palette, randomize=randomize)
        sound_emulated |= self.cartridge.gamename == 'ZELDA DIN'
        self.sound = sound.Sound(sound_enabled, sound_emulated)
        self.key1 = 0
        self.double_speed = False
        self.cgb = cgb
        if self.cgb:
            self.hdma = HDMA()
        self.bootrom_enabled = True
        self.serialbuffer = [0] * 1024
        self.serialbuffer_count = 0
        self.breakpoints_enabled = False
        self.breakpoints_list = []
        self.breakpoint_latch = 0

    def switch_speed(self):
        if False:
            while True:
                i = 10
        bit0 = self.key1 & 1
        if bit0 == 1:
            self.double_speed = not self.double_speed
            self.lcd.double_speed = self.double_speed
            self.key1 ^= 129

    def add_breakpoint(self, bank, addr):
        if False:
            return 10
        self.breakpoints_enabled = True
        self.breakpoints_list.append((bank, addr))

    def remove_breakpoint(self, index):
        if False:
            print('Hello World!')
        self.breakpoints_list.pop(index)
        if self.breakpoints == []:
            self.breakpoints_enabled = False

    def getserial(self):
        if False:
            return 10
        b = ''.join([chr(x) for x in self.serialbuffer[:self.serialbuffer_count]])
        self.serialbuffer_count = 0
        return b

    def buttonevent(self, key):
        if False:
            i = 10
            return i + 15
        if self.interaction.key_event(key):
            self.cpu.set_interruptflag(INTR_HIGHTOLOW)

    def stop(self, save):
        if False:
            for i in range(10):
                print('nop')
        self.sound.stop()
        if save:
            self.cartridge.stop()

    def save_state(self, f):
        if False:
            while True:
                i = 10
        logger.debug('Saving state...')
        f.write(STATE_VERSION)
        f.write(self.bootrom_enabled)
        f.write(self.key1)
        f.write(self.double_speed)
        f.write(self.cgb)
        if self.cgb:
            self.hdma.save_state(f)
        self.cpu.save_state(f)
        self.lcd.save_state(f)
        self.sound.save_state(f)
        self.lcd.renderer.save_state(f)
        self.ram.save_state(f)
        self.timer.save_state(f)
        self.cartridge.save_state(f)
        self.interaction.save_state(f)
        f.flush()
        logger.debug('State saved.')

    def load_state(self, f):
        if False:
            print('Hello World!')
        logger.debug('Loading state...')
        state_version = f.read()
        if state_version >= 2:
            logger.debug(f'State version: {state_version}')
            self.bootrom_enabled = f.read()
        else:
            logger.debug(f'State version: 0-1')
            self.bootrom_enabled = state_version
        if state_version >= 8:
            self.key1 = f.read()
            self.double_speed = f.read()
            _cgb = f.read()
            if self.cgb != _cgb:
                logger.critical(f'Loading state which is not CGB, but PyBoy is loaded in CGB mode!')
                return
            self.cgb = _cgb
            if self.cgb:
                self.hdma.load_state(f, state_version)
        self.cpu.load_state(f, state_version)
        self.lcd.load_state(f, state_version)
        if state_version >= 8:
            self.sound.load_state(f, state_version)
        self.lcd.renderer.load_state(f, state_version)
        self.lcd.renderer.clear_cache()
        self.ram.load_state(f, state_version)
        if state_version < 5:
            self.cpu.interrupts_enabled_register = f.read()
        if state_version >= 5:
            self.timer.load_state(f, state_version)
        self.cartridge.load_state(f, state_version)
        self.interaction.load_state(f, state_version)
        f.flush()
        logger.debug('State loaded.')

    def breakpoint_reached(self):
        if False:
            i = 10
            return i + 15
        if self.breakpoint_latch > 0:
            self.breakpoint_latch -= 1
            return True
        for (bank, pc) in self.breakpoints_list:
            if self.cpu.PC == pc and (pc < 16384 and bank == 0 and (not self.bootrom_enabled) or (16384 <= pc < 32768 and self.cartridge.rombank_selected == bank) or (40960 <= pc < 49152 and self.cartridge.rambank_selected == bank) or (49152 <= pc <= 65535 and bank == -1) or (pc < 256 and bank == -1 and self.bootrom_enabled)):
                return True
        return False

    def processing_frame(self):
        if False:
            while True:
                i = 10
        b = not self.lcd.frame_done
        self.lcd.frame_done = False
        return b

    def tick(self):
        if False:
            for i in range(10):
                print('nop')
        while self.processing_frame():
            if self.cgb and self.hdma.transfer_active and (self.lcd._STAT._mode & 3 == 0):
                cycles = self.hdma.tick(self)
            else:
                cycles = self.cpu.tick()
            if self.cpu.halted:
                mode0_cycles = 1 << 32
                if self.cgb and self.hdma.transfer_active:
                    mode0_cycles = self.lcd.cycles_to_mode0()
                cycles = min(self.lcd.cycles_to_interrupt(), self.timer.cycles_to_interrupt(), mode0_cycles)
            sclock = self.sound.clock
            if self.cgb and self.double_speed:
                self.sound.clock = sclock + cycles // 2
            else:
                self.sound.clock = sclock + cycles
            if self.timer.tick(cycles):
                self.cpu.set_interruptflag(INTR_TIMER)
            lcd_interrupt = self.lcd.tick(cycles)
            if lcd_interrupt:
                self.cpu.set_interruptflag(lcd_interrupt)
            escape_halt = self.cpu.halted and self.breakpoint_latch == 1
            if self.breakpoints_enabled and (not escape_halt) and self.breakpoint_reached():
                return True
        self.sound.sync()
        return False

    def getitem(self, i):
        if False:
            return 10
        if 0 <= i < 16384:
            if self.bootrom_enabled and (i <= 255 or (self.cgb and 512 <= i < 2304)):
                return self.bootrom.getitem(i)
            else:
                return self.cartridge.getitem(i)
        elif 16384 <= i < 32768:
            return self.cartridge.getitem(i)
        elif 32768 <= i < 40960:
            if not self.cgb or self.lcd.vbk.active_bank == 0:
                return self.lcd.VRAM0[i - 32768]
            else:
                return self.lcd.VRAM1[i - 32768]
        elif 40960 <= i < 49152:
            return self.cartridge.getitem(i)
        elif 49152 <= i < 57344:
            bank_offset = 0
            if self.cgb and 53248 <= i:
                bank = self.getitem(65392)
                bank &= 7
                if bank == 0:
                    bank = 1
                bank_offset = (bank - 1) * 4096
            return self.ram.internal_ram0[i - 49152 + bank_offset]
        elif 57344 <= i < 65024:
            return self.getitem(i - 8192)
        elif 65024 <= i < 65184:
            return self.lcd.OAM[i - 65024]
        elif 65184 <= i < 65280:
            return self.ram.non_io_internal_ram0[i - 65184]
        elif 65280 <= i < 65356:
            if i == 65284:
                return self.timer.DIV
            elif i == 65285:
                return self.timer.TIMA
            elif i == 65286:
                return self.timer.TMA
            elif i == 65287:
                return self.timer.TAC
            elif i == 65295:
                return self.cpu.interrupts_flag_register
            elif 65296 <= i < 65344:
                return self.sound.get(i - 65296)
            elif i == 65344:
                return self.lcd.get_lcdc()
            elif i == 65345:
                return self.lcd.get_stat()
            elif i == 65346:
                return self.lcd.SCY
            elif i == 65347:
                return self.lcd.SCX
            elif i == 65348:
                return self.lcd.LY
            elif i == 65349:
                return self.lcd.LYC
            elif i == 65350:
                return 0
            elif i == 65351:
                return self.lcd.BGP.get()
            elif i == 65352:
                return self.lcd.OBP0.get()
            elif i == 65353:
                return self.lcd.OBP1.get()
            elif i == 65354:
                return self.lcd.WY
            elif i == 65355:
                return self.lcd.WX
            else:
                return self.ram.io_ports[i - 65280]
        elif 65356 <= i < 65408:
            if self.cgb and i == 65357:
                return self.key1
            elif self.cgb and i == 65359:
                return self.lcd.vbk.get()
            elif self.cgb and i == 65384:
                return self.lcd.bcps.get() | 64
            elif self.cgb and i == 65385:
                return self.lcd.bcpd.get()
            elif self.cgb and i == 65386:
                return self.lcd.ocps.get() | 64
            elif self.cgb and i == 65387:
                return self.lcd.ocpd.get()
            elif self.cgb and i == 65361:
                return 0
            elif self.cgb and i == 65362:
                return 0
            elif self.cgb and i == 65363:
                return 0
            elif self.cgb and i == 65364:
                return 0
            elif self.cgb and i == 65365:
                return self.hdma.hdma5 & 255
            return self.ram.non_io_internal_ram1[i - 65356]
        elif 65408 <= i < 65535:
            return self.ram.internal_ram1[i - 65408]
        elif i == 65535:
            return self.cpu.interrupts_enabled_register

    def setitem(self, i, value):
        if False:
            for i in range(10):
                print('nop')
        if 0 <= i < 16384:
            self.cartridge.setitem(i, value)
        elif 16384 <= i < 32768:
            self.cartridge.setitem(i, value)
        elif 32768 <= i < 40960:
            if not self.cgb or self.lcd.vbk.active_bank == 0:
                self.lcd.VRAM0[i - 32768] = value
                if i < 38912:
                    self.lcd.renderer.invalidate_tile(((i & 65520) - 32768) // 16, 0)
            else:
                self.lcd.VRAM1[i - 32768] = value
                if i < 38912:
                    self.lcd.renderer.invalidate_tile(((i & 65520) - 32768) // 16, 1)
        elif 40960 <= i < 49152:
            self.cartridge.setitem(i, value)
        elif 49152 <= i < 57344:
            bank_offset = 0
            if self.cgb and 53248 <= i:
                bank = self.getitem(65392)
                bank &= 7
                if bank == 0:
                    bank = 1
                bank_offset = (bank - 1) * 4096
            self.ram.internal_ram0[i - 49152 + bank_offset] = value
        elif 57344 <= i < 65024:
            self.setitem(i - 8192, value)
        elif 65024 <= i < 65184:
            self.lcd.OAM[i - 65024] = value
        elif 65184 <= i < 65280:
            self.ram.non_io_internal_ram0[i - 65184] = value
        elif 65280 <= i < 65356:
            if i == 65280:
                self.ram.io_ports[i - 65280] = self.interaction.pull(value)
            elif i == 65281:
                self.serialbuffer[self.serialbuffer_count] = value
                self.serialbuffer_count += 1
                self.serialbuffer_count &= 1023
                self.ram.io_ports[i - 65280] = value
            elif i == 65284:
                self.timer.reset()
            elif i == 65285:
                self.timer.TIMA = value
            elif i == 65286:
                self.timer.TMA = value
            elif i == 65287:
                self.timer.TAC = value & 7
            elif i == 65295:
                self.cpu.interrupts_flag_register = value
            elif 65296 <= i < 65344:
                self.sound.set(i - 65296, value)
            elif i == 65344:
                self.lcd.set_lcdc(value)
            elif i == 65345:
                self.lcd.set_stat(value)
            elif i == 65346:
                self.lcd.SCY = value
            elif i == 65347:
                self.lcd.SCX = value
            elif i == 65348:
                self.lcd.LY = value
            elif i == 65349:
                self.lcd.LYC = value
            elif i == 65350:
                self.transfer_DMA(value)
            elif i == 65351:
                if self.lcd.BGP.set(value):
                    self.lcd.renderer.clear_tilecache0()
            elif i == 65352:
                if self.lcd.OBP0.set(value):
                    self.lcd.renderer.clear_spritecache0()
            elif i == 65353:
                if self.lcd.OBP1.set(value):
                    self.lcd.renderer.clear_spritecache1()
            elif i == 65354:
                self.lcd.WY = value
            elif i == 65355:
                self.lcd.WX = value
            else:
                self.ram.io_ports[i - 65280] = value
        elif 65356 <= i < 65408:
            if self.bootrom_enabled and i == 65360 and (value == 1 or value == 17):
                self.bootrom_enabled = False
            elif self.cgb and i == 65357:
                self.key1 = value
            elif self.cgb and i == 65359:
                self.lcd.vbk.set(value)
            elif self.cgb and i == 65361:
                self.hdma.hdma1 = value
            elif self.cgb and i == 65362:
                self.hdma.hdma2 = value
            elif self.cgb and i == 65363:
                self.hdma.hdma3 = value
            elif self.cgb and i == 65364:
                self.hdma.hdma4 = value
            elif self.cgb and i == 65365:
                self.hdma.set_hdma5(value, self)
            elif self.cgb and i == 65384:
                self.lcd.bcps.set(value)
            elif self.cgb and i == 65385:
                self.lcd.bcpd.set(value)
                self.lcd.renderer.clear_tilecache0()
                self.lcd.renderer.clear_tilecache1()
            elif self.cgb and i == 65386:
                self.lcd.ocps.set(value)
            elif self.cgb and i == 65387:
                self.lcd.ocpd.set(value)
                self.lcd.renderer.clear_spritecache0()
                self.lcd.renderer.clear_spritecache1()
            else:
                self.ram.non_io_internal_ram1[i - 65356] = value
        elif 65408 <= i < 65535:
            self.ram.internal_ram1[i - 65408] = value
        elif i == 65535:
            self.cpu.interrupts_enabled_register = value

    def transfer_DMA(self, src):
        if False:
            for i in range(10):
                print('nop')
        dst = 65024
        offset = src * 256
        for n in range(160):
            self.setitem(dst + n, self.getitem(n + offset))

class HDMA:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.hdma1 = 0
        self.hdma2 = 0
        self.hdma3 = 0
        self.hdma4 = 0
        self.hdma5 = 255
        self.transfer_active = False
        self.curr_src = 0
        self.curr_dst = 0

    def save_state(self, f):
        if False:
            while True:
                i = 10
        f.write(self.hdma1)
        f.write(self.hdma2)
        f.write(self.hdma3)
        f.write(self.hdma4)
        f.write(self.hdma5)
        f.write(self.transfer_active)
        f.write_16bit(self.curr_src)
        f.write_16bit(self.curr_dst)

    def load_state(self, f, state_version):
        if False:
            return 10
        self.hdma1 = f.read()
        self.hdma2 = f.read()
        self.hdma3 = f.read()
        self.hdma4 = f.read()
        self.hdma5 = f.read()
        if STATE_VERSION <= 8:
            f.read()
        self.transfer_active = f.read()
        self.curr_src = f.read_16bit()
        self.curr_dst = f.read_16bit()

    def set_hdma5(self, value, mb):
        if False:
            i = 10
            return i + 15
        if self.transfer_active:
            bit7 = value & 128
            if bit7 == 0:
                self.transfer_active = False
                self.hdma5 = self.hdma5 & 127 | 128
            else:
                self.hdma5 = value & 127
        else:
            self.hdma5 = value & 255
            bytes_to_transfer = (value & 127) * 16 + 16
            src = self.hdma1 << 8 | self.hdma2 & 240
            dst = (self.hdma3 & 31) << 8 | self.hdma4 & 240
            dst |= 32768
            transfer_type = value >> 7
            if transfer_type == 0:
                for i in range(bytes_to_transfer):
                    mb.setitem(dst + i & 65535, mb.getitem(src + i & 65535))
                self.hdma5 = 255
                self.hdma4 = 255
                self.hdma3 = 255
                self.hdma2 = 255
                self.hdma1 = 255
            else:
                self.hdma5 = self.hdma5 & 127
                self.transfer_active = True
                self.curr_dst = dst
                self.curr_src = src

    def tick(self, mb):
        if False:
            while True:
                i = 10
        src = self.curr_src & 65520
        dst = self.curr_dst & 8176 | 32768
        for i in range(16):
            mb.setitem(dst + i, mb.getitem(src + i))
        self.curr_dst += 16
        self.curr_src += 16
        if self.curr_dst == 40960:
            self.curr_dst = 32768
        if self.curr_src == 32768:
            self.curr_src = 40960
        self.hdma1 = (self.curr_src & 65280) >> 8
        self.hdma2 = self.curr_src & 255
        self.hdma3 = (self.curr_dst & 65280) >> 8
        self.hdma4 = self.curr_dst & 255
        if self.hdma5 == 0:
            self.transfer_active = False
            self.hdma5 = 255
        else:
            self.hdma5 -= 1
        return 206