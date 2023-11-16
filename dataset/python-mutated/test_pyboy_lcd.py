import os
import platform
import pytest
from pyboy.core.lcd import LCD
is_pypy = platform.python_implementation() == 'PyPy'
(INTR_VBLANK, INTR_LCDC, INTR_TIMER, INTR_SERIAL, INTR_HIGHTOLOW) = [1 << x for x in range(5)]
color_palette = (16777215, 10066329, 5592405, 0)
cgb_color_palette = ((16777215, 8126257, 25541, 0), (16777215, 16745604, 16745604, 0), (16777215, 16745604, 16745604, 0))

@pytest.mark.skipif(not hasattr(LCD, 'get_stat'), reason='This test requires access to internal registers not available in Cython')
class TestLCD:

    def test_set_stat_mode(self):
        if False:
            for i in range(10):
                print('nop')
        lcd = LCD(False, False, False, color_palette, cgb_color_palette)
        lcd._STAT._mode = 2
        assert lcd._STAT._mode == 2
        assert lcd._STAT.set_mode(2) == 0
        lcd._STAT._mode = 0
        assert lcd._STAT.set_mode(1) == 0
        lcd._STAT._mode = 0
        lcd.set_stat(1 << 1 + 3)
        assert lcd._STAT.set_mode(1) == INTR_LCDC

    def test_stat_register(self):
        if False:
            return 10
        lcd = LCD(False, False, False, color_palette, cgb_color_palette)
        lcd.set_lcdc(128)
        lcd._STAT.value &= 248
        lcd.set_stat(127)
        assert lcd.get_stat() & 128 == 128
        assert lcd.get_stat() & 7 == 0
        assert lcd.get_stat() & 3 == 0
        lcd._STAT.set_mode(2)
        assert lcd.get_stat() & 3 == 2

    def test_check_lyc(self):
        if False:
            i = 10
            return i + 15
        lcd = LCD(False, False, False, color_palette, cgb_color_palette)
        lcd.LYC = 0
        lcd.LY = 0
        assert not lcd.get_stat() & 4
        assert lcd._STAT.update_LYC(lcd.LYC, lcd.LY) == 0
        assert lcd.get_stat() & 4
        lcd.LYC = 0
        lcd.LY = 1
        assert lcd._STAT.update_LYC(lcd.LYC, lcd.LY) == 0
        assert not lcd.get_stat() & 4
        lcd.LYC = 0
        lcd.LY = 0
        lcd.set_stat(64)
        assert not lcd.get_stat() & 4
        assert lcd._STAT.update_LYC(lcd.LYC, lcd.LY) == INTR_LCDC
        assert lcd._STAT.update_LYC(lcd.LYC, lcd.LY) == INTR_LCDC
        assert lcd.get_stat() & 4