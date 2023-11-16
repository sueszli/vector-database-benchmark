from ssd1306 import SSD1306_I2C
from micropython import const
SET_COL_ADDR = const(33)
SET_PAGE_ADDR = const(34)

class SSD1306_I2C_Mod(SSD1306_I2C):

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        x0 = 0
        x1 = self.width - 1
        if self.width == 64:
            x0 += 32
            x1 += 32
        self.write_cmd(SET_COL_ADDR)
        self.write_cmd(x0)
        self.write_cmd(x1)
        self.write_cmd(SET_PAGE_ADDR)
        self.write_cmd(0)
        self.write_cmd(self.pages - 1)
        chunk_size = 254
        num_of_chunks = len(self.buffer) // chunk_size
        leftover = len(self.buffer) - num_of_chunks * chunk_size
        for i in range(0, num_of_chunks):
            self.write_data(self.buffer[chunk_size * i:chunk_size * (i + 1)])
        if leftover > 0:
            self.write_data(self.buffer[chunk_size * num_of_chunks:])

    def write_data(self, buf):
        if False:
            return 10
        buffer = bytearray([64]) + buf
        self.i2c.writeto(self.addr, buffer)