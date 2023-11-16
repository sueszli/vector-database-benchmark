"""
Driver for accelerometer on STM32F4 Discover board.

Sets accelerometer range at +-2g.
Returns list containing X,Y,Z axis acceleration values in 'g' units (9.8m/s^2).

See:
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/Components/lis302dl/lis302dl.h
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/Components/lis302dl/lis302dl.c
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.c
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/STM32F4-Discovery/stm32f4_discovery.h
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/STM32F4-Discovery/stm32f4_discovery_accelerometer.c
    STM32Cube_FW_F4_V1.1.0/Drivers/BSP/STM32F4-Discovery/stm32f4_discovery_accelerometer.h
    STM32Cube_FW_F4_V1.1.0/Projects/STM32F4-Discovery/Demonstrations/Src/main.c
"""
from micropython import const
from pyb import Pin
from pyb import SPI
READWRITE_CMD = const(128)
MULTIPLEBYTE_CMD = const(64)
WHO_AM_I_ADDR = const(15)
OUT_X_ADDR = const(41)
OUT_Y_ADDR = const(43)
OUT_Z_ADDR = const(45)
OUT_T_ADDR = const(12)
LIS302DL_WHO_AM_I_VAL = const(59)
LIS302DL_CTRL_REG1_ADDR = const(32)
LIS302DL_CONF = const(71)
LIS3DSH_WHO_AM_I_VAL = const(63)
LIS3DSH_CTRL_REG4_ADDR = const(32)
LIS3DSH_CTRL_REG5_ADDR = const(36)
LIS3DSH_CTRL_REG4_CONF = const(103)
LIS3DSH_CTRL_REG5_CONF = const(0)

class STAccel:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.cs_pin = Pin('PE3', Pin.OUT_PP, Pin.PULL_NONE)
        self.cs_pin.high()
        self.spi = SPI(1, SPI.MASTER, baudrate=328125, polarity=0, phase=1, bits=8)
        self.who_am_i = self.read_id()
        if self.who_am_i == LIS302DL_WHO_AM_I_VAL:
            self.write_bytes(LIS302DL_CTRL_REG1_ADDR, bytearray([LIS302DL_CONF]))
            self.sensitivity = 18
        elif self.who_am_i == LIS3DSH_WHO_AM_I_VAL:
            self.write_bytes(LIS3DSH_CTRL_REG4_ADDR, bytearray([LIS3DSH_CTRL_REG4_CONF]))
            self.write_bytes(LIS3DSH_CTRL_REG5_ADDR, bytearray([LIS3DSH_CTRL_REG5_CONF]))
            self.sensitivity = 0.06 * 256
        else:
            raise Exception('LIS302DL or LIS3DSH accelerometer not present')

    def convert_raw_to_g(self, x):
        if False:
            print('Hello World!')
        if x & 128:
            x = x - 256
        return x * self.sensitivity / 1000

    def read_bytes(self, addr, nbytes):
        if False:
            print('Hello World!')
        if nbytes > 1:
            addr |= READWRITE_CMD | MULTIPLEBYTE_CMD
        else:
            addr |= READWRITE_CMD
        self.cs_pin.low()
        self.spi.send(addr)
        buf = self.spi.recv(nbytes)
        self.cs_pin.high()
        return buf

    def write_bytes(self, addr, buf):
        if False:
            print('Hello World!')
        if len(buf) > 1:
            addr |= MULTIPLEBYTE_CMD
        self.cs_pin.low()
        self.spi.send(addr)
        for b in buf:
            self.spi.send(b)
        self.cs_pin.high()

    def read_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.read_bytes(WHO_AM_I_ADDR, 1)[0]

    def x(self):
        if False:
            for i in range(10):
                print('nop')
        return self.convert_raw_to_g(self.read_bytes(OUT_X_ADDR, 1)[0])

    def y(self):
        if False:
            return 10
        return self.convert_raw_to_g(self.read_bytes(OUT_Y_ADDR, 1)[0])

    def z(self):
        if False:
            while True:
                i = 10
        return self.convert_raw_to_g(self.read_bytes(OUT_Z_ADDR, 1)[0])

    def xyz(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.x(), self.y(), self.z())