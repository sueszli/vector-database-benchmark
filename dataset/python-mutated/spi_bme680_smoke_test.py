import board
import busio
import digitalio
cs = digitalio.DigitalInOut(board.SS)
cs.direction = digitalio.Direction.OUTPUT
cs.value = True
BME680_SPI_REGISTER = 115
BME680_CHIPID_REGISTER = 208
BME680_CHIPID = 97
SPI_HERZ = 5242880
spi = busio.SPI(board.SCK, MISO=board.MISO, MOSI=board.MOSI)

def readByte(addr):
    if False:
        i = 10
        return i + 15
    value = -1
    while not spi.try_lock():
        pass
    try:
        spi.configure(baudrate=500000, phase=0, polarity=0)
        cs.value = False
        result = bytearray(1)
        result[0] = addr | 128
        spi.write(result)
        spi.readinto(result)
        value = result[0]
        return value
    finally:
        spi.unlock()
        cs.value = True

def writeByte(addr, value):
    if False:
        print('Hello World!')
    while not spi.try_lock():
        pass
    try:
        spi.configure(baudrate=500000, phase=0, polarity=0)
        cs.value = False
        result = bytearray(2)
        result[0] = addr & ~128
        result[1] = value
        spi.write(result)
    finally:
        spi.unlock()
reg = readByte(BME680_SPI_REGISTER)
if reg & 16 != 0:
    writeByte(BME680_SPI_REGISTER, reg & ~16)
id = readByte(BME680_CHIPID_REGISTER)
print(f'id is {id}, expected {BME680_CHIPID}')