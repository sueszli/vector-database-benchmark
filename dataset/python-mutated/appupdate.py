from micropython import const
import struct, machine, fwupdate, spiflash, pyb
_IOCTL_BLOCK_COUNT = const(4)
_IOCTL_BLOCK_SIZE = const(5)
_SPIFLASH_UPDATE_KEY_ADDR = const(1020 * 1024)
_SPIFLASH_UPDATE_KEY_VALUE = const(305419896)
_FILESYSTEM_ADDR = const(2147483648 + 1024 * 1024)
flash = pyb.Flash(start=0)
_FILESYSTEM_LEN = flash.ioctl(_IOCTL_BLOCK_COUNT, None) * flash.ioctl(_IOCTL_BLOCK_SIZE, None)

def update_app(filename):
    if False:
        while True:
            i = 10
    print(f'Updating application firmware from {filename}')
    elems = fwupdate.update_app_elements(filename, _FILESYSTEM_ADDR, _FILESYSTEM_LEN)
    if not elems:
        return
    key = struct.pack('<I', _SPIFLASH_UPDATE_KEY_VALUE)
    spi = machine.SoftSPI(sck=machine.Pin.board.FLASH_SCK, mosi=machine.Pin.board.FLASH_MOSI, miso=machine.Pin.board.FLASH_MISO, baudrate=50000000)
    cs = machine.Pin(machine.Pin.board.FLASH_NSS, machine.Pin.OUT, value=1)
    flash = spiflash.SPIFlash(spi, cs)
    flash.erase_block(_SPIFLASH_UPDATE_KEY_ADDR)
    flash.write(_SPIFLASH_UPDATE_KEY_ADDR, key + elems)
    machine.bootloader(elems)