import os
from flashbdev import bdev

def check_bootsec():
    if False:
        while True:
            i = 10
    buf = bytearray(bdev.ioctl(5, 0))
    bdev.readblocks(0, buf)
    empty = True
    for b in buf:
        if b != 255:
            empty = False
            break
    if empty:
        return True
    fs_corrupted()

def fs_corrupted():
    if False:
        while True:
            i = 10
    import time
    import micropython
    micropython.kbd_intr(3)
    while 1:
        print('The filesystem appears to be corrupted. If you had important data there, you\nmay want to make a flash snapshot to try to recover it. Otherwise, perform\nfactory reprogramming of MicroPython firmware (completely erase flash, followed\nby firmware programming).\n')
        time.sleep(3)

def setup():
    if False:
        while True:
            i = 10
    check_bootsec()
    print('Performing initial setup')
    if bdev.info()[4] == 'vfs':
        os.VfsLfs2.mkfs(bdev)
        vfs = os.VfsLfs2(bdev)
    elif bdev.info()[4] == 'ffat':
        os.VfsFat.mkfs(bdev)
        vfs = os.VfsFat(bdev)
    os.mount(vfs, '/')
    with open('boot.py', 'w') as f:
        f.write('# This file is executed on every boot (including wake-boot from deepsleep)\n#import esp\n#esp.osdebug(None)\n#import webrepl\n#webrepl.start()\n')
    return vfs