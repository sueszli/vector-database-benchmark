import machine
from esp32 import Partition
cur = Partition(Partition.RUNNING)
cur_name = cur.info()[4]
if not cur_name.startswith('ota_'):
    print('SKIP')
    raise SystemExit
DEBUG = True

def log(*args):
    if False:
        return 10
    if DEBUG:
        print(*args)
import os
try:
    os.rename('boot.py', 'boot-orig.py')
except:
    pass
with open('boot.py', 'w') as f:
    f.write('DEBUG=' + str(DEBUG))
    f.write('\nimport machine\nfrom esp32 import Partition\ncur = Partition(Partition.RUNNING)\ncur_name = cur.info()[4]\n\ndef log(*args):\n    if DEBUG: print(*args)\n\nfrom step import STEP, EXPECT\nlog("Running partition: " + cur_name + " STEP=" + str(STEP) + " EXPECT=" + EXPECT)\nif cur_name != EXPECT:\n    print("\\x04FAILED: step " + str(STEP) + " expected " + EXPECT + " got " + cur_name + "\\x04")\n\nif STEP == 0:\n    log("Not confirming boot ok and resetting back into first")\n    nxt = cur.get_next_update()\n    with open("step.py", "w") as f:\n        f.write("STEP=1\\nEXPECT=\\"" + nxt.info()[4] + "\\"\\n")\n    machine.reset()\nelif STEP == 1:\n    log("Booting into second partition again")\n    nxt = cur.get_next_update()\n    nxt.set_boot()\n    with open("step.py", "w") as f:\n        f.write("STEP=2\\nEXPECT=\\"" + nxt.info()[4] + "\\"\\n")\n    machine.reset()\nelif STEP == 2:\n    log("Confirming boot ok and rebooting into same partition")\n    Partition.mark_app_valid_cancel_rollback()\n    with open("step.py", "w") as f:\n        f.write("STEP=3\\nEXPECT=\\"" + cur_name + "\\"\\n")\n    machine.reset()\nelif STEP == 3:\n    log("Booting into original partition")\n    nxt = cur.get_next_update()\n    nxt.set_boot()\n    with open("step.py", "w") as f:\n        f.write("STEP=4\\nEXPECT=\\"" + nxt.info()[4] + "\\"\\n")\n    machine.reset()\nelif STEP == 4:\n    log("Confirming boot ok and DONE!")\n    Partition.mark_app_valid_cancel_rollback()\n    import os\n    os.remove("step.py")\n    os.remove("boot.py")\n    os.rename("boot-orig.py", "boot.py")\n    print("\\nSUCCESS!\\n\\x04\\x04")\n\n')

def copy_partition(src, dest):
    if False:
        i = 10
        return i + 15
    log('Partition copy: {} --> {}'.format(src.info(), dest.info()))
    sz = src.info()[3]
    if dest.info()[3] != sz:
        raise ValueError("Sizes don't match: {} vs {}".format(sz, dest.info()[3]))
    addr = 0
    blk = bytearray(4096)
    while addr < sz:
        if sz - addr < 4096:
            blk = blk[:sz - addr]
        if addr & 65535 == 0:
            print('   ... 0x{:06x}'.format(addr))
        src.readblocks(addr >> 12, blk)
        dest.writeblocks(addr >> 12, blk)
        addr += len(blk)
print('Copying current to next partition')
nxt = cur.get_next_update()
copy_partition(cur, nxt)
print('Partition copied, booting into it')
nxt.set_boot()
with open('step.py', 'w') as f:
    f.write('STEP=0\nEXPECT="' + nxt.info()[4] + '"\n')
machine.reset()