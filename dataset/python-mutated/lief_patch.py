import sys
from lief import ELF
import distorm3

def remove_anti_debug(binary):
    if False:
        for i in range(10):
            print('nop')
    patch = [131, 248, 255, 144, 144]
    ep = binary.header.entrypoint
    text_section = binary.section_from_virtual_address(ep)
    code = ''.join(map(chr, text_section.content))
    iterable = distorm3.DecodeGenerator(text_section.virtual_address, code, distorm3.Decode32Bits)
    for (offset, size, instruction, hexdump) in iterable:
        if 'CMP EAX, 0x3000' in instruction:
            binary.patch_address(offset, patch)
            print('[PATCH] %.8x: %-32s %s' % (offset, hexdump, instruction))
    binary.patch_address(134517611, patch)

def crack_it(binary):
    if False:
        while True:
            i = 10
    patch1 = [49, 210]
    patch2 = [49, 192]
    binary.patch_address(134517894, patch1)
    binary.patch_address(134517896, patch2)

def main(argv):
    if False:
        while True:
            i = 10
    binary = ELF.parse('./KeygenMe')
    remove_anti_debug(binary)
    crack_it(binary)
    binary.write('./KeygenMe.crack')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))