import sys
import lief
from lief import ELF

def remove_section_table(filename, output):
    if False:
        for i in range(10):
            print('nop')
    binary = lief.parse(filename)
    header = binary.header
    header.section_header_offset = 0
    header.numberof_sections = 0
    binary.write(output)

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) != 3:
        print('Usage: {} <elf binary> <output>'.format(sys.argv[0]))
        sys.exit(1)
    remove_section_table(sys.argv[1], sys.argv[2])
if __name__ == '__main__':
    main()