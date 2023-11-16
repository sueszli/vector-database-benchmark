from lief import ELF
import sys

def main():
    if False:
        return 10
    binary = ELF.parse(sys.argv[1])
    symtab_section = ELF.Section()
    symtab_section.name = ''
    symtab_section.type = ELF.SECTION_TYPES.SYMTAB
    symtab_section.entry_size = 24
    symtab_section.alignment = 8
    symtab_section.link = len(binary.sections) + 1
    symtab_section.content = [0] * 100
    symstr_section = ELF.Section()
    symstr_section.name = ''
    symstr_section.type = ELF.SECTION_TYPES.STRTAB
    symstr_section.entry_size = 1
    symstr_section.alignment = 1
    symstr_section.content = [0] * 100
    symtab_section = binary.add(symtab_section, loaded=False)
    symstr_section = binary.add(symstr_section, loaded=False)
    symbol = ELF.Symbol()
    symbol.name = ''
    symbol.type = ELF.SYMBOL_TYPES.NOTYPE
    symbol.value = 0
    symbol.binding = ELF.SYMBOL_BINDINGS.LOCAL
    symbol.size = 0
    symbol.shndx = 0
    symbol = binary.add_static_symbol(symbol)
    symbol = ELF.Symbol()
    symbol.name = 'main'
    symbol.type = ELF.SYMBOL_TYPES.FUNC
    symbol.value = 4205056
    symbol.binding = ELF.SYMBOL_BINDINGS.LOCAL
    symbol.shndx = 14
    symbol = binary.add_static_symbol(symbol)
    print(symbol)
    binary.write(sys.argv[2])
if __name__ == '__main__':
    main()