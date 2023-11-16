
#include "MachoProgramInfoLoader.hpp"
#include <dlfcn.h>
#include <map>
#include <set>
#include <teenypath.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_images.h>
#include <mach-o/nlist.h>
#include <mach-o/reloc.h>
#include <mach-o/stab.h>
#include <mach-o/x86_64/reloc.h>
#include "jet/live/LiveContext.hpp"
#include "jet/live/Utility.hpp"

#define N_TYPE_GET_STAB(nType) static_cast<uint8_t>((nType)&N_STAB)  // NOLINT
#define N_TYPE_GET_PEXT(nType) static_cast<bool>((nType)&N_PEXT)     // NOLINT
#define N_TYPE_GET_TYPE(nType) static_cast<uint8_t>((nType)&N_TYPE)  // NOLINT
#define N_TYPE_GET_EXT(nType) static_cast<bool>((nType)&N_EXT)       // NOLINT

#define N_DESC_GET_REFERENCE_TYPE(nDesc) static_cast<uint16_t>((nDesc)&REFERENCE_TYPE)              // NOLINT
#define N_DESC_GET_REFERENCED_DYNAMICALLY(nDesc) static_cast<bool>((nDesc)&REFERENCED_DYNAMICALLY)  // NOLINT
#define N_DESC_GET_DISCARDED(nDesc) static_cast<bool>((nDesc)&N_DESC_DISCARDED)                     // NOLINT
#define N_DESC_GET_WEAK_DEF(nDesc) static_cast<bool>((nDesc)&N_WEAK_DEF)                            // NOLINT
#define N_DESC_GET_WEAK_REF(nDesc) static_cast<bool>((nDesc)&N_WEAK_REF)                            // NOLINT

namespace
{
    uintptr_t thisExecutableLoadAddress = 0;
}

namespace jet
{
    std::vector<std::string> MachoProgramInfoLoader::getAllLoadedProgramsPaths(const LiveContext* context) const
    {
        std::vector<std::string> filepaths;

        for (uint32_t i = 0; i < _dyld_image_count(); i++) {
            auto imagePath = TeenyPath::path{_dyld_get_image_name(i)};
            if (imagePath.exists()) {
                imagePath = imagePath.resolve_absolute();
                if (imagePath.string() == context->thisExecutablePath) {
                    filepaths.emplace_back("");
                } else {
                    filepaths.emplace_back(imagePath.string());
                }
            } else {
                context->events->addLog(LogSeverity::kWarning, "Image doesn't exist: " + imagePath.string());
            }
        }

        return filepaths;
    }

    Symbols MachoProgramInfoLoader::getProgramSymbols(const LiveContext* context, const std::string& filepath) const
    {
        Symbols res;
        MachoContext machoContext;

        intptr_t imageAddressSlide = 0;
        bool found = false;
        std::string realFilepath = filepath.empty() ? context->thisExecutablePath : filepath;
        for (uint32_t i = 0; i < _dyld_image_count(); i++) {
            auto imagePath = TeenyPath::path{_dyld_get_image_name(i)};
            if (imagePath.exists()) {
                imagePath = imagePath.resolve_absolute();
                if (imagePath.string() == realFilepath) {
                    imageAddressSlide = _dyld_get_image_vmaddr_slide(i);
                    found = true;
                    break;
                }
            } else {
                context->events->addLog(LogSeverity::kWarning, "Image doesn't exist: " + imagePath.string());
            }
        }
        if (!found) {
            context->events->addLog(LogSeverity::kError, "Cannot find address slide of image " + realFilepath);
            return res;
        }

        const auto baseAddress = static_cast<uintptr_t>(imageAddressSlide);

        // Parsing mach-o binary
        auto f = fopen(realFilepath.c_str(), "r");
        if (!f) {
            context->events->addLog(LogSeverity::kError, "Cannot open " + filepath);
            return res;
        }
        fseek(f, 0, SEEK_END);
        auto length = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);
        auto content = std::unique_ptr<char[]>(new char[length]);
        fread(content.get(), 1, length, f);
        fclose(f);

        auto header = reinterpret_cast<mach_header_64*>(content.get());
        if (header->magic != MH_MAGIC_64) {
            // Probably it is some system "fat" library, we're not interested in it
            // context->events->addLog(LogSeverity::kError, "Cannot read symbols, not a Mach-O 64 binary");
            return res;
        }

        uint32_t sectionIndex = 0;
        auto machoPtr = content.get();
        auto commandOffset = sizeof(mach_header_64);
        std::vector<std::set<uint64_t>> symbolsBounds;
        for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
            auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
            switch (command->cmd) {
                case LC_SEGMENT_64: {
                    auto segmentCommand = reinterpret_cast<segment_command_64*>(machoPtr + commandOffset);
                    auto sectionsPtr =
                        reinterpret_cast<struct section_64*>(machoPtr + commandOffset + sizeof(*segmentCommand));
                    for (uint32_t i = 0; i < segmentCommand->nsects; i++) {
                        auto& section = sectionsPtr[i];
                        sectionIndex++;
                        if (machoContext.sectionNames.size() <= sectionIndex) {
                            machoContext.sectionNames.resize(sectionIndex + 1);
                            symbolsBounds.resize(sectionIndex + 1);
                        }
                        machoContext.sectionNames[sectionIndex] = std::string(section.sectname);
                        symbolsBounds[sectionIndex].insert(section.addr + section.size);
                    }
                    break;
                }

                default: break;
            }

            commandOffset += command->cmdsize;
        }

        commandOffset = sizeof(mach_header_64);
        for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
            auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
            switch (command->cmd) {
                case LC_SYMTAB: {
                    auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                    auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                    for (uint32_t i = 0; i < table->nsyms; i++) {
                        if (N_TYPE_GET_TYPE(symbolsPtr[i].n_type) == N_SECT) {
                            symbolsBounds[symbolsPtr[i].n_sect].insert(symbolsPtr[i].n_value);
                        }
                    }
                    break;
                }

                default: break;
            }
            commandOffset += command->cmdsize;
        }

        std::hash<std::string> stringHasher;
        uint64_t currentHash = 0;
        std::string currentFileName;
        std::unordered_map<uintptr_t, uint64_t> addressHashMap;
        std::unordered_map<uint64_t, std::string> hashNameMap;
        std::unordered_map<uint64_t, std::unordered_map<uintptr_t, MachoSymbolType>> hashAddressStabType;
        commandOffset = sizeof(mach_header_64);
        for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
            auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
            switch (command->cmd) {
                case LC_SYMTAB: {
                    auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                    auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                    auto stringTable = machoPtr + table->stroff;
                    for (uint32_t i = 0; i < table->nsyms; i++) {
                        auto& symbol = symbolsPtr[i];

                        if (N_TYPE_GET_EXT(symbol.n_type) && symbol.n_sect != NO_SECT) {
                            res.exportedSymbolNames.insert(std::string(stringTable + symbol.n_un.n_strx + 1));
                        }

                        MachoSymbol machoSymbol;
                        if (N_TYPE_GET_STAB(symbol.n_type) != 0) {
                            switch (symbol.n_type) {
                                case N_GSYM: machoSymbol.type = MachoSymbolType::kGSYM; break;
                                case N_FNAME: machoSymbol.type = MachoSymbolType::kFNAME; break;
                                case N_FUN: machoSymbol.type = MachoSymbolType::kFUN; break;
                                case N_STSYM: machoSymbol.type = MachoSymbolType::kSTSYM; break;
                                case N_LCSYM: machoSymbol.type = MachoSymbolType::kLCSYM; break;
                                case N_BNSYM: machoSymbol.type = MachoSymbolType::kBNSYM; break;
                                case N_AST: machoSymbol.type = MachoSymbolType::kAST; break;
                                case N_OPT: machoSymbol.type = MachoSymbolType::kOPT; break;
                                case N_RSYM: machoSymbol.type = MachoSymbolType::kRSYM; break;
                                case N_SLINE: machoSymbol.type = MachoSymbolType::kSLINE; break;
                                case N_ENSYM: machoSymbol.type = MachoSymbolType::kENSYM; break;
                                case N_SSYM: machoSymbol.type = MachoSymbolType::kSSYM; break;
                                case N_SO: machoSymbol.type = MachoSymbolType::kSO; break;
                                case N_OSO: machoSymbol.type = MachoSymbolType::kOSO; break;
                                case N_LSYM: machoSymbol.type = MachoSymbolType::kLSYM; break;
                                case N_BINCL: machoSymbol.type = MachoSymbolType::kBINCL; break;
                                case N_SOL: machoSymbol.type = MachoSymbolType::kSOL; break;
                                case N_PARAMS: machoSymbol.type = MachoSymbolType::kPARAMS; break;
                                case N_VERSION: machoSymbol.type = MachoSymbolType::kVERSION; break;
                                case N_OLEVEL: machoSymbol.type = MachoSymbolType::kOLEVEL; break;
                                case N_PSYM: machoSymbol.type = MachoSymbolType::kPSYM; break;
                                case N_EINCL: machoSymbol.type = MachoSymbolType::kEINCL; break;
                                case N_ENTRY: machoSymbol.type = MachoSymbolType::kENTRY; break;
                                case N_LBRAC: machoSymbol.type = MachoSymbolType::kLBRAC; break;
                                case N_EXCL: machoSymbol.type = MachoSymbolType::kEXCL; break;
                                case N_RBRAC: machoSymbol.type = MachoSymbolType::kRBRAC; break;
                                case N_BCOMM: machoSymbol.type = MachoSymbolType::kBCOMM; break;
                                case N_ECOMM: machoSymbol.type = MachoSymbolType::kECOMM; break;
                                case N_ECOML: machoSymbol.type = MachoSymbolType::kECOML; break;
                                case N_LENG: machoSymbol.type = MachoSymbolType::kLENG; break;
                                case N_PC: machoSymbol.type = MachoSymbolType::kPC; break;
                                default: continue;  // Some symbol we're not interested in
                            }
                        } else {
                            switch (N_TYPE_GET_TYPE(symbol.n_type)) {
                                case N_UNDF: machoSymbol.type = MachoSymbolType::kUndefined; break;
                                case N_ABS: machoSymbol.type = MachoSymbolType::kAbsolute; break;
                                case N_SECT: machoSymbol.type = MachoSymbolType::kSection; break;
                                case N_PBUD: machoSymbol.type = MachoSymbolType::kPreboundUndefined; break;
                                case N_INDR: machoSymbol.type = MachoSymbolType::kIndirect; break;
                                default: continue;  // Some symbol we're not interested in
                            }
                        }

                        switch (N_DESC_GET_REFERENCE_TYPE(symbol.n_desc)) {
                            case REFERENCE_FLAG_UNDEFINED_NON_LAZY:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kUndefinedNonLazy;
                                break;
                            case REFERENCE_FLAG_UNDEFINED_LAZY:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kUndefinedLazy;
                                break;
                            case REFERENCE_FLAG_DEFINED:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kDefined;
                                break;
                            case REFERENCE_FLAG_PRIVATE_DEFINED:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateDefined;
                                break;
                            case REFERENCE_FLAG_PRIVATE_UNDEFINED_NON_LAZY:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateUndefinedNonLazy;
                                break;
                            case REFERENCE_FLAG_PRIVATE_UNDEFINED_LAZY:
                                machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateUndefinedLazy;
                                break;
                        }

                        machoSymbol.referencedDynamically = N_DESC_GET_REFERENCED_DYNAMICALLY(symbol.n_desc);
                        machoSymbol.descDiscarded = N_DESC_GET_DISCARDED(symbol.n_desc);
                        machoSymbol.weakRef = N_DESC_GET_WEAK_REF(symbol.n_desc);
                        machoSymbol.weakDef = N_DESC_GET_WEAK_DEF(symbol.n_desc);
                        machoSymbol.privateExternal = N_TYPE_GET_PEXT(symbol.n_type);
                        machoSymbol.external = N_TYPE_GET_EXT(symbol.n_type);
                        machoSymbol.sectionIndex = symbol.n_sect;
                        machoSymbol.virtualAddress = symbol.n_value;
                        // All symbol names starts with '_', so just skipping 1 char
                        machoSymbol.name = stringTable + symbol.n_un.n_strx + 1;

                        if (machoSymbol.type == MachoSymbolType::kOSO) {
                            machoSymbol.name = stringTable + symbol.n_un.n_strx;
                            currentHash = stringHasher(machoSymbol.name);
                            currentFileName = machoSymbol.name;
                        } else if (machoSymbol.type == MachoSymbolType::kSTSYM) {
                            addressHashMap[machoSymbol.virtualAddress] = currentHash;
                            hashNameMap[currentHash] = currentFileName;
                            hashAddressStabType[currentHash][machoSymbol.virtualAddress] = MachoSymbolType::kSTSYM;
                        }

                        if (machoSymbol.type == MachoSymbolType::kSection) {
                            auto addrFound = symbolsBounds[machoSymbol.sectionIndex].find(machoSymbol.virtualAddress);
                            assert(addrFound != symbolsBounds[machoSymbol.sectionIndex].end());
                            addrFound++;
                            if (addrFound != symbolsBounds[machoSymbol.sectionIndex].end()) {
                                machoSymbol.size = *addrFound - machoSymbol.virtualAddress;
                            } else {
                                context->events->addLog(LogSeverity::kDebug, "wtf?");
                            }
                        }

                        Symbol sym;
                        sym.name = machoSymbol.name;
                        sym.runtimeAddress = baseAddress + machoSymbol.virtualAddress;
                        sym.size = machoSymbol.size;
                        sym.hash = machoSymbol.hash = addressHashMap[machoSymbol.virtualAddress];
                        if (hashAddressStabType[sym.hash][machoSymbol.virtualAddress] == MachoSymbolType::kSTSYM) {
                            sym.checkHash = true;
                        }

                        if (::thisExecutableLoadAddress == 0 &&
                            filepath.empty()) {
                            // Trying to distinguish real address of the executable (macOS 10.15, __JET_TEXT)
                            ::thisExecutableLoadAddress = sym.runtimeAddress;
                        }
                        if (sym.runtimeAddress >= ::thisExecutableLoadAddress &&
                            context->symbolsFilter->shouldReloadMachoSymbol(machoContext, machoSymbol)) {
                            res.functions[sym.name].push_back(sym);
                        }

                        if (context->symbolsFilter->shouldTransferMachoSymbol(machoContext, machoSymbol)) {
                            res.variables[sym.name].push_back(sym);
                        }
                    }
                    break;
                }

                default: break;
            }
            commandOffset += command->cmdsize;
        }
        return res;
    }

    std::vector<Relocation> MachoProgramInfoLoader::getLinkTimeRelocations(const LiveContext* context,
        const std::vector<std::string>& objFilePaths)
    {
        std::vector<Relocation> res;

        for (const auto& filepath : objFilePaths) {
            auto f = fopen(filepath.c_str(), "r");
            if (!f) {
                context->events->addLog(LogSeverity::kError, "Cannot open " + filepath);
                continue;
            }
            fseek(f, 0, SEEK_END);
            auto length = static_cast<size_t>(ftell(f));
            fseek(f, 0, SEEK_SET);
            auto content = std::unique_ptr<char[]>(new char[length]);
            fread(content.get(), 1, length, f);
            fclose(f);

            auto header = reinterpret_cast<mach_header_64*>(content.get());
            if (header->magic != MH_MAGIC_64) {
                // Probably it is some system "fat" library, we're not interested in it
                // context->events->addLog(LogSeverity::kError, "Cannot read symbols, not a Mach-O 64 binary");
                return res;
            }

            struct ShortMachoSymbol
            {
                std::string name;
                uint64_t hash = 0;
                int sectionIndex = 0;
                nlist_64* symPtr = nullptr;
                symtab_command* table = nullptr;
            };

            std::vector<ShortMachoSymbol> orderedSymbols;
            std::unordered_map<int64_t, int64_t> symbolsSectionIndexes;
            std::vector<std::map<uintptr_t, MachoSymbol>> symbolsInSections;
            std::vector<std::string> sectionNames;
            int textSectionIndex = -1;
            int bssSectionIndex = -1;
            int dataSectionIndex = -1;
            int commonSectionIndex = -1;
            std::hash<std::string> stringHasher;
            const uint64_t currentHash = stringHasher(filepath);
            std::vector<std::set<uint64_t>> symbolsBounds;

            uint32_t sectionIndex = 0;
            auto machoPtr = content.get();
            auto commandOffset = sizeof(mach_header_64);
            for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
                auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
                switch (command->cmd) {
                    case LC_SEGMENT_64: {
                        auto segmentCommand = reinterpret_cast<segment_command_64*>(machoPtr + commandOffset);
                        auto sectionsPtr =
                            reinterpret_cast<struct section_64*>(machoPtr + commandOffset + sizeof(*segmentCommand));
                        for (uint32_t i = 0; i < segmentCommand->nsects; i++) {
                            auto& section = sectionsPtr[i];
                            sectionIndex++;
                            if (sectionNames.size() <= sectionIndex) {
                                sectionNames.resize(sectionIndex + 1);
                                symbolsBounds.resize(sectionIndex + 1);
                                symbolsInSections.resize(sectionIndex + 1);
                            }
                            sectionNames[sectionIndex] = std::string(section.sectname);
                            symbolsBounds[sectionIndex].insert(section.addr + section.size);
                            if (section.sectname == std::string("__text")) {
                                textSectionIndex = static_cast<int>(sectionIndex);
                            } else if (section.sectname == std::string("__data")) {
                                dataSectionIndex = static_cast<int>(sectionIndex);
                            } else if (section.sectname == std::string("__bss")) {
                                bssSectionIndex = static_cast<int>(sectionIndex);
                            } else if (section.sectname == std::string("__common")) {
                                commonSectionIndex = static_cast<int>(sectionIndex);
                            }
                        }
                        break;
                    }
                    default: break;
                }
                commandOffset += command->cmdsize;
            }

            commandOffset = sizeof(mach_header_64);
            for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
                auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
                switch (command->cmd) {
                    case LC_SYMTAB: {
                        auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                        auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                        for (uint32_t i = 0; i < table->nsyms; i++) {
                            if (N_TYPE_GET_TYPE(symbolsPtr[i].n_type) == N_SECT) {
                                symbolsBounds[symbolsPtr[i].n_sect].insert(symbolsPtr[i].n_value);
                            }
                        }
                        break;
                    }
                    default: break;
                }
                commandOffset += command->cmdsize;
            }

            commandOffset = sizeof(mach_header_64);
            for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
                auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
                switch (command->cmd) {
                    case LC_SYMTAB: {
                        auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                        auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                        auto stringTable = machoPtr + table->stroff;
                        for (uint32_t i = 0; i < table->nsyms; i++) {
                            auto& symbol = symbolsPtr[i];

                            ShortMachoSymbol shortSym;
                            shortSym.symPtr = &symbolsPtr[i];
                            shortSym.table = table;
                            shortSym.name = stringTable + symbol.n_un.n_strx + 1;
                            shortSym.hash = currentHash;
                            shortSym.sectionIndex = symbol.n_sect;
                            orderedSymbols.push_back(shortSym);

                            MachoSymbol machoSymbol;
                            if (N_TYPE_GET_STAB(symbol.n_type) != 0) {
                                switch (symbol.n_type) {
                                    case N_GSYM: machoSymbol.type = MachoSymbolType::kGSYM; break;
                                    case N_FNAME: machoSymbol.type = MachoSymbolType::kFNAME; break;
                                    case N_FUN: machoSymbol.type = MachoSymbolType::kFUN; break;
                                    case N_STSYM: machoSymbol.type = MachoSymbolType::kSTSYM; break;
                                    case N_LCSYM: machoSymbol.type = MachoSymbolType::kLCSYM; break;
                                    case N_BNSYM: machoSymbol.type = MachoSymbolType::kBNSYM; break;
                                    case N_AST: machoSymbol.type = MachoSymbolType::kAST; break;
                                    case N_OPT: machoSymbol.type = MachoSymbolType::kOPT; break;
                                    case N_RSYM: machoSymbol.type = MachoSymbolType::kRSYM; break;
                                    case N_SLINE: machoSymbol.type = MachoSymbolType::kSLINE; break;
                                    case N_ENSYM: machoSymbol.type = MachoSymbolType::kENSYM; break;
                                    case N_SSYM: machoSymbol.type = MachoSymbolType::kSSYM; break;
                                    case N_SO: machoSymbol.type = MachoSymbolType::kSO; break;
                                    case N_OSO: machoSymbol.type = MachoSymbolType::kOSO; break;
                                    case N_LSYM: machoSymbol.type = MachoSymbolType::kLSYM; break;
                                    case N_BINCL: machoSymbol.type = MachoSymbolType::kBINCL; break;
                                    case N_SOL: machoSymbol.type = MachoSymbolType::kSOL; break;
                                    case N_PARAMS: machoSymbol.type = MachoSymbolType::kPARAMS; break;
                                    case N_VERSION: machoSymbol.type = MachoSymbolType::kVERSION; break;
                                    case N_OLEVEL: machoSymbol.type = MachoSymbolType::kOLEVEL; break;
                                    case N_PSYM: machoSymbol.type = MachoSymbolType::kPSYM; break;
                                    case N_EINCL: machoSymbol.type = MachoSymbolType::kEINCL; break;
                                    case N_ENTRY: machoSymbol.type = MachoSymbolType::kENTRY; break;
                                    case N_LBRAC: machoSymbol.type = MachoSymbolType::kLBRAC; break;
                                    case N_EXCL: machoSymbol.type = MachoSymbolType::kEXCL; break;
                                    case N_RBRAC: machoSymbol.type = MachoSymbolType::kRBRAC; break;
                                    case N_BCOMM: machoSymbol.type = MachoSymbolType::kBCOMM; break;
                                    case N_ECOMM: machoSymbol.type = MachoSymbolType::kECOMM; break;
                                    case N_ECOML: machoSymbol.type = MachoSymbolType::kECOML; break;
                                    case N_LENG: machoSymbol.type = MachoSymbolType::kLENG; break;
                                    case N_PC: machoSymbol.type = MachoSymbolType::kPC; break;
                                    default: continue;  // Some symbol we're not interested in
                                }
                            } else {
                                switch (N_TYPE_GET_TYPE(symbol.n_type)) {
                                    case N_UNDF: machoSymbol.type = MachoSymbolType::kUndefined; break;
                                    case N_ABS: machoSymbol.type = MachoSymbolType::kAbsolute; break;
                                    case N_SECT: machoSymbol.type = MachoSymbolType::kSection; break;
                                    case N_PBUD: machoSymbol.type = MachoSymbolType::kPreboundUndefined; break;
                                    case N_INDR: machoSymbol.type = MachoSymbolType::kIndirect; break;
                                    default: continue;  // Some symbol we're not interested in
                                }
                            }

                            switch (N_DESC_GET_REFERENCE_TYPE(symbol.n_desc)) {
                                case REFERENCE_FLAG_UNDEFINED_NON_LAZY:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kUndefinedNonLazy;
                                    break;
                                case REFERENCE_FLAG_UNDEFINED_LAZY:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kUndefinedLazy;
                                    break;
                                case REFERENCE_FLAG_DEFINED:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kDefined;
                                    break;
                                case REFERENCE_FLAG_PRIVATE_DEFINED:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateDefined;
                                    break;
                                case REFERENCE_FLAG_PRIVATE_UNDEFINED_NON_LAZY:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateUndefinedNonLazy;
                                    break;
                                case REFERENCE_FLAG_PRIVATE_UNDEFINED_LAZY:
                                    machoSymbol.referenceType = MachoSymbolReferenceType::kPrivateUndefinedLazy;
                                    break;
                            }

                            machoSymbol.referencedDynamically = N_DESC_GET_REFERENCED_DYNAMICALLY(symbol.n_desc);
                            machoSymbol.descDiscarded = N_DESC_GET_DISCARDED(symbol.n_desc);
                            machoSymbol.weakRef = N_DESC_GET_WEAK_REF(symbol.n_desc);
                            machoSymbol.weakDef = N_DESC_GET_WEAK_DEF(symbol.n_desc);
                            machoSymbol.privateExternal = N_TYPE_GET_PEXT(symbol.n_type);
                            machoSymbol.external = N_TYPE_GET_EXT(symbol.n_type);
                            machoSymbol.sectionIndex = symbol.n_sect;
                            machoSymbol.virtualAddress = symbol.n_value;
                            // All symbol names starts with '_', so just skipping 1 char
                            machoSymbol.name = stringTable + symbol.n_un.n_strx + 1;

                            if (machoSymbol.type == MachoSymbolType::kSection) {
                                auto addrFound =
                                    symbolsBounds[machoSymbol.sectionIndex].find(machoSymbol.virtualAddress);
                                assert(addrFound != symbolsBounds[machoSymbol.sectionIndex].end());
                                addrFound++;
                                if (addrFound != symbolsBounds[machoSymbol.sectionIndex].end()) {
                                    machoSymbol.size = *addrFound - machoSymbol.virtualAddress;
                                } else {
                                    context->events->addLog(LogSeverity::kDebug, "wtf?");
                                }
                            }

                            machoSymbol.hash = currentHash;

                            symbolsInSections[machoSymbol.sectionIndex][machoSymbol.virtualAddress] = machoSymbol;
                        }
                        break;
                    }

                    default: break;
                }
                commandOffset += command->cmdsize;
            }

            commandOffset = sizeof(mach_header_64);
            for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
                auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
                switch (command->cmd) {
                    case LC_SEGMENT_64: {
                        auto segmentCommand = reinterpret_cast<segment_command_64*>(machoPtr + commandOffset);
                        auto sectionsPtr =
                            reinterpret_cast<struct section_64*>(machoPtr + commandOffset + sizeof(*segmentCommand));
                        for (uint32_t i = 0; i < segmentCommand->nsects; i++) {
                            auto& section = sectionsPtr[i];
                            if (section.sectname != std::string("__text")) {
                                continue;
                            }

                            auto* relocs = reinterpret_cast<relocation_info*>(machoPtr + section.reloff);
                            for (uint32_t j = 0; j < section.nreloc; j++) {
                                const auto& reloc = relocs[j];
                                const auto& shortSymbol = orderedSymbols[reloc.r_symbolnum];
                                if (shortSymbol.sectionIndex != bssSectionIndex
                                    && shortSymbol.sectionIndex != dataSectionIndex
                                    && shortSymbol.sectionIndex != commonSectionIndex) {
                                    continue;
                                }

                                Relocation rel;

                                switch (reloc.r_length) {
                                    case 2: rel.size = sizeof(uint32_t); break;
                                    case 3: rel.size = sizeof(uint64_t); break;
                                    default:
                                        context->events->addLog(LogSeverity::kError,
                                            "Unsupported relocation length: " + std::to_string(reloc.r_length));
                                        continue;
                                }

                                if (!reloc.r_pcrel) {
                                    context->events->addLog(LogSeverity::kDebug, "reloc.r_pcrel == 0");
                                    continue;
                                }

                                switch (reloc.r_type) {
                                    case X86_64_RELOC_SIGNED:    // for signed 32-bit displacement
                                    case X86_64_RELOC_SIGNED_1:  // for signed 32-bit displacement with a -1 addend
                                    case X86_64_RELOC_SIGNED_2:  // for signed 32-bit displacement with a -2 addend
                                    case X86_64_RELOC_SIGNED_4:  // for signed 32-bit displacement with a -4 addend
                                        break;

                                    case X86_64_RELOC_GOT_LOAD:  // a MOVQ load of a GOT entry
                                        continue;

                                    case X86_64_RELOC_UNSIGNED:    // for absolute addresses
                                    case X86_64_RELOC_BRANCH:      // a CALL/JMP instruction with 32-bit displacement
                                    case X86_64_RELOC_GOT:         // other GOT references
                                    case X86_64_RELOC_SUBTRACTOR:  // must be followed by a X86_64_RELOC_UNSIGNED
                                    case X86_64_RELOC_TLV:         // for thread local variables
                                        context->events->addLog(LogSeverity::kError,
                                            "Unsupported relocation type: " + relToString(reloc.r_type));
                                        continue;
                                    default:
                                        context->events->addLog(LogSeverity::kError,
                                            "Unknown relocation type: " + std::to_string(reloc.r_type));
                                        continue;
                                }

                                auto relocAddr = static_cast<uintptr_t>(reloc.r_address);
                                auto found =
                                    symbolsInSections[static_cast<size_t>(textSectionIndex)].upper_bound(relocAddr);
                                if (found != symbolsInSections[static_cast<size_t>(textSectionIndex)].begin()) {
                                    found--;
                                }
                                if (found->second.virtualAddress > relocAddr
                                    || relocAddr >= found->second.virtualAddress + found->second.size) {
                                    context->events->addLog(LogSeverity::kError, "WTF1");
                                    continue;
                                }

                                rel.targetSymbolName = found->second.name;
                                rel.targetSymbolHash = found->second.hash;
                                rel.relocationOffsetRelativeTargetSymbolAddress =
                                    relocAddr - found->second.virtualAddress;
                                rel.relocationSymbolName = shortSymbol.name;
                                rel.relocationSymbolHash = shortSymbol.hash;
                                res.push_back(rel);
                            }
                        }
                        break;
                    }
                    default: break;
                }
                commandOffset += command->cmdsize;
            }
        }

        return res;
    }

    std::vector<std::string> MachoProgramInfoLoader::getUndefinedSymbolNames(const LiveContext* context,
        const std::string filepath)
    {
        std::vector<std::string> res;

        // Parsing mach-o binary
        auto f = fopen(filepath.c_str(), "r");
        if (!f) {
            context->events->addLog(LogSeverity::kError, "Cannot open " + filepath);
            return res;
        }
        fseek(f, 0, SEEK_END);
        auto length = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);
        auto content = std::unique_ptr<char[]>(new char[length]);
        fread(content.get(), 1, length, f);
        fclose(f);

        auto header = reinterpret_cast<mach_header_64*>(content.get());
        if (header->magic != MH_MAGIC_64) {
            // Probably it is some system "fat" library, we're not interested in it
            // context->events->addLog(LogSeverity::kError, "Cannot read symbols, not a Mach-O 64 binary");
            return res;
        }

        auto machoPtr = content.get();
        auto commandOffset = sizeof(mach_header_64);
        for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
            auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
            switch (command->cmd) {
                case LC_SYMTAB: {
                    auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                    auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                    auto stringTable = machoPtr + table->stroff;
                    res.reserve(table->nsyms);
                    for (uint32_t i = 0; i < table->nsyms; i++) {
                        auto& symbol = symbolsPtr[i];
                        if (N_TYPE_GET_EXT(symbol.n_type) && symbol.n_sect == NO_SECT
                            && N_TYPE_GET_TYPE(symbol.n_type) == N_UNDF) {
                            res.emplace_back(stringTable + symbol.n_un.n_strx + 1);
                        }
                    }
                    break;
                }
                default: break;
            }
            commandOffset += command->cmdsize;
        }

        return res;
    }

    std::vector<std::string> MachoProgramInfoLoader::getExportedSymbolNames(const LiveContext* context,
        const std::string filepath)
    {
        std::vector<std::string> res;

        // Parsing mach-o binary
        auto f = fopen(filepath.c_str(), "r");
        if (!f) {
            context->events->addLog(LogSeverity::kError, "Cannot open " + filepath);
            return res;
        }
        fseek(f, 0, SEEK_END);
        auto length = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);
        auto content = std::unique_ptr<char[]>(new char[length]);
        fread(content.get(), 1, length, f);
        fclose(f);

        auto header = reinterpret_cast<mach_header_64*>(content.get());
        if (header->magic != MH_MAGIC_64) {
            // Probably it is some system "fat" library, we're not interested in it
            // context->events->addLog(LogSeverity::kError, "Cannot read symbols, not a Mach-O 64 binary");
            return res;
        }

        auto machoPtr = content.get();
        auto commandOffset = sizeof(mach_header_64);
        for (uint32_t iCmd = 0; iCmd < header->ncmds; iCmd++) {
            auto command = reinterpret_cast<load_command*>(machoPtr + commandOffset);
            switch (command->cmd) {
                case LC_SYMTAB: {
                    auto table = reinterpret_cast<symtab_command*>(machoPtr + commandOffset);
                    auto symbolsPtr = reinterpret_cast<nlist_64*>(machoPtr + table->symoff);
                    auto stringTable = machoPtr + table->stroff;
                    res.reserve(table->nsyms);
                    for (uint32_t i = 0; i < table->nsyms; i++) {
                        auto& symbol = symbolsPtr[i];
                        if (N_TYPE_GET_EXT(symbol.n_type) && symbol.n_sect != NO_SECT) {
                            res.emplace_back(stringTable + symbol.n_un.n_strx + 1);
                        }
                    }
                    break;
                }
                default: break;
            }
            commandOffset += command->cmdsize;
        }

        return res;
    }
}
