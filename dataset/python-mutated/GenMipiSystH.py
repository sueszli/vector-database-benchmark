import os
import re
ExistingValueToBeReplaced = [['@SYST_CFG_VERSION_MAJOR@', '1'], ['@SYST_CFG_VERSION_MINOR@', '0'], ['@SYST_CFG_VERSION_PATCH@', '0'], ['@SYST_CFG_CONFORMANCE_LEVEL@', '30'], ['mipi_syst/platform.h', 'Platform.h']]
ExistingDefinitionToBeRemoved = ['MIPI_SYST_PCFG_ENABLE_PLATFORM_STATE_DATA', 'MIPI_SYST_PCFG_ENABLE_HEAP_MEMORY', 'MIPI_SYST_PCFG_ENABLE_PRINTF_API', 'MIPI_SYST_PCFG_ENABLE_LOCATION_RECORD', 'MIPI_SYST_PCFG_ENABLE_LOCATION_ADDRESS']
NewItemToBeAdded = ['typedef struct mipi_syst_handle_flags MIPI_SYST_HANDLE_FLAGS;', 'typedef struct mipi_syst_msg_tag MIPI_SYST_MSG_TAG;', 'typedef struct mipi_syst_guid MIPI_SYST_GUID;', 'typedef enum mipi_syst_severity MIPI_SYST_SEVERITY;', 'typedef struct mipi_syst_handle MIPI_SYST_HANDLE;', 'typedef struct mipi_syst_header MIPI_SYST_HEADER;']

def ProcessSpecialCharacter(Str):
    if False:
        for i in range(10):
            print('nop')
    Str = Str.rstrip(' \n')
    Str = Str.replace('\t', '  ')
    Str += '\n'
    return Str

def ReplaceOldValue(Str):
    if False:
        i = 10
        return i + 15
    for i in range(len(ExistingValueToBeReplaced)):
        Result = re.search(ExistingValueToBeReplaced[i][0], Str)
        if Result is not None:
            Str = Str.replace(ExistingValueToBeReplaced[i][0], ExistingValueToBeReplaced[i][1])
            break
    return Str

def RemoveDefinition(Str):
    if False:
        for i in range(10):
            print('nop')
    Result = re.search('\\*', Str)
    if Result is None:
        for i in range(len(ExistingDefinitionToBeRemoved)):
            Result = re.search(ExistingDefinitionToBeRemoved[i], Str)
            if Result is not None:
                Result = re.search('defined', Str)
                if Result is None:
                    Str = Str + '#undef ' + ExistingDefinitionToBeRemoved[i]
                    break
    return Str

def main():
    if False:
        while True:
            i = 10
    MipiSystHSrcDir = 'mipisyst/library/include/mipi_syst.h.in'
    MipiSystHRealSrcDir = os.path.join(os.getcwd(), os.path.normpath(MipiSystHSrcDir))
    MipiSystHRealDstDir = os.path.join(os.getcwd(), 'mipi_syst.h')
    with open(MipiSystHRealSrcDir, 'r') as rfObj:
        SrcFile = rfObj.readlines()
        for lineIndex in range(len(SrcFile)):
            SrcFile[lineIndex] = ProcessSpecialCharacter(SrcFile[lineIndex])
            SrcFile[lineIndex] = ReplaceOldValue(SrcFile[lineIndex])
            SrcFile[lineIndex] = RemoveDefinition(SrcFile[lineIndex])
    i = -1
    for struct in NewItemToBeAdded:
        struct += '\n'
        SrcFile.insert(i, struct)
        i -= 1
    with open(MipiSystHRealDstDir, 'w') as wfObj:
        wfObj.writelines(SrcFile)
if __name__ == '__main__':
    main()