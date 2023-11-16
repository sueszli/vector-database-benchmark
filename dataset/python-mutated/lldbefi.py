import lldb
import os
import uuid
import string
import optparse
import shlex
guid_dict = {}

def EFI_GUID_TypeSummary(valobj, internal_dict):
    if False:
        while True:
            i = 10
    ' Type summary for EFI GUID, print C Name if known\n    '
    SBError = lldb.SBError()
    data1_val = valobj.GetChildMemberWithName('Data1')
    data1 = data1_val.GetValueAsUnsigned(0)
    data2_val = valobj.GetChildMemberWithName('Data2')
    data2 = data2_val.GetValueAsUnsigned(0)
    data3_val = valobj.GetChildMemberWithName('Data3')
    data3 = data3_val.GetValueAsUnsigned(0)
    str = '%x-%x-%x-' % (data1, data2, data3)
    data4_val = valobj.GetChildMemberWithName('Data4')
    for i in range(data4_val.num_children):
        if i == 2:
            str += '-'
        str += '%02x' % data4_val.GetChildAtIndex(i).data.GetUnsignedInt8(SBError, 0)
    return guid_dict.get(str.upper(), '')
EFI_STATUS_Dict = {9223372036854775808 | 1: 'Load Error', 9223372036854775808 | 2: 'Invalid Parameter', 9223372036854775808 | 3: 'Unsupported', 9223372036854775808 | 4: 'Bad Buffer Size', 9223372036854775808 | 5: 'Buffer Too Small', 9223372036854775808 | 6: 'Not Ready', 9223372036854775808 | 7: 'Device Error', 9223372036854775808 | 8: 'Write Protected', 9223372036854775808 | 9: 'Out of Resources', 9223372036854775808 | 10: 'Volume Corrupt', 9223372036854775808 | 11: 'Volume Full', 9223372036854775808 | 12: 'No Media', 9223372036854775808 | 13: 'Media changed', 9223372036854775808 | 14: 'Not Found', 9223372036854775808 | 15: 'Access Denied', 9223372036854775808 | 16: 'No Response', 9223372036854775808 | 17: 'No mapping', 9223372036854775808 | 18: 'Time out', 9223372036854775808 | 19: 'Not started', 9223372036854775808 | 20: 'Already started', 9223372036854775808 | 21: 'Aborted', 9223372036854775808 | 22: 'ICMP Error', 9223372036854775808 | 23: 'TFTP Error', 9223372036854775808 | 24: 'Protocol Error', 0: 'Success', 1: 'Warning Unknown Glyph', 2: 'Warning Delete Failure', 3: 'Warning Write Failure', 4: 'Warning Buffer Too Small', 2147483648 | 1: 'Load Error', 2147483648 | 2: 'Invalid Parameter', 2147483648 | 3: 'Unsupported', 2147483648 | 4: 'Bad Buffer Size', 2147483648 | 5: 'Buffer Too Small', 2147483648 | 6: 'Not Ready', 2147483648 | 7: 'Device Error', 2147483648 | 8: 'Write Protected', 2147483648 | 9: 'Out of Resources', 2147483648 | 10: 'Volume Corrupt', 2147483648 | 11: 'Volume Full', 2147483648 | 12: 'No Media', 2147483648 | 13: 'Media changed', 2147483648 | 14: 'Not Found', 2147483648 | 15: 'Access Denied', 2147483648 | 16: 'No Response', 2147483648 | 17: 'No mapping', 2147483648 | 18: 'Time out', 2147483648 | 19: 'Not started', 2147483648 | 20: 'Already started', 2147483648 | 21: 'Aborted', 2147483648 | 22: 'ICMP Error', 2147483648 | 23: 'TFTP Error', 2147483648 | 24: 'Protocol Error'}

def EFI_STATUS_TypeSummary(valobj, internal_dict):
    if False:
        i = 10
        return i + 15
    Status = valobj.GetValueAsUnsigned(0)
    return EFI_STATUS_Dict.get(Status, '')

def EFI_TPL_TypeSummary(valobj, internal_dict):
    if False:
        print('Hello World!')
    if valobj.TypeIsPointerType():
        return ''
    Tpl = valobj.GetValueAsUnsigned(0)
    if Tpl < 4:
        Str = '%d' % Tpl
    elif Tpl == 6:
        Str = 'TPL_DRIVER (Obsolete Concept in edk2)'
    elif Tpl < 8:
        Str = 'TPL_APPLICATION'
        if Tpl - 4 > 0:
            Str += ' + ' + '%d' % (Tpl - 4)
    elif Tpl < 16:
        Str = 'TPL_CALLBACK'
        if Tpl - 8 > 0:
            Str += ' + ' + '%d' % (Tpl - 4)
    elif Tpl < 31:
        Str = 'TPL_NOTIFY'
        if Tpl - 16 > 0:
            Str += ' + ' + '%d' % (Tpl - 4)
    elif Tpl == 31:
        Str = 'TPL_HIGH_LEVEL'
    else:
        Str = 'Invalid TPL'
    return Str

def CHAR16_TypeSummary(valobj, internal_dict):
    if False:
        return 10
    SBError = lldb.SBError()
    Str = ''
    if valobj.TypeIsPointerType():
        if valobj.GetValueAsUnsigned() == 0:
            return 'NULL'
        for i in range(1024):
            Char = valobj.GetPointeeData(i, 1).GetUnsignedInt16(SBError, 0)
            if SBError.fail or Char == 0:
                break
            Str += unichr(Char)
        Str = 'L"' + Str + '"'
        return Str.encode('utf-8', 'replace')
    if valobj.num_children == 0:
        if chr(valobj.unsigned) in string.printable:
            Str = "L'" + unichr(valobj.unsigned) + "'"
            return Str.encode('utf-8', 'replace')
    else:
        for i in range(valobj.num_children):
            Char = valobj.GetChildAtIndex(i).data.GetUnsignedInt16(SBError, 0)
            if Char == 0:
                break
            Str += unichr(Char)
        Str = 'L"' + Str + '"'
        return Str.encode('utf-8', 'replace')
    return Str

def CHAR8_TypeSummary(valobj, internal_dict):
    if False:
        while True:
            i = 10
    SBError = lldb.SBError()
    Str = ''
    if valobj.TypeIsPointerType():
        if valobj.GetValueAsUnsigned() == 0:
            return 'NULL'
        for i in range(1024):
            Char = valobj.GetPointeeData(i, 1).GetUnsignedInt8(SBError, 0)
            if SBError.fail or Char == 0:
                break
            Str += unichr(Char)
        Str = '"' + Str + '"'
        return Str.encode('utf-8', 'replace')
    if valobj.num_children == 0:
        if chr(valobj.unsigned) in string.printable:
            Str = '"' + unichr(valobj.unsigned) + '"'
            return Str.encode('utf-8', 'replace')
    else:
        for i in range(valobj.num_children):
            Char = valobj.GetChildAtIndex(i).data.GetUnsignedInt8(SBError, 0)
            if Char == 0:
                break
            Str += unichr(Char)
        Str = '"' + Str + '"'
        return Str.encode('utf-8', 'replace')
    return Str
device_path_dict = {(1, 1): 'PCI_DEVICE_PATH', (1, 2): 'PCCARD_DEVICE_PATH', (1, 3): 'MEMMAP_DEVICE_PATH', (1, 4): 'VENDOR_DEVICE_PATH', (1, 5): 'CONTROLLER_DEVICE_PATH', (2, 1): 'ACPI_HID_DEVICE_PATH', (2, 2): 'ACPI_EXTENDED_HID_DEVICE_PATH', (2, 3): 'ACPI_ADR_DEVICE_PATH', (3, 1): 'ATAPI_DEVICE_PATH', (3, 18): 'SATA_DEVICE_PATH', (3, 2): 'SCSI_DEVICE_PATH', (3, 3): 'FIBRECHANNEL_DEVICE_PATH', (3, 4): 'F1394_DEVICE_PATH', (3, 5): 'USB_DEVICE_PATH', (3, 15): 'USB_CLASS_DEVICE_PATH', (3, 16): 'FW_SBP2_UNIT_LUN_DEVICE_PATH', (3, 17): 'DEVICE_LOGICAL_UNIT_DEVICE_PATH', (3, 6): 'I2O_DEVICE_PATH', (3, 11): 'MAC_ADDR_DEVICE_PATH', (3, 12): 'IPv4_DEVICE_PATH', (3, 9): 'INFINIBAND_DEVICE_PATH', (3, 14): 'UART_DEVICE_PATH', (3, 10): 'VENDOR_DEVICE_PATH', (3, 19): 'ISCSI_DEVICE_PATH', (4, 1): 'HARDDRIVE_DEVICE_PATH', (4, 2): 'CDROM_DEVICE_PATH', (4, 3): 'VENDOR_DEVICE_PATH', (4, 4): 'FILEPATH_DEVICE_PATH', (4, 5): 'MEDIA_PROTOCOL_DEVICE_PATH', (5, 1): 'BBS_BBS_DEVICE_PATH', (127, 255): 'EFI_DEVICE_PATH_PROTOCOL', (255, 255): 'EFI_DEVICE_PATH_PROTOCOL'}

def EFI_DEVICE_PATH_PROTOCOL_TypeSummary(valobj, internal_dict):
    if False:
        return 10
    if valobj.TypeIsPointerType():
        return ''
    Str = ''
    if valobj.num_children == 3:
        Type = valobj.GetChildMemberWithName('Type').unsigned
        SubType = valobj.GetChildMemberWithName('SubType').unsigned
        if (Type, SubType) in device_path_dict:
            TypeStr = device_path_dict[Type, SubType]
        else:
            TypeStr = ''
        LenLow = valobj.GetChildMemberWithName('Length').GetChildAtIndex(0).unsigned
        LenHigh = valobj.GetChildMemberWithName('Length').GetChildAtIndex(1).unsigned
        Len = LenLow + (LenHigh >> 8)
        Address = long('%d' % valobj.addr)
        if Address == lldb.LLDB_INVALID_ADDRESS:
            ExprStr = ''
        elif Type & 127 == 127:
            ExprStr = 'End Device Path' if SubType == 255 else 'End This Instance'
        else:
            ExprStr = 'expr *(%s *)0x%08x' % (TypeStr, Address)
        Str = ' {\n'
        Str += '   (UINT8) Type    = 0x%02x // %s\n' % (Type, 'END' if Type & 127 == 127 else '')
        Str += '   (UINT8) SubType = 0x%02x // %s\n' % (SubType, ExprStr)
        Str += '   (UINT8 [2]) Length = { // 0x%04x (%d) bytes\n' % (Len, Len)
        Str += '     (UINT8) [0] = 0x%02x\n' % LenLow
        Str += '     (UINT8) [1] = 0x%02x\n' % LenHigh
        Str += '   }\n'
        if Type & 127 == 127 and SubType == 255:
            pass
        elif ExprStr != '':
            NextNode = Address + Len
            Str += "// Next node 'expr *(EFI_DEVICE_PATH_PROTOCOL *)0x%08x'\n" % NextNode
    return Str

def TypePrintFormating(debugger):
    if False:
        return 10
    category = debugger.GetDefaultCategory()
    FormatBool = lldb.SBTypeFormat(lldb.eFormatBoolean)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('BOOLEAN'), FormatBool)
    FormatHex = lldb.SBTypeFormat(lldb.eFormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('UINT64'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('INT64'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('UINT32'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('INT32'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('UINT16'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('INT16'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('UINT8'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('INT8'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('UINTN'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('INTN'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('CHAR8'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('CHAR16'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_PHYSICAL_ADDRESS'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('PHYSICAL_ADDRESS'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_STATUS'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_TPL'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_LBA'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_BOOT_MODE'), FormatHex)
    category.AddTypeFormat(lldb.SBTypeNameSpecifier('EFI_FV_FILETYPE'), FormatHex)
    debugger.HandleCommand('type summary add EFI_GUID --python-function lldbefi.EFI_GUID_TypeSummary')
    debugger.HandleCommand('type summary add EFI_STATUS --python-function lldbefi.EFI_STATUS_TypeSummary')
    debugger.HandleCommand('type summary add EFI_TPL --python-function lldbefi.EFI_TPL_TypeSummary')
    debugger.HandleCommand('type summary add EFI_DEVICE_PATH_PROTOCOL --python-function lldbefi.EFI_DEVICE_PATH_PROTOCOL_TypeSummary')
    debugger.HandleCommand('type summary add CHAR16 --python-function lldbefi.CHAR16_TypeSummary')
    debugger.HandleCommand('type summary add --regex "CHAR16 \\[[0-9]+\\]" --python-function lldbefi.CHAR16_TypeSummary')
    debugger.HandleCommand('type summary add CHAR8 --python-function lldbefi.CHAR8_TypeSummary')
    debugger.HandleCommand('type summary add --regex "CHAR8 \\[[0-9]+\\]" --python-function lldbefi.CHAR8_TypeSummary')
    debugger.HandleCommand('setting set frame-format "frame #${frame.index}: ${frame.pc}{ ${module.file.basename}{:${function.name}()${function.pc-offset}}}{ at ${line.file.fullpath}:${line.number}}\n"')
gEmulatorBreakWorkaroundNeeded = True

def LoadEmulatorEfiSymbols(frame, bp_loc, internal_dict):
    if False:
        for i in range(10):
            print('nop')
    global gEmulatorBreakWorkaroundNeeded
    if gEmulatorBreakWorkaroundNeeded:
        frame.thread.process.target.debugger.HandleCommand('process handle SIGALRM -n false')
        gEmulatorBreakWorkaroundNeeded = False
    Error = lldb.SBError()
    FileNamePtr = frame.FindVariable('FileName').GetValueAsUnsigned()
    FileNameLen = frame.FindVariable('FileNameLength').GetValueAsUnsigned()
    FileName = frame.thread.process.ReadCStringFromMemory(FileNamePtr, FileNameLen, Error)
    if not Error.Success():
        print('!ReadCStringFromMemory() did not find a %d byte C string at %x' % (FileNameLen, FileNamePtr))
        return False
    debugger = frame.thread.process.target.debugger
    if frame.FindVariable('AddSymbolFlag').GetValueAsUnsigned() == 1:
        LoadAddress = frame.FindVariable('LoadAddress').GetValueAsUnsigned() - 576
        debugger.HandleCommand('target modules add  %s' % FileName)
        print('target modules load --slid 0x%x %s' % (LoadAddress, FileName))
        debugger.HandleCommand('target modules load --slide 0x%x --file %s' % (LoadAddress, FileName))
    else:
        target = debugger.GetSelectedTarget()
        for SBModule in target.module_iter():
            ModuleName = SBModule.GetFileSpec().GetDirectory() + '/'
            ModuleName += SBModule.GetFileSpec().GetFilename()
            if FileName == ModuleName or FileName == SBModule.GetFileSpec().GetFilename():
                target.ClearModuleLoadAddress(SBModule)
                if not target.RemoveModule(SBModule):
                    print('!lldb.target.RemoveModule (%s) FAILED' % SBModule)
    return False

def GuidToCStructStr(guid, Name=False):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(guid, bytearray):
        Uuid = uuid.UUID(guid)
        guid = bytearray(Uuid.bytes_le)
    return '{ 0x%02.2X%02.2X%02.2X%02.2X, 0x%02.2X%02.2X, 0x%02.2X%02.2X, { 0x%02.2X, 0x%02.2X, 0x%02.2X, 0x%02.2X, 0x%02.2X, 0x%02.2X, 0x%02.2X, 0x%02.2X } }' % (guid[3], guid[2], guid[1], guid[0], guid[5], guid[4], guid[7], guid[6], guid[8], guid[9], guid[10], guid[11], guid[12], guid[13], guid[14], guid[15])

def ParseGuidString(GuidStr):
    if False:
        for i in range(10):
            print('nop')
    if '{' in GuidStr:
        Hex = ''.join((x for x in GuidStr if x not in '{,}')).split()
        Str = '%08X-%04X-%04X-%02.2X%02.2X-%02.2X%02.2X%02.2X%02.2X%02.2X%02.2X' % (int(Hex[0], 0), int(Hex[1], 0), int(Hex[2], 0), int(Hex[3], 0), int(Hex[4], 0), int(Hex[5], 0), int(Hex[6], 0), int(Hex[7], 0), int(Hex[8], 0), int(Hex[9], 0), int(Hex[10], 0))
    elif GuidStr.count('-') == 4:
        Check = '%s' % str(uuid.UUID(GuidStr)).upper()
        if GuidStr.upper() == Check:
            Str = GuidStr.upper()
        else:
            Ste = ''
    else:
        Str = ''
    return Str

def create_guid_options():
    if False:
        print('Hello World!')
    usage = 'usage: %prog [data]'
    description = 'lookup EFI_GUID by CName, C struct, or GUID string and print out all three.\n    '
    parser = optparse.OptionParser(description=description, prog='guid', usage=usage)
    return parser

def efi_guid_command(debugger, command, result, dict):
    if False:
        while True:
            i = 10
    command_args = shlex.split(command)
    parser = create_guid_options()
    try:
        (options, args) = parser.parse_args(command_args)
        if len(args) >= 1:
            if args[0] == '{':
                args[0] = ' '.join(args)
            GuidStr = ParseGuidString(args[0])
            if GuidStr == '':
                GuidStr = [Key for (Key, Value) in guid_dict.iteritems() if Value == args[0]][0]
            GuidStr = GuidStr.upper()
    except:
        result.SetError('option parsing failed')
        return
    if len(args) >= 1:
        if GuidStr in guid_dict:
            print('%s = %s' % (guid_dict[GuidStr], GuidStr))
            print('%s = %s' % (guid_dict[GuidStr], GuidToCStructStr(GuidStr)))
        else:
            print(GuidStr)
    else:
        width = max((len(v) for (k, v) in guid_dict.iteritems()))
        for value in sorted(guid_dict, key=guid_dict.get):
            print('%-*s %s %s' % (width, guid_dict[value], value, GuidToCStructStr(value)))
    return

def __lldb_init_module(debugger, internal_dict):
    if False:
        print('Hello World!')
    global guid_dict
    inputfile = os.getcwd()
    inputfile += os.sep + os.pardir + os.sep + 'FV' + os.sep + 'Guid.xref'
    with open(inputfile) as f:
        for line in f:
            data = line.split(' ')
            if len(data) >= 2:
                guid_dict[data[0].upper()] = data[1].strip('\n')
    TypePrintFormating(debugger)
    parser = create_guid_options()
    efi_guid_command.__doc__ = parser.format_help()
    debugger.HandleCommand('command script add -f lldbefi.efi_guid_command guid')
    Target = debugger.GetTargetAtIndex(0)
    if Target:
        Breakpoint = Target.BreakpointCreateByName('SecGdbScriptBreak')
        if Breakpoint.GetNumLocations() == 1:
            debugger.HandleCommand('breakpoint command add -s python -F lldbefi.LoadEmulatorEfiSymbols {id}'.format(id=Breakpoint.GetID()))
            print('Type r to run emulator. SecLldbScriptBreak armed. EFI modules should now get source level debugging in the emulator.')