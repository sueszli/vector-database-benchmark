import volatility.obj as obj

class _KDDEBUGGER_DATA64(obj.CType):
    """A class for KDBG"""

    def is_valid(self):
        if False:
            while True:
                i = 10
        'Returns true if the kdbg_object appears valid'
        return obj.CType.is_valid(self) and self.Header.OwnerTag == 1195525195

    @property
    def ServicePack(self):
        if False:
            while True:
                i = 10
        'Get the service pack number. This is something\n        like 0x100 for SP1, 0x200 for SP2 etc. \n        '
        csdresult = obj.Object('unsigned long', offset=self.CmNtCSDVersion, vm=self.obj_native_vm)
        return csdresult >> 8 & 4294967295

    def processes(self):
        if False:
            while True:
                i = 10
        'Enumerate processes'
        list_head = self.PsActiveProcessHead.dereference()
        if not list_head:
            raise AttributeError('Could not list tasks, please verify your --profile with kdbgscan')
        for l in list_head.list_of_type('_EPROCESS', 'ActiveProcessLinks'):
            yield l

    def modules(self):
        if False:
            i = 10
            return i + 15
        'Enumerate modules'
        list_head = self.PsLoadedModuleList.dereference()
        if not list_head:
            raise AttributeError('Could not list modules, please verify your --profile with kdbgscan')
        for l in list_head.dereference_as('_LIST_ENTRY').list_of_type('_LDR_DATA_TABLE_ENTRY', 'InLoadOrderLinks'):
            yield l

    def dbgkd_version64(self):
        if False:
            i = 10
            return i + 15
        'Finds _DBGKD_GET_VERSION64 corresponding to this KDBG'
        verinfo = self.dbgkd_find_version64(pages_to_scan=1)
        if verinfo:
            return verinfo
        return self.dbgkd_find_version64(pages_to_scan=16)

    def dbgkd_find_version64(self, pages_to_scan):
        if False:
            print('Hello World!')
        'Scan backwards from the base of KDBG to find the \n        _DBGKD_GET_VERSION64. We have a winner when kernel \n        base addresses and process list head match.'
        memory_model = self.obj_native_vm.profile.metadata.get('memory_model', '32bit')
        dbgkd_off = self.obj_offset & 18446744073709547520
        dbgkd_off -= pages_to_scan / 2 * 4096
        dbgkd_end = dbgkd_off + pages_to_scan * 4096
        dbgkd_size = self.obj_native_vm.profile.get_obj_size('_DBGKD_GET_VERSION64')
        while dbgkd_off <= dbgkd_end - dbgkd_size:
            dbgkd = obj.Object('_DBGKD_GET_VERSION64', offset=dbgkd_off, vm=self.obj_native_vm)
            if memory_model == '32bit':
                KernBase = dbgkd.KernBase & 4294967295
                PsLoadedModuleList = dbgkd.PsLoadedModuleList & 4294967295
            else:
                KernBase = dbgkd.KernBase
                PsLoadedModuleList = dbgkd.PsLoadedModuleList
            if KernBase == self.KernBase and PsLoadedModuleList == self.PsLoadedModuleList:
                return dbgkd
            dbgkd_off += 1
        return obj.NoneObject('Cannot find _DBGKD_GET_VERSION64')

    def kpcrs(self):
        if False:
            i = 10
            return i + 15
        'Generator for KPCRs referenced by this KDBG. \n\n        These are returned in the order in which the \n        processors were registered. \n        '
        if self.obj_native_vm.profile.metadata.get('memory_model', '32bit') == '32bit':
            prcb_member = 'PrcbData'
        else:
            prcb_member = 'Prcb'
        cpu_array = self.KiProcessorBlock.dereference()
        for p in cpu_array:
            if p == None or p == 0:
                break
            kpcrb = p.dereference_as('_KPRCB')
            kpcr = obj.Object('_KPCR', offset=kpcrb.obj_offset - self.obj_native_vm.profile.get_obj_offset('_KPCR', prcb_member), vm=self.obj_native_vm, parent=self)
            if kpcr.is_valid():
                yield kpcr

class KDBGObjectClass(obj.ProfileModification):
    """Add the KDBG object class to all Windows profiles"""
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.object_classes.update({'_KDDEBUGGER_DATA64': _KDDEBUGGER_DATA64})
        if profile.metadata.get('memory_model', '32bit'):
            max_processors = 32
        else:
            max_processors = 64
        profile.merge_overlay({'_KDDEBUGGER_DATA64': [None, {'NtBuildLab': [None, ['pointer', ['String', dict(length=32)]]], 'KiProcessorBlock': [None, ['pointer', ['array', max_processors, ['pointer', ['_KPRCB']]]]], 'PsActiveProcessHead': [None, ['pointer', ['_LIST_ENTRY']]], 'PsLoadedModuleList': [None, ['pointer', ['_LIST_ENTRY']]], 'MmUnloadedDrivers': [None, ['pointer', ['pointer', ['array', lambda x: x.MmLastUnloadedDriver.dereference(), ['_UNLOADED_DRIVER']]]]], 'MmLastUnloadedDriver': [None, ['pointer', ['unsigned int']]]}]})

class UnloadedDriverVTypes(obj.ProfileModification):
    """Add the unloaded driver structure definitions"""
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            vtypes = {'_UNLOADED_DRIVER': [24, {'Name': [0, ['_UNICODE_STRING']], 'StartAddress': [8, ['address']], 'EndAddress': [12, ['address']], 'CurrentTime': [16, ['WinTimeStamp', {}]]}]}
        else:
            vtypes = {'_UNLOADED_DRIVER': [40, {'Name': [0, ['_UNICODE_STRING']], 'StartAddress': [16, ['address']], 'EndAddress': [24, ['address']], 'CurrentTime': [32, ['WinTimeStamp', {}]]}]}
        profile.vtypes.update(vtypes)
kdbg_vtypes = {'_DBGKD_DEBUG_DATA_HEADER64': [24, {'List': [0, ['LIST_ENTRY64']], 'OwnerTag': [16, ['unsigned long']], 'Size': [20, ['unsigned long']]}], '_KDDEBUGGER_DATA64': [832, {'Header': [0, ['_DBGKD_DEBUG_DATA_HEADER64']], 'KernBase': [24, ['unsigned long long']], 'BreakpointWithStatus': [32, ['unsigned long long']], 'SavedContext': [40, ['unsigned long long']], 'ThCallbackStack': [48, ['unsigned short']], 'NextCallback': [50, ['unsigned short']], 'FramePointer': [52, ['unsigned short']], 'KiCallUserMode': [56, ['unsigned long long']], 'KeUserCallbackDispatcher': [64, ['unsigned long long']], 'PsLoadedModuleList': [72, ['unsigned long long']], 'PsActiveProcessHead': [80, ['unsigned long long']], 'PspCidTable': [88, ['unsigned long long']], 'ExpSystemResourcesList': [96, ['unsigned long long']], 'ExpPagedPoolDescriptor': [104, ['unsigned long long']], 'ExpNumberOfPagedPools': [112, ['unsigned long long']], 'KeTimeIncrement': [120, ['unsigned long long']], 'KeBugCheckCallbackListHead': [128, ['unsigned long long']], 'KiBugcheckData': [136, ['unsigned long long']], 'IopErrorLogListHead': [144, ['unsigned long long']], 'ObpRootDirectoryObject': [152, ['unsigned long long']], 'ObpTypeObjectType': [160, ['unsigned long long']], 'MmSystemCacheStart': [168, ['unsigned long long']], 'MmSystemCacheEnd': [176, ['unsigned long long']], 'MmSystemCacheWs': [184, ['unsigned long long']], 'MmPfnDatabase': [192, ['unsigned long long']], 'MmSystemPtesStart': [200, ['unsigned long long']], 'MmSystemPtesEnd': [208, ['unsigned long long']], 'MmSubsectionBase': [216, ['unsigned long long']], 'MmNumberOfPagingFiles': [224, ['unsigned long long']], 'MmLowestPhysicalPage': [232, ['unsigned long long']], 'MmHighestPhysicalPage': [240, ['unsigned long long']], 'MmNumberOfPhysicalPages': [248, ['unsigned long long']], 'MmMaximumNonPagedPoolInBytes': [256, ['unsigned long long']], 'MmNonPagedSystemStart': [264, ['unsigned long long']], 'MmNonPagedPoolStart': [272, ['unsigned long long']], 'MmNonPagedPoolEnd': [280, ['unsigned long long']], 'MmPagedPoolStart': [288, ['unsigned long long']], 'MmPagedPoolEnd': [296, ['unsigned long long']], 'MmPagedPoolInformation': [304, ['unsigned long long']], 'MmPageSize': [312, ['unsigned long long']], 'MmSizeOfPagedPoolInBytes': [320, ['unsigned long long']], 'MmTotalCommitLimit': [328, ['unsigned long long']], 'MmTotalCommittedPages': [336, ['unsigned long long']], 'MmSharedCommit': [344, ['unsigned long long']], 'MmDriverCommit': [352, ['unsigned long long']], 'MmProcessCommit': [360, ['unsigned long long']], 'MmPagedPoolCommit': [368, ['unsigned long long']], 'MmExtendedCommit': [376, ['unsigned long long']], 'MmZeroedPageListHead': [384, ['unsigned long long']], 'MmFreePageListHead': [392, ['unsigned long long']], 'MmStandbyPageListHead': [400, ['unsigned long long']], 'MmModifiedPageListHead': [408, ['unsigned long long']], 'MmModifiedNoWritePageListHead': [416, ['unsigned long long']], 'MmAvailablePages': [424, ['unsigned long long']], 'MmResidentAvailablePages': [432, ['unsigned long long']], 'PoolTrackTable': [440, ['unsigned long long']], 'NonPagedPoolDescriptor': [448, ['unsigned long long']], 'MmHighestUserAddress': [456, ['unsigned long long']], 'MmSystemRangeStart': [464, ['unsigned long long']], 'MmUserProbeAddress': [472, ['unsigned long long']], 'KdPrintCircularBuffer': [480, ['unsigned long long']], 'KdPrintCircularBufferEnd': [488, ['unsigned long long']], 'KdPrintWritePointer': [496, ['unsigned long long']], 'KdPrintRolloverCount': [504, ['unsigned long long']], 'MmLoadedUserImageList': [512, ['unsigned long long']], 'NtBuildLab': [520, ['unsigned long long']], 'KiNormalSystemCall': [528, ['unsigned long long']], 'KiProcessorBlock': [536, ['unsigned long long']], 'MmUnloadedDrivers': [544, ['unsigned long long']], 'MmLastUnloadedDriver': [552, ['unsigned long long']], 'MmTriageActionTaken': [560, ['unsigned long long']], 'MmSpecialPoolTag': [568, ['unsigned long long']], 'KernelVerifier': [576, ['unsigned long long']], 'MmVerifierData': [584, ['unsigned long long']], 'MmAllocatedNonPagedPool': [592, ['unsigned long long']], 'MmPeakCommitment': [600, ['unsigned long long']], 'MmTotalCommitLimitMaximum': [608, ['unsigned long long']], 'CmNtCSDVersion': [616, ['unsigned long long']], 'MmPhysicalMemoryBlock': [624, ['unsigned long long']], 'MmSessionBase': [632, ['unsigned long long']], 'MmSessionSize': [640, ['unsigned long long']], 'MmSystemParentTablePage': [648, ['unsigned long long']], 'MmVirtualTranslationBase': [656, ['unsigned long long']], 'OffsetKThreadNextProcessor': [664, ['unsigned short']], 'OffsetKThreadTeb': [666, ['unsigned short']], 'OffsetKThreadKernelStack': [668, ['unsigned short']], 'OffsetKThreadInitialStack': [670, ['unsigned short']], 'OffsetKThreadApcProcess': [672, ['unsigned short']], 'OffsetKThreadState': [674, ['unsigned short']], 'OffsetKThreadBStore': [676, ['unsigned short']], 'OffsetKThreadBStoreLimit': [678, ['unsigned short']], 'SizeEProcess': [680, ['unsigned short']], 'OffsetEprocessPeb': [682, ['unsigned short']], 'OffsetEprocessParentCID': [684, ['unsigned short']], 'OffsetEprocessDirectoryTableBase': [686, ['unsigned short']], 'SizePrcb': [688, ['unsigned short']], 'OffsetPrcbDpcRoutine': [690, ['unsigned short']], 'OffsetPrcbCurrentThread': [692, ['unsigned short']], 'OffsetPrcbMhz': [694, ['unsigned short']], 'OffsetPrcbCpuType': [696, ['unsigned short']], 'OffsetPrcbVendorString': [698, ['unsigned short']], 'OffsetPrcbProcStateContext': [700, ['unsigned short']], 'OffsetPrcbNumber': [702, ['unsigned short']], 'SizeEThread': [704, ['unsigned short']], 'KdPrintCircularBufferPtr': [712, ['unsigned long long']], 'KdPrintBufferSize': [720, ['unsigned long long']], 'KeLoaderBlock': [728, ['unsigned long long']], 'SizePcr': [736, ['unsigned short']], 'OffsetPcrSelfPcr': [738, ['unsigned short']], 'OffsetPcrCurrentPrcb': [740, ['unsigned short']], 'OffsetPcrContainedPrcb': [742, ['unsigned short']], 'OffsetPcrInitialBStore': [744, ['unsigned short']], 'OffsetPcrBStoreLimit': [746, ['unsigned short']], 'OffsetPcrInitialStack': [748, ['unsigned short']], 'OffsetPcrStackLimit': [750, ['unsigned short']], 'OffsetPrcbPcrPage': [752, ['unsigned short']], 'OffsetPrcbProcStateSpecialReg': [754, ['unsigned short']], 'GdtR0Code': [756, ['unsigned short']], 'GdtR0Data': [758, ['unsigned short']], 'GdtR0Pcr': [760, ['unsigned short']], 'GdtR3Code': [762, ['unsigned short']], 'GdtR3Data': [764, ['unsigned short']], 'GdtR3Teb': [766, ['unsigned short']], 'GdtLdt': [768, ['unsigned short']], 'GdtTss': [770, ['unsigned short']], 'Gdt64R3CmCode': [772, ['unsigned short']], 'Gdt64R3CmTeb': [774, ['unsigned short']], 'IopNumTriageDumpDataBlocks': [776, ['unsigned long long']], 'IopTriageDumpDataBlocks': [784, ['unsigned long long']], 'VfCrashDataBlock': [792, ['unsigned long long']], 'MmBadPagesDetected': [800, ['unsigned long long']], 'MmZeroedPageSingleBitErrorsDetected': [808, ['unsigned long long']], 'EtwpDebuggerData': [816, ['unsigned long long']], 'OffsetPrcbContext': [824, ['unsigned short']]}]}