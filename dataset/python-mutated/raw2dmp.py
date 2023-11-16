import os
import volatility.obj as obj
import volatility.utils as utils
import volatility.addrspace as addrspace
import volatility.plugins.imagecopy as imagecopy

class Raw2dmp(imagecopy.ImageCopy):
    """Converts a physical memory sample to a windbg crash dump"""

    def calculate(self):
        if False:
            print('Hello World!')
        config = self._config
        output = self._config.OUTPUT_IMAGE
        return self.convert_to_crash(config, output)

    @staticmethod
    def convert_to_crash(config, output):
        if False:
            return 10
        blocksize = config.BLOCKSIZE
        config.WRITE = True
        pspace = utils.load_as(config, astype='physical')
        vspace = utils.load_as(config)
        memory_model = pspace.profile.metadata.get('memory_model', '32bit')
        if memory_model == '64bit':
            header_format = '_DMP_HEADER64'
        else:
            header_format = '_DMP_HEADER'
        headerlen = pspace.profile.get_obj_size(header_format)
        headerspace = addrspace.BufferAddressSpace(config, 0, 'PAGE' * (headerlen / 4))
        header = obj.Object(header_format, offset=0, vm=headerspace)
        kuser = obj.Object('_KUSER_SHARED_DATA', offset=obj.VolMagic(vspace).KUSER_SHARED_DATA.v(), vm=vspace)
        kdbg = obj.VolMagic(vspace).KDBG.v()
        if not kdbg:
            raise RuntimeError("Couldn't find KDBG block. Wrong profile?")
        dbgkd = kdbg.dbgkd_version64()
        if not dbgkd:
            raise RuntimeError("Couldn't find _DBGKD_GET_VERSION64.")
        for i in range(len('PAGE')):
            header.Signature[i] = [ord(x) for x in 'PAGE'][i]
        dumptext = 'DUMP'
        header.KdDebuggerDataBlock = kdbg.obj_offset
        if memory_model == '64bit':
            dumptext = 'DU64'
            header.KdDebuggerDataBlock = kdbg.obj_offset | 18446462598732840960
        for i in range(len(dumptext)):
            header.ValidDump[i] = ord(dumptext[i])
        if memory_model == '32bit':
            if hasattr(vspace, 'pae') and vspace.pae == True:
                header.PaeEnabled = 1
            else:
                header.PaeEnabled = 0
        header.MajorVersion = dbgkd.MajorVersion
        header.MinorVersion = dbgkd.MinorVersion
        header.DirectoryTableBase = vspace.dtb
        header.PfnDataBase = kdbg.MmPfnDatabase
        header.PsLoadedModuleList = kdbg.PsLoadedModuleList
        header.PsActiveProcessHead = kdbg.PsActiveProcessHead
        header.MachineImageType = dbgkd.MachineType
        headerspace.write(header.DumpType.obj_offset, '\x01\x00\x00\x00')
        header.NumberProcessors = len(list(kdbg.kpcrs()))
        header.SystemTime = kuser.SystemTime.as_windows_timestamp()
        header.BugCheckCode = 0
        header.BugCheckCodeParameter[0] = 0
        header.BugCheckCodeParameter[1] = 0
        header.BugCheckCodeParameter[2] = 0
        header.BugCheckCodeParameter[3] = 0
        last_run = list(pspace.get_available_addresses())[-1]
        num_pages = (last_run[0] + last_run[1]) / 4096
        header.PhysicalMemoryBlockBuffer.NumberOfRuns = 1
        header.PhysicalMemoryBlockBuffer.NumberOfPages = num_pages
        header.PhysicalMemoryBlockBuffer.Run[0].BasePage = 0
        header.PhysicalMemoryBlockBuffer.Run[0].PageCount = num_pages
        header.RequiredDumpSpace = (num_pages + 2) * 4096
        ContextRecordOffset = headerspace.profile.get_obj_offset(header_format, 'ContextRecord')
        ExceptionOffset = headerspace.profile.get_obj_offset(header_format, 'Exception')
        headerspace.write(ContextRecordOffset, '\x00' * (ExceptionOffset - ContextRecordOffset))
        CommentOffset = headerspace.profile.get_obj_offset(header_format, 'Comment')
        headerspace.write(CommentOffset, 'File was converted with Volatility' + '\x00')
        yield (0, headerlen, headerspace.read(0, headerlen))
        for (s, l) in pspace.get_available_addresses():
            for i in range(s, s + l, blocksize):
                len_to_read = min(blocksize, s + l - i)
                yield (i + headerlen, len_to_read, pspace.read(i, len_to_read))
        config.LOCATION = 'file://' + output
        crash_vspace = utils.load_as(config)
        crash_kdbg = obj.VolMagic(crash_vspace).KDBG.v()
        kpcr = list(crash_kdbg.kpcrs())[0]
        if memory_model == '32bit':
            kpcr.PrcbData.ProcessorState.ContextFrame.SegGs = 0
            kpcr.PrcbData.ProcessorState.ContextFrame.SegCs = 8
            kpcr.PrcbData.ProcessorState.ContextFrame.SegDs = 35
            kpcr.PrcbData.ProcessorState.ContextFrame.SegEs = 35
            kpcr.PrcbData.ProcessorState.ContextFrame.SegFs = 48
            kpcr.PrcbData.ProcessorState.ContextFrame.SegSs = 16
        else:
            kpcr.Prcb.ProcessorState.ContextFrame.SegGs = 0
            kpcr.Prcb.ProcessorState.ContextFrame.SegCs = 24
            kpcr.Prcb.ProcessorState.ContextFrame.SegDs = 43
            kpcr.Prcb.ProcessorState.ContextFrame.SegEs = 43
            kpcr.Prcb.ProcessorState.ContextFrame.SegFs = 83
            kpcr.Prcb.ProcessorState.ContextFrame.SegSs = 24
        if hasattr(kdbg, 'block_encoded') and kdbg.block_encoded:
            crash_vspace.write(crash_kdbg.obj_offset, kdbg.obj_vm.data)