import volatility.obj as obj

class _KPCROnx86(obj.CType):
    """KPCR for 32bit windows"""

    def idt_entries(self):
        if False:
            i = 10
            return i + 15
        for (i, entry) in enumerate(self.IDT.dereference()):
            yield (i, entry)

    def gdt_entries(self):
        if False:
            return 10
        for (i, entry) in enumerate(self.GDT.dereference()):
            yield (i, entry)

    def get_kdbg(self):
        if False:
            print('Hello World!')
        'Find this CPUs KDBG. \n\n        Please note the KdVersionBlock pointer is NULL on\n        all KPCR structures except the one for the first CPU. \n        In some cases on x64, even the first CPU has a NULL\n        KdVersionBlock, so this is really a hit-or-miss. \n        '
        DebuggerDataList = self.KdVersionBlock.dereference_as('_DBGKD_GET_VERSION64').DebuggerDataList
        return DebuggerDataList.dereference().dereference_as('_KDDEBUGGER_DATA64')

    @property
    def ProcessorBlock(self):
        if False:
            for i in range(10):
                print('nop')
        return self.PrcbData

class _KPCROnx64(_KPCROnx86):
    """KPCR for x64 windows"""

    @property
    def ProcessorBlock(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Prcb

    @property
    def IDT(self):
        if False:
            i = 10
            return i + 15
        return self.IdtBase

    @property
    def GDT(self):
        if False:
            return 10
        return self.GdtBase

class KPCRProfileModification(obj.ProfileModification):
    before = ['WindowsObjectClasses']
    conditions = {'os': lambda x: x == 'windows'}

    def modification(self, profile):
        if False:
            i = 10
            return i + 15
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            kpcr_class = _KPCROnx86
        else:
            kpcr_class = _KPCROnx64
        profile.object_classes.update({'_KPCR': kpcr_class})
        profile.merge_overlay({'_KPRCB': [None, {'VendorString': [None, ['String', dict(length=13)]]}]})