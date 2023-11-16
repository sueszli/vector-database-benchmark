from fairseq.data import Dictionary

class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', mask='<mask>'):
        if False:
            return 10
        super().__init__(pad=pad, eos=eos, unk=unk)
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        if False:
            for i in range(10):
                print('nop')
        'Helper to get index of mask symbol'
        return self.mask_index

class BertDictionary(MaskedLMDictionary):
    """
    Dictionary for BERT task. This extends MaskedLMDictionary by adding support
    for cls and sep symbols.
    """

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', mask='<mask>', cls='<cls>', sep='<sep>'):
        if False:
            return 10
        super().__init__(pad=pad, eos=eos, unk=unk, mask=mask)
        self.cls_word = cls
        self.sep_word = sep
        self.cls_index = self.add_symbol(cls)
        self.sep_index = self.add_symbol(sep)
        self.nspecial = len(self.symbols)

    def cls(self):
        if False:
            print('Hello World!')
        'Helper to get index of cls symbol'
        return self.cls_index

    def sep(self):
        if False:
            i = 10
            return i + 15
        'Helper to get index of sep symbol'
        return self.sep_index