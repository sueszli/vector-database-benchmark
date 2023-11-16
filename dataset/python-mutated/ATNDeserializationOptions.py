ATNDeserializationOptions = None

class ATNDeserializationOptions(object):
    __slots__ = ('readonly', 'verifyATN', 'generateRuleBypassTransitions')
    defaultOptions = None

    def __init__(self, copyFrom: ATNDeserializationOptions=None):
        if False:
            for i in range(10):
                print('nop')
        self.readonly = False
        self.verifyATN = True if copyFrom is None else copyFrom.verifyATN
        self.generateRuleBypassTransitions = False if copyFrom is None else copyFrom.generateRuleBypassTransitions

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key != 'readonly' and self.readonly:
            raise Exception('The object is read only.')
        super(type(self), self).__setattr__(key, value)
ATNDeserializationOptions.defaultOptions = ATNDeserializationOptions()
ATNDeserializationOptions.defaultOptions.readonly = True