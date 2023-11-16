""" Objects use to describe control flow escapes.

Typically returned by shape operations to indicate what can and can not
have happened.

"""

class ControlFlowDescriptionBase(object):

    @staticmethod
    def isUnsupported():
        if False:
            for i in range(10):
                print('nop')
        return False

class ControlFlowDescriptionElementBasedEscape(ControlFlowDescriptionBase):

    @staticmethod
    def getExceptionExit():
        if False:
            print('Hello World!')
        return BaseException

    @staticmethod
    def isValueEscaping():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isControlFlowEscape():
        if False:
            print('Hello World!')
        return True

class ControlFlowDescriptionFullEscape(ControlFlowDescriptionBase):

    @staticmethod
    def getExceptionExit():
        if False:
            for i in range(10):
                print('nop')
        return BaseException

    @staticmethod
    def isValueEscaping():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isControlFlowEscape():
        if False:
            print('Hello World!')
        return True

class ControlFlowDescriptionNoEscape(ControlFlowDescriptionBase):

    @staticmethod
    def getExceptionExit():
        if False:
            print('Hello World!')
        return None

    @staticmethod
    def isValueEscaping():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isControlFlowEscape():
        if False:
            i = 10
            return i + 15
        return False

class ControlFlowDescriptionZeroDivisionNoEscape(ControlFlowDescriptionNoEscape):

    @staticmethod
    def getExceptionExit():
        if False:
            i = 10
            return i + 15
        return ZeroDivisionError

class ControlFlowDescriptionValueErrorNoEscape(ControlFlowDescriptionNoEscape):

    @staticmethod
    def getExceptionExit():
        if False:
            print('Hello World!')
        return ValueError

class ControlFlowDescriptionComparisonUnorderable(ControlFlowDescriptionNoEscape):

    @staticmethod
    def getExceptionExit():
        if False:
            for i in range(10):
                print('nop')
        return TypeError

    @staticmethod
    def isUnsupported():
        if False:
            while True:
                i = 10
        return True

class ControlFlowDescriptionFormatError(ControlFlowDescriptionFullEscape):
    pass

class ControlFlowDescriptionOperationUnsupportedBase(ControlFlowDescriptionNoEscape):

    @staticmethod
    def getExceptionExit():
        if False:
            return 10
        return TypeError

    @staticmethod
    def isUnsupported():
        if False:
            return 10
        return True

class ControlFlowDescriptionAddUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionSubUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionMulUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionFloorDivUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionTrueDivUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionOldDivUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionModUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionDivmodUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionPowUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionBitorUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionBitandUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionBitxorUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionLshiftUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionRshiftUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass

class ControlFlowDescriptionMatmultUnsupported(ControlFlowDescriptionOperationUnsupportedBase):
    pass