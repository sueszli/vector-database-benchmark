import pythoncom
from win32com.server import exception, util
VT_EMPTY = pythoncom.VT_EMPTY

class Bag:
    _public_methods_ = ['Read', 'Write']
    _com_interfaces_ = [pythoncom.IID_IPropertyBag]

    def __init__(self):
        if False:
            while True:
                i = 10
        self.data = {}

    def Read(self, propName, varType, errorLog):
        if False:
            return 10
        print('read: name=', propName, 'type=', varType)
        if propName not in self.data:
            if errorLog:
                hr = 2147942487
                exc = pythoncom.com_error(0, 'Bag.Read', 'no such item', None, 0, hr)
                errorLog.AddError(propName, exc)
            raise exception.Exception(scode=hr)
        return self.data[propName]

    def Write(self, propName, value):
        if False:
            for i in range(10):
                print('nop')
        print('write: name=', propName, 'value=', value)
        self.data[propName] = value

class Target:
    _public_methods_ = ['GetClassID', 'InitNew', 'Load', 'Save']
    _com_interfaces_ = [pythoncom.IID_IPersist, pythoncom.IID_IPersistPropertyBag]

    def GetClassID(self):
        if False:
            i = 10
            return i + 15
        raise exception.Exception(scode=2147500037)

    def InitNew(self):
        if False:
            i = 10
            return i + 15
        pass

    def Load(self, bag, log):
        if False:
            return 10
        print(bag.Read('prop1', VT_EMPTY, log))
        print(bag.Read('prop2', VT_EMPTY, log))
        try:
            print(bag.Read('prop3', VT_EMPTY, log))
        except exception.Exception:
            pass

    def Save(self, bag, clearDirty, saveAllProps):
        if False:
            i = 10
            return i + 15
        bag.Write('prop1', 'prop1.hello')
        bag.Write('prop2', 'prop2.there')

class Log:
    _public_methods_ = ['AddError']
    _com_interfaces_ = [pythoncom.IID_IErrorLog]

    def AddError(self, propName, excepInfo):
        if False:
            for i in range(10):
                print('nop')
        print('error: propName=', propName, 'error=', excepInfo)

def test():
    if False:
        return 10
    bag = Bag()
    target = Target()
    log = Log()
    target.Save(bag, 1, 1)
    target.Load(bag, log)
    comBag = util.wrap(bag, pythoncom.IID_IPropertyBag)
    comTarget = util.wrap(target, pythoncom.IID_IPersistPropertyBag)
    comLog = util.wrap(log, pythoncom.IID_IErrorLog)
    comTarget.Save(comBag, 1, 1)
    comTarget.Load(comBag, comLog)
if __name__ == '__main__':
    test()