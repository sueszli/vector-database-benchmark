"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""
from TDStoreTools import StorageManager
import TDFunctions as TDF

class Utils:
    """
	Utils for SystemFailed Poll/Record Component
	"""

    def __init__(self, ownerComp):
        if False:
            while True:
                i = 10
        self.ownerComp = ownerComp
        self.recordDat = ownerComp.op('./data')

    def Write(self, answerlist):
        if False:
            i = 10
            return i + 15
        rd = self.recordDat
        rd.clear()
        rd.appendRow(['Trackid', 'Answer'])
        for ans in answerlist:
            tid = ans[0]
            val = ans[1]
            rd.appendRow([tid, val])

    def Clear(self):
        if False:
            for i in range(10):
                print('nop')
        rd = self.recordDat
        rd.clear()
        rd.appendRow(['Trackid', 'Answer'])

    def Save(self):
        if False:
            return 10
        mtime = mod.time
        lt = mod.time.localtime
        lt_str = mod.time.strftime('%d%m%y_%H%M')
        filename = f'{lt_str}_{self.ownerComp.name}.csv'
        folder = parent.Poll.par.Recorddir.eval()
        mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
        fop = op(f'writeout_null')
        handle = f'{folder}/{filename}'
        fop.save(handle, createFolders=True)