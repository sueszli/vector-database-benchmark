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
	Utils description
	"""

    def __init__(self, ownerComp):
        if False:
            i = 10
            return i + 15
        self.ownerComp = ownerComp
        self.pars = ownerComp.par
        self.Grabs = ownerComp.ops('*readlive')
        self.Writes = ownerComp.ops('*writelive')
        self.Files = ownerComp.ops('*file')

    def WriteSnapshot(self, components=[]):
        if False:
            return 10
        folder = self.pars.Livefolder.eval()
        mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
        if len(components):
            for comp in components:
                fop = op(f'{comp}_writelive')
                handle = fop.par.file.eval()
                fop.save(handle)
        else:
            for fop in self.Writes:
                fop.par.write.pulse()
                handle = fop.par.file.eval()
                fop.save(handle)

    def Write(self, components=[]):
        if False:
            print('Hello World!')
        folder = self.pars.Livefolder.eval()
        mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
        if len(components):
            for comp in components:
                fop = op(f'{comp}_file')
                handle = fop.par.file.eval()
                fop.save(handle)
        else:
            for fop in self.Files:
                handle = fop.par.file.eval()
                fop.save(handle)

    def Load(self, components=[]):
        if False:
            return 10
        if len(components):
            for comp in components:
                op(f'{comp}_file').par.loadonstartpulse.pulse()
        else:
            for fop in self.Files:
                fop.par.loadonstartpulse.pulse()

    def Go(self, components=[]):
        if False:
            for i in range(10):
                print('nop')
        if len(components):
            for comp in components:
                op(comp).copy(f'{comp}_file')
        else:
            for fop in self.Files:
                go = self.ownerComp.op(fop.name.split('_')[0])
                op(go).copy(fop)