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
            print('Hello World!')
        self.ownerComp = ownerComp

    def Start(self):
        if False:
            print('Hello World!')
        parent.groupprofile.par.Recordactive.val = 1
        return

    def Stop(self):
        if False:
            i = 10
            return i + 15
        parent.groupprofile.par.Recordactive.val = 0
        return

    def Reset(self):
        if False:
            return 10
        op('grouptrail').par.reset.pulse()
        return

    def Update(self):
        if False:
            while True:
                i = 10
        op('traillock').lock = 0
        run("op('traillock').lock = 1", delayFrames=2)
        run('parent.groupprofile.Save()', delayFrames=3)
        return

    def Save(self):
        if False:
            for i in range(10):
                print('nop')
        lt = mod.time.localtime
        lt_str = mod.time.strftime('%d%m%y_%H%M')
        folder = parent.groupprofile.par.Recorddir.eval()
        mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
        trailop = op('trails_exp')
        scalarop = op('scalars_exp')
        fileprefix = f'{lt_str}_{op.Control.par.Ident.eval()}'
        trailhandle = f'{folder}/{fileprefix}_trails.bclip'
        scalarhandle = f'{folder}/{fileprefix}_scalars.bclip'
        trailop.save(trailhandle, createFolders=True)
        scalarop.save(scalarhandle, createFolders=True)
        return