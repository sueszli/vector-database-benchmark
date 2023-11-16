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
		# The component to which this extension is attached
		self.ownerComp = ownerComp

	def Update(self):
		op('traillock').lock = 0
		run("op('traillock').lock = 1", delayFrames = 1)
		run("parent.recorder.Save()", delayFrames = (1+(parent.track.par.Trackid.eval()%10)))
		return

	def Save(self):
		mtime = mod.time
		lt = mod.time.localtime
		lt_str = mod.time.strftime("%d%m%y_%H%M")
		folder = parent.track.par.Recorddir.eval()
		mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
		trailop = op('roundtrail_exp')
		scalarop = op('scalars_exp')
		fileprefix = f"{lt_str}_{op.Control.par.Ident.eval()}"
		trailhandle = f"{folder}/{fileprefix}_trails.bclip"
		scalarhandle = f"{folder}/{fileprefix}_scalars.bclip"		
		trailop.save(trailhandle, createFolders=True)
		scalarop.save(scalarhandle, createFolders=True)
		return