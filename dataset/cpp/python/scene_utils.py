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
		self.pars = ownerComp.par
		self.Grabs = ownerComp.ops('*readlive')
		self.Writes = ownerComp.ops('*writelive')
		self.Files = ownerComp.ops('*file')

	def WriteSnapshot(self, components = []):
		folder = self.pars.Livefolder.eval()
		mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
		if len(components):
			for comp in components:
				fop = op(f'{comp}_writelive')
				handle = fop.par.file.eval()
				fop.save(handle)
				# fop.save(fop.par.file.val, createFolders=True)
		else:
			for fop in self.Writes:
				# fop.save(fop.par.file.val, createFolders=True)
				fop.par.write.pulse()
				handle = fop.par.file.eval()
				fop.save(handle)

	def Write(self, components = []):
		folder = self.pars.Livefolder.eval()
		mod.os.makedirs(f'{project.folder}/{folder}', exist_ok=True)
		if len(components):
			for comp in components:
				fop = op(f'{comp}_file')
				# fop.save(fop.par.file.val, createFolders=True)
				handle = fop.par.file.eval()
				fop.save(handle)
		else:
			for fop in self.Files:
				handle = fop.par.file.eval()
				fop.save(handle)
				# fop.save(fop.par.file.val, createFolders=True)

	def Load(self, components = []):
		if len(components):
			for comp in components:
				op(f'{comp}_file').par.loadonstartpulse.pulse()
		else:
			for fop in self.Files:
				fop.par.loadonstartpulse.pulse()

	def Go(self, components = []):
		if len(components):
			for comp in components:
				# e.g. 'guide_file' -> 'guide'
				op(comp).copy(f'{comp}_file')
		else:
			for fop in self.Files:
				go = self.ownerComp.op(fop.name.split('_')[0])
				op(go).copy(fop)
