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

class Slotutils:
	"""
	Slotutils description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.pars = ownerComp.par
		# ownerComp.par.Performer.val = 0
		self.Freezeop = ownerComp.op('freeze')
		self.Freezeop.par.Init.pulse()
		self.Scoreop = ownerComp.op('score')
		self.Recorderop = ownerComp.op('recorder')
		self.Neighboursop = ownerComp.op('neighbours')

	def Reset(self):
		# self.ownerComp.par.Performer.val = 0
		self.ownerComp.par.Unstrike.pulse()
		self.Freezeop.par.Unbenchsilent.pulse()
		self.Freezeop.par.Unfreezesilent.pulse()
		self.Scoreop.par.Roundreset.pulse()
		self.Scoreop.par.Highscorereset.pulse()
		self.Recorderop.par.Resetrecord.pulse()
		pass

	def Flagupdate(self):
		active = bool(self.ownerComp.par.Active.eval())
		performer = bool(self.ownerComp.par.Performer.eval())
		freeze = bool(self.ownerComp.par.Freeze.eval())
		if not active:
			self.SetActive(False)
		elif performer:
			self.SetPerformer()
		elif freeze:
			self.SetFreeze()
		else:
			self.SetParticipant()

	def Resetscore(self):
		if not self.pars.Benched:
			self.Scoreop.par.Roundreset.pulse()
			return self.pars.Score
		return 0

	def Resetrecord(self):
		self.Recorderop.par.Resetrecord.pulse()
		pass

	def Startrecord(self):
		self.Recorderop.par.Recordactive.val = 1
		pass

	def Stoprecord(self):
		self.Recorderop.par.Recordactive.val = 0
		pass

	def SetFreeze(self):
		self.Freezeop.par.Forcefreeze.pulse()
		pass
	
	def Unfreeze(self):
		self.Freezeop.par.Unfreeze.pulse()
		pass