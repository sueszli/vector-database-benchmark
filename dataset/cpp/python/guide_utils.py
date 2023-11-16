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
		self.MainSender = op('sender_magicq')
		self.TestSender = op('sender_magicq_debug')


	def SendCue(self, cue):
		numbers = str(cue).split(".")
		if len(numbers) > 2:
			pass
		else:
			message = f'/rpc/8,{cue}J'
			args = [int(1)]
			self.MainSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
			self.TestSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)

	def SendEvent(self, event):
		message = f'{event}'
		args = [int(1)]
		self.MainSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		self.TestSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)

	def SendJoker(self, toggle=1):
		message = f'/joker'
		args = [float(toggle)]
		self.MainSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
		self.TestSender.sendOSC(message, args, asBundle=False, useNonStandardTypes=True)
