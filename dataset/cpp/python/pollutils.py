"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager, DependDict, DependList
import TDFunctions as TDF

class Utils:
	"""
	Utils for SystemFailed Poll Component
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.Questions = ownerComp.op('poll_exp')
		self.Answers = ownerComp.op('answers_dat')

	def UpdateRecord(self):
		qr = int(self.ownerComp.par.Question.eval())
		debug(f'updating {qr}')
		qident = self.Questions.row(qr)[0]
		record = self.ownerComp.op(f'record_{qident}')
		answers = self.Answers.rows()
		record.Write(answers)
		record.Save()