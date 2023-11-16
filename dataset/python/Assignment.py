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

class Assignment:
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.TrackLimitPar = op.Settings.par.Maxtracks
		self.PerfLimitPar = op.Settings.par.Maxperformers
		# self.Unassigned = [x for x in self.TrackSlotIds if not x == 0]
		self.TrackUnassigned = set(range(self.PerfLimitPar.eval() + 1, self.PerfLimitPar.eval() + self.TrackLimitPar.eval() + 1))
		self.TrackAssigned = dict()
		self.TrackSlotIds = self.TrackAssigned.values()
		self.PerformUnassigned = set(range(1,self.PerfLimitPar.eval() + 1))
		self.PerformAssigned = dict()
		self.PerformSlotIds = self.TrackAssigned.values()
		self.PidTable = op('pharus_ids')
		self.AssignmentTable = op('assignment_new')

	"""
	Update to a given pharus id set without changing existing assigns
	"""
	def PidUpdate(self, pharusIds):
		# clear
		trackPids = self.TrackAssigned.copy()
		for assigned in trackPids:
			if not assigned in pharusIds:
				self.TPUnassign(assigned)
		performPids = self.PerformAssigned.copy()
		for assigned in performPids:
			if not assigned in pharusIds:
				self.TPUnassign(assigned)
		for pid in pharusIds:
			if (int(pid.val) in self.TrackAssigned) or (int(pid.val) in self.PerformAssigned):
				pass
			else:
				self.TrackAssign(int(pid.val))

	"""
	Clear Assignments and recreate them based on current table
	"""
	def FullReassign(self):
		# debug("reassign")
		self.TrackUnassigned = set(range(self.PerfLimitPar.eval() + 1, self.PerfLimitPar.eval() + self.TrackLimitPar.eval() + 1))
		self.TrackAssigned = dict()
		self.TrackSlotIds = self.TrackAssigned.values()
		self.PerformUnassigned = set(range(1,self.PerfLimitPar.eval() + 1))
		self.PerformAssigned = dict()
		self.PerformSlotIds = self.PerformAssigned.values()
		self.AssignmentTable.clear(keepSize=True, keepFirstRow=True, keepFirstCol=True)
		# for row in self.AssignmentTable.rows():
			# row[1].val = '0'
		for cell in self.PidTable.col(0):
			pid = cell.val
			self.TrackAssign(pid)

	"""
	Set up an assignment and publish it
	"""
	def TrackAssign(self, pid):
		pid = int(pid)
		if pid in self.TrackAssigned:
			return
		if pid in self.PerformAssigned:
			return
		if len(self.TrackUnassigned) <= 0:
			debug(f"unable to add tracker for pharus id {pid} - no free assignment")
			return
		else:
			slot = self.TrackUnassigned.pop()
			self.TrackAssigned[pid] = slot
			self.AssignmentTable[f'{slot}','Pharusid'].val = pid
			return slot

	def PerformAssign(self, pid, slot):
		pid = int(pid)
		# if pid in self.PerformAssigned:
			# return
		if not (slot <= self.PerfLimitPar.eval()):
			return
		try:
			self.PerformUnassigned.remove(slot)
		except KeyError:
			pass
		finally:
			oldAssign = self.PerformAssigned.copy()
			for k in oldAssign:
				if self.PerformAssigned[k] == slot:
					self.PerformAssigned.pop(k)
			self.PerformAssigned[pid] = slot
			if pid in self.TrackAssigned:
				self.TPUnassign(pid)
			self.AssignmentTable[f'{slot}','Pharusid'].val = pid
			self.PidUpdate(self.PidTable.col(0))
			return slot

	"""
	Delete an assignment and make the slot available again
	"""
	def TPUnassign(self, pid):
		pid = int(pid)
		slot = None
		if pid in self.TrackAssigned:
			slot = self.TrackAssigned.pop(pid)
			self.TrackUnassigned.add(slot)
		elif pid in self.PerformAssigned:
			slot = self.PerformAssigned.pop(pid)
			self.PerformUnassigned.add(slot)
		if slot != None:
			self.AssignmentTable[f'{slot}','Pharusid'].val = '0'
		return slot

	def PharusSwitch(self, first, second):
		slot1 = None
		slot2 = None
		performer1 = False
		performer2 = False
	
		if first in self.TrackAssigned:
			slot1 = self.TrackAssigned[first]
		elif first in self.PerformAssigned:
			slot1 = self.PerformAssigned[first]
			performer1 = True
		if second in self.TrackAssigned:
			slot2 = self.TrackAssigned[second]
		elif second in self.PerformAssigned:
			slot2 = self.PerformAssigned[second]
			performer2 = True
		# if (not slot1) or (not slot2):
			# return
		if slot1:
			if performer2:
				self.PerformAssigned[second] = slot1
			else:
				self.TrackAssigned[second] = slot1
		else:
			if performer2:
				free = self.PerformAssigned.pop(second)
				self.PerformUnassigned.add(free)
			else:
				free = self.TrackAssigned.pop(second)
				self.TrackUnassigned.add(free)
		if slot2:
			if performer1:
				self.PerformAssigned[first] = slot2
			else:
				self.TrackAssigned[first] = slot2
		else:
			if performer1:
				free = self.PerformAssigned.pop(first)
				self.PerformUnassigned.add(free)
			else:
				free = self.TrackAssigned.pop(first)
				self.TrackUnassigned.add(free)

	def UpdateTable(self):
		maxSlot = self.PerfLimitPar.eval() + self.TrackLimitPar.eval() + 1
		self.AssignmentTable.setSize(maxSlot, 2)
		self.AssignmentTable.clear(keepSize=True, keepFirstRow=True, keepFirstCol=True)
		for i in range(1, maxSlot):
			self.AssignmentTable[i,'Trackid'].val = i
		for pid in self.PerformAssigned:
			slotid = self.PerformAssigned[pid]
			self.AssignmentTable[slotid, 'Pharusid'].val = pid
		for pid in self.TrackAssigned:
			slotid = self.TrackAssigned[pid]
			self.AssignmentTable[slotid, 'Pharusid'].val = pid			
