"""
This extension provides utility functions for the trackers
	- id and status assignment
"""

from TDStoreTools import StorageManager, DependList, DependDict

import TDFunctions as TDF

class Utils:
	"""
	Utils for SystemFailed Pharus Interface
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.MaxSlots = op.Settings.par.Maxtracks
		self.Pharus = op('ordered_pharus_set')
		self.Unassigned = set(range(1, 1 + self.MaxSlots))
		self.Assignment = dict()
		self.AssignmentTable = op('assign_table')

	"""
	Update to a given pharus id set without changing existing assigns
	"""
	def Update(self, ids):
		tmp = self.Assignment.copy()
		for aid in tmp:
			if not aid in ids:
				self.Unassign(aid)
		for pid in ids:
			parent().Assign(pid)

	"""
	Clear Assignments and recreate them based on current table
	"""
	def Reassign(self):
		# debug("reassign")
		self.Unassigned = set(range(1, 1 + self.MaxSlots))
		self.Assignment = dict()
		self.AssignmentTable.setSize(self.MaxSlots,1)
		for row in self.AssignmentTable.rows():
			row[0].val = '0'
		for cell in self.Pharus.col(0):
			pid = cell.val
			self.Assign(pid)

	"""
	Set up an assignment and publish it
	"""
	def Assign(self, pid):
		pid = int(pid)
		# debug(f'{me.name}: assign {pid}')
		slot = self.Assignment.get(pid)
		if not slot:
			if len(self.Unassigned) <= 0:
				debug(f"unable to add tracker for pharus id {pid} - no free assignment")
				slot = None
			else:
				slot = self.Unassigned.pop()
				self.Assignment[pid] = slot
				self.AssignmentTable[slot-1,0].val = pid
		return slot

	"""
	Delete an assignment and make the slot available again
	"""
	def Unassign(self, pid):
		pid = int(pid)
		try:
			slot = self.Assignment.pop(pid)
		except KeyError:
			slot = None
			# debug(f"Unassign {pid} failed - assignment not found")
		else:
			self.AssignmentTable[slot-1,0].val = '0'
			self.Unassigned.add(slot)
			# debug(f'Unassigned {pid} from {slot}')
		finally:
			return slot
