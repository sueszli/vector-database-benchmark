# me - this DAT
# 
# comp - the replicator component which is cooking
# allOps - a list of all replicants, created or existing
# newOps - the subset that were just created
# template - table DAT specifying the replicator attributes
# master - the master operator
#

def onRemoveReplicant(comp, replicant):
	replicant.destroy()
	return

def onReplicate(comp, allOps, newOps, template, master):
	pm = op('parameter_merge')
	rm = op('record_merge')
	lm = op('lines_merge')
	nm = op('neighbours_merge')
	# em = op('export_merge')
	for c in newOps:
		tid = c.digits
		c.par.Trackid.val = tid
		c.par.Timestamp.val = absTime.seconds
		# c.outputConnectors[0].connect(pm.inputConnectors[tid])
		# c.outputConnectors[1].connect(rm.inputConnectors[tid])
		# c.outputConnectors[2].connect(lm.inputConnectors[tid])
		# c.outputConnectors[3].connect(nm.inputConnectors[tid])
		# c.outputConnectors[4].connect(em.inputConnectors[tid])
		pass
	return
