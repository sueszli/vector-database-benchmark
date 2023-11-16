# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

def onValueChange(par, prev):
	# use par.eval() to get current value
	return

# Called at end of frame with complete list of individual parameter changes.
# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):
	for c in changes:
		# use par.eval() to get current value
		par = c.par
		prev = c.prev
	return

def onPulse(par):
	if par.name == 'Initround':
		parent().InitRound()
	elif par.name == 'Startround':
		parent().StartRound()
	elif par.name == 'Pauseround':
		parent().PauseRound()
	elif par.name == 'Resumeround':
		parent().ResumeRound()
	elif par.name == 'Stopround':
		parent().StopRound()
	elif par.name == 'Evaluateround':
		parent().EvaluateRound()
	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	