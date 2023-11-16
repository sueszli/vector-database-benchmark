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
	if (not parent.freeze.par.Grace.eval()) and (not parent.freeze.par.Benched.eval()):
		ft = op('freezetimer')
		ft.par.gotodone.pulse()
		ft.par.play = 1
		run(op('script_freezesound')[0,0].val, delayFrames = 2)
		run(op('script_freezeposition')[0,0].val, delayFrames = 2)
		op('violationgrace').par.start.pulse()
	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	