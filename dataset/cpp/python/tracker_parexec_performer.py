def onValueChange(par, prev):
	if parent.track.par.Freeze.eval():
		parent.track.par.Unfreezesilent.pulse()
	if parent.track.par.Benched.eval():
		parent.track.par.Unbenchsilent.pulse()
	# use par.eval() to get current value
	return
	