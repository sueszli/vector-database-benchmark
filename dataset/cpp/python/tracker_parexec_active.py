def onValueChange(par, prev):
	track = me.parent.track
	val = par.eval()
	if track.par.Resetonassign.eval():
		track.Reset()
	if val:
		track.par.Timestamp.val = absTime.seconds
	return