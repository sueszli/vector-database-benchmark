# me - this DAT
# 
# channel - the Channel object which has changed
# sampleIndex - the index of the changed sample
# val - the numeric value of the changed sample
# prev - the previous sample value
# 
# Make sure the corresponding toggle is enabled in the CHOP Execute DAT.

def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	fc = op('freeze_chans')
	freeze = fc['Freeze']
	bench = fc['Benched']
	tid = parent.track.par.Trackid
	score = parent.track.par.Score
	if channel.name == 'Freeze':
		if not bench:
			if val == 0:
				op.Sound.SendFreeze('release',tid,score)
			elif val == 1:
				op.Sound.SendFreeze('start',tid,score)
	else:
		if bench == 0:
			op.Sound.SendBenched('release',tid,score)
		elif bench == 1:
			op.Sound.SendBenched('start',tid,score)
	return
	