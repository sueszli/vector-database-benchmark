# me - this DAT
# 
# frame - the current frame
# state - True if the timeline is paused
# 
# Make sure the corresponding toggle is enabled in the Execute DAT.

def onStart():
	return

def onCreate():
	return

def onExit():
	return

def onFrameStart(frame):
	strobechop = op('strobes')
	strobes = []
	numS = strobechop.numSamples
	for k in range(numS):
		tid = int(strobechop['Trackid'][k])
		act = strobechop['Freezeviolation'][k]
		tx = strobechop['Positionx'][k]
		ty = strobechop['Positiony'][k]
		if act:
			strobes.append([tid, tx, ty])
	parent.Sound.SendStrobes(strobes)
	return

def onFrameEnd(frame):
	return

def onPlayStateChange(state):
	return

def onDeviceChange():
	return

def onProjectPreSave():
	return

def onProjectPostSave():
	return

	