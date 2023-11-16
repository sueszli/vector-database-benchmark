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
	zapchop = op('zaps')
	zaps = []
	numZ = zapchop.numSamples
	for k in range(numZ):
		tid = int(zapchop['Trackid'][k])
		tx = zapchop['Positionx'][k]
		ty = zapchop['Positiony'][k]
		zaps.append([tid, tx, ty])
	parent.Sound.SendZaps(zaps)
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

	