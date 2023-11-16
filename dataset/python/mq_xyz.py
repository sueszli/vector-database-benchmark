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
	return

def onFrameEnd(frame):
	mqSender = parent.Guide.op('./mqout')
	pars = parent.mqguide.par
	tid = int(pars.Trackid.eval())
	gid = int(pars.Guideid.eval())-1
	x = float(pars.Positionx.eval())
	y = -float(pars.Positiony.eval())
	z = float(pars.Height.eval())
	msg = f'{x:.2f},{z:.2f},{y:.2f},{gid},Tracker:{tid}'
	mqSender.send(msg)
	return

def onPlayStateChange(state):
	return

def onDeviceChange():
	return

def onProjectPreSave():
	return

def onProjectPostSave():
	return

	