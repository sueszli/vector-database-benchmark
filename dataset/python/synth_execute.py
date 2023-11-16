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
	# self.synthSet = [i for i in range(1,51)]
	synth = op('synth_set_dat')
	args = []
	for i in range(1,51): 
		# self.synthIndex = ((self.synthIndex + 1) % 50)
		pitch = int(synth[i,'Trackid'].val)
		args.append(pitch)
		level = float(synth[i,'Level'].val or 0)
		args.append(level)
		posx = float(synth[i,'Positionx'].val or 0)
		args.append(posx)
		posy = float(synth[i,'Positiony'].val or 0)
		args.append(posy)
		# op.Sound.SendSynthSingle(pitch, level, posx, posy)
		# debug(self.synthIndex+1)
	op.Sound.SendSynth(f'/synth', args)
	# op.Sound.SendSynthCycle()
	# op.Sound.SendSynth(pitch,level,posx,posy)
	return

def onPlayStateChange(state):
	return

def onDeviceChange():
	return

def onProjectPreSave():
	return

def onProjectPostSave():
	return
