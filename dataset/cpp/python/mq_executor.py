# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

def onValueChange(par, prev):
	return

# Called at end of frame with complete list of individual parameter changes.
# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):	
	osc = parent.Guide.op('./oscout')
	gid = parent.mqguide.par.Guideid.eval() -1
	table = op('cue_table')

	for c in changes:
		par = c.par
		prev = c.prev
		# debug(f'{prev} -> {par}')
		if par.name == 'Cue' and int(par.eval()) == 0:
			#turn previous exec off
			cue = str(prev)
			level = 0
			break
		elif par.name == 'Level':
			#change level only
			cue = str(parent.mqguide.par.Cue.eval())
			level = (float(table[str(cue),'Intensity'].val)/100) * float(parent.mqguide.par.Level.eval())
		else:
			#new cue, send full look
			cue = par.eval()
			level = (float(table[str(cue),'Intensity'].val)/100) * float(parent.mqguide.par.Level.eval())
			color = int(op('cue_table')[str(cue),'Color'].val) + gid
			beam = int(op('cue_table')[str(cue),'Beam'].val) + gid
			shutter = int(op('cue_table')[str(cue),'Shutter'].val) + gid
			color_ex = f'/exec/14/{color}'
			beam_ex = f'/exec/12/{beam}'
			shut_ex = f'/exec/12/{shutter}'
			osc.sendOSC(color_ex, [int(100)], useNonStandardTypes=True)
			osc.sendOSC(beam_ex, [int(100)], useNonStandardTypes=True)
			osc.sendOSC(shut_ex, [int(100)], useNonStandardTypes=True)
		pass
	#send level (including activation/deactivation) _once exactly_ per active frame
	# debug(f'{cue} set to {level}')
	act = table[str(cue),'Activation'].val
	act_ex = f'/exec/13/{int(act)+gid}'
	osc.sendOSC(act_ex, [float(level)], useNonStandardTypes=True)
	return

def onPulse(par):
	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	