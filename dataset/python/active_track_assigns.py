# me - this DAT
# 
# channel - the Channel object which has changed
# sampleIndex - the index of the changed sample
# val - the numeric value of the changed sample
# prev - the previous sample value
# 
# Make sure the corresponding toggle is enabled in the CHOP Execute DAT.

def onOffToOn(channel, sampleIndex, val, prev):
	debug(f'ON: channel: {channel.name} - value: {val}')
	op.Tracker.Assign(int(channel.name))
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	debug(f'OFF: channel: {channel.name} - value: {val}')
	op.Tracker.Unassign(int(channel.name))
	return

def whileOff(channel, sampleIndex, val, prev):
	op.Tracker.Assign(int(channel.name))
	return

def onValueChange(channel, sampleIndex, val, prev):
	val = int(val)
	if val:
		op.Tracker.Assign(int(channel.name))
	else:
		op.Tracker.Unassign(int(channel.name))
	return
	