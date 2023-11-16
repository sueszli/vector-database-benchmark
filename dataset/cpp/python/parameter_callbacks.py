def onValueChange(par, prev):
	return
	
def onValuesChanged(changes):
	return

def onPulse(par):
	rt = me.parent.Roundtimer
	name = par.name
	if name == 'Reinit':
		rt.ReInit()
	elif name == 'Gointro':
		rt.GoIntro()
	elif name == 'Endintro':
		rt.EndIntro()
	elif name == 'Goround':
		rt.GoRound()
	elif name == 'Endround':
		rt.EndRound()
	elif name == 'Gooutro':
		rt.GoOutro()
	elif name == 'Endoutro':
		rt.EndOutro()
	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	