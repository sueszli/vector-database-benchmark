def onValueChange(par, prev):
    if False:
        i = 10
        return i + 15
    if parent.track.par.Freeze.eval():
        parent.track.par.Unfreezesilent.pulse()
    if parent.track.par.Benched.eval():
        parent.track.par.Unbenchsilent.pulse()
    return