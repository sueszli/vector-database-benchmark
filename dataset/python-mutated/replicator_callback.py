def onRemoveReplicant(comp, replicant):
    if False:
        i = 10
        return i + 15
    replicant.destroy()
    return

def onReplicate(comp, allOps, newOps, template, master):
    if False:
        while True:
            i = 10
    pm = op('parameter_merge')
    rm = op('record_merge')
    lm = op('lines_merge')
    nm = op('neighbours_merge')
    for c in newOps:
        tid = c.digits
        c.par.Trackid.val = tid
        c.par.Timestamp.val = absTime.seconds
        pass
    return