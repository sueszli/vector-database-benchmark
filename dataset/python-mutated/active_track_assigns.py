def onOffToOn(channel, sampleIndex, val, prev):
    if False:
        return 10
    debug(f'ON: channel: {channel.name} - value: {val}')
    op.Tracker.Assign(int(channel.name))
    return

def whileOn(channel, sampleIndex, val, prev):
    if False:
        for i in range(10):
            print('nop')
    return

def onOnToOff(channel, sampleIndex, val, prev):
    if False:
        while True:
            i = 10
    debug(f'OFF: channel: {channel.name} - value: {val}')
    op.Tracker.Unassign(int(channel.name))
    return

def whileOff(channel, sampleIndex, val, prev):
    if False:
        return 10
    op.Tracker.Assign(int(channel.name))
    return

def onValueChange(channel, sampleIndex, val, prev):
    if False:
        print('Hello World!')
    val = int(val)
    if val:
        op.Tracker.Assign(int(channel.name))
    else:
        op.Tracker.Unassign(int(channel.name))
    return