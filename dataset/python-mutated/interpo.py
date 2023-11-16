def scale(inVal, inMin, inMax, outMin, outMax):
    if False:
        return 10
    return (inVal - inMin) * (outMax - outMin) / (inMax - inMin) + outMin

def linearInterpo(x1, x2, y1, y2, x):
    if False:
        while True:
            i = 10
    dx = x2 - x1
    dy = y2 - y1
    slope = dy / dx
    tx = x - x1
    return y1 + slope * tx