from colorsys import hls_to_rgb, rgb_to_hls
try:
    from md5 import md5
except ImportError:
    from hashlib import md5
import sys
from .colortrans import *
from .utils import py3

def getOppositeColor(r, g, b):
    if False:
        while True:
            i = 10
    (r, g, b) = [x / 255.0 for x in [r, g, b]]
    hls = rgb_to_hls(r, g, b)
    opp = list(hls[:])
    opp[0] = (opp[0] + 0.2) % 1
    if opp[1] > 255 / 2:
        opp[1] -= 255 / 2
    else:
        opp[1] += 255 / 2
    if opp[2] > -0.5:
        opp[2] -= 0.5
    opp = hls_to_rgb(*opp)
    m = max(opp)
    if m > 255:
        opp = [x * 254 / m for x in opp]
    return tuple([int(x) for x in opp])

def stringToHashToColorAndOpposite(string):
    if False:
        return 10
    if py3:
        string = string.encode('utf-8')
    string = md5(string).hexdigest()[:6]
    color1 = rgbstring2tuple(string)
    color2 = getOppositeColor(*color1)
    return (color1, color2)