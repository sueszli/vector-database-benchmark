"""
Colour Rendering of Spectra

by John Walker
http://www.fourmilab.ch/

Last updated: March 9, 2003

Converted to Python by Andrew Hutchins, sometime in early
2011.

    This program is in the public domain.
    The modifications are also public domain. (AH)

For complete information about the techniques employed in
this program, see the World-Wide Web document:

    http://www.fourmilab.ch/documents/specrend/

The xyz_to_rgb() function, which was wrong in the original
version of this program, was corrected by:

    Andrew J. S. Hamilton 21 May 1999
    Andrew.Hamilton@Colorado.EDU
    http://casa.colorado.edu/~ajsh/

who also added the gamma correction facilities and
modified constrain_rgb() to work by desaturating the
colour by adding white.

A program which uses these functions to plot CIE
"tongue" diagrams called "ppmcie" is included in
the Netpbm graphics toolkit:

    http://netpbm.sourceforge.net/

(The program was called cietoppm in earlier
versions of Netpbm.)

"""
import math
GAMMA_REC709 = 0
NTSCsystem = {'name': 'NTSC', 'xRed': 0.67, 'yRed': 0.33, 'xGreen': 0.21, 'yGreen': 0.71, 'xBlue': 0.14, 'yBlue': 0.08, 'xWhite': 0.3101, 'yWhite': 0.3163, 'gamma': GAMMA_REC709}
EBUsystem = {'name': 'SUBU (PAL/SECAM)', 'xRed': 0.64, 'yRed': 0.33, 'xGreen': 0.29, 'yGreen': 0.6, 'xBlue': 0.15, 'yBlue': 0.06, 'xWhite': 0.3127, 'yWhite': 0.3291, 'gamma': GAMMA_REC709}
SMPTEsystem = {'name': 'SMPTE', 'xRed': 0.63, 'yRed': 0.34, 'xGreen': 0.31, 'yGreen': 0.595, 'xBlue': 0.155, 'yBlue': 0.07, 'xWhite': 0.3127, 'yWhite': 0.3291, 'gamma': GAMMA_REC709}
HDTVsystem = {'name': 'HDTV', 'xRed': 0.67, 'yRed': 0.33, 'xGreen': 0.21, 'yGreen': 0.71, 'xBlue': 0.15, 'yBlue': 0.06, 'xWhite': 0.3127, 'yWhite': 0.3291, 'gamma': GAMMA_REC709}
CIEsystem = {'name': 'CIE', 'xRed': 0.7355, 'yRed': 0.2645, 'xGreen': 0.2658, 'yGreen': 0.7243, 'xBlue': 0.1669, 'yBlue': 0.0085, 'xWhite': 0.3333333333, 'yWhite': 0.3333333333, 'gamma': GAMMA_REC709}
Rec709system = {'name': 'CIE REC709', 'xRed': 0.64, 'yRed': 0.33, 'xGreen': 0.3, 'yGreen': 0.6, 'xBlue': 0.15, 'yBlue': 0.06, 'xWhite': 0.3127, 'yWhite': 0.3291, 'gamma': GAMMA_REC709}

def upvp_to_xy(up, vp):
    if False:
        print('Hello World!')
    xc = 9 * up / (6 * up - 16 * vp + 12)
    yc = 4 * vp / (6 * up - 16 * vp + 12)
    return (xc, yc)

def xy_toupvp(xc, yc):
    if False:
        return 10
    up = 4 * xc / (-2 * xc + 12 * yc + 3)
    vp = 9 * yc / (-2 * xc + 12 * yc + 3)
    return (up, vp)

def xyz_to_rgb(cs, xc, yc, zc):
    if False:
        i = 10
        return i + 15
    '\n    Given an additive tricolour system CS, defined by the CIE x\n    and y chromaticities of its three primaries (z is derived\n    trivially as 1-(x+y)), and a desired chromaticity (XC, YC,\n    ZC) in CIE space, determine the contribution of each\n    primary in a linear combination which sums to the desired\n    chromaticity.  If the  requested chromaticity falls outside\n    the Maxwell  triangle (colour gamut) formed by the three\n    primaries, one of the r, g, or b weights will be negative.\n\n    Caller can use constrain_rgb() to desaturate an\n    outside-gamut colour to the closest representation within\n    the available gamut and/or norm_rgb to normalise the RGB\n    components so the largest nonzero component has value 1.\n    '
    xr = cs['xRed']
    yr = cs['yRed']
    zr = 1 - (xr + yr)
    xg = cs['xGreen']
    yg = cs['yGreen']
    zg = 1 - (xg + yg)
    xb = cs['xBlue']
    yb = cs['yBlue']
    zb = 1 - (xb + yb)
    xw = cs['xWhite']
    yw = cs['yWhite']
    zw = 1 - (xw + yw)
    rx = yg * zb - yb * zg
    ry = xb * zg - xg * zb
    rz = xg * yb - xb * yg
    gx = yb * zr - yr * zb
    gy = xr * zb - xb * zr
    gz = xb * yr - xr * yb
    bx = yr * zg - yg * zr
    by = xg * zr - xr * zg
    bz = xr * yg - xg * yr
    rw = (rx * xw + ry * yw + rz * zw) / yw
    gw = (gx * xw + gy * yw + gz * zw) / yw
    bw = (bx * xw + by * yw + bz * zw) / yw
    rx = rx / rw
    ry = ry / rw
    rz = rz / rw
    gx = gx / gw
    gy = gy / gw
    gz = gz / gw
    bx = bx / bw
    by = by / bw
    bz = bz / bw
    r = rx * xc + ry * yc + rz * zc
    g = gx * xc + gy * yc + gz * zc
    b = bx * xc + by * yc + bz * zc
    return (r, g, b)

def inside_gamut(r, g, b):
    if False:
        print('Hello World!')
    '\n     Test whether a requested colour is within the gamut\n     achievable with the primaries of the current colour\n     system.  This amounts simply to testing whether all the\n     primary weights are non-negative. */\n    '
    return r >= 0 and g >= 0 and (b >= 0)

def constrain_rgb(r, g, b):
    if False:
        print('Hello World!')
    '\n    If the requested RGB shade contains a negative weight for\n    one of the primaries, it lies outside the colour gamut\n    accessible from the given triple of primaries.  Desaturate\n    it by adding white, equal quantities of R, G, and B, enough\n    to make RGB all positive.  The function returns 1 if the\n    components were modified, zero otherwise.\n    '
    w = -min([0, r, g, b])
    if w > 0:
        r += w
        g += w
        b += w
    return (r, g, b)

def gamma_correct(cs, c):
    if False:
        print('Hello World!')
    "\n    Transform linear RGB values to nonlinear RGB values. Rec.\n    709 is ITU-R Recommendation BT. 709 (1990) ``Basic\n    Parameter Values for the HDTV Standard for the Studio and\n    for International Programme Exchange'', formerly CCIR Rec.\n    709. For details see\n\n       http://www.poynton.com/ColorFAQ.html\n       http://www.poynton.com/GammaFAQ.html\n    "
    gamma = cs.gamma
    if gamma == GAMMA_REC709:
        cc = 0.018
        if c < cc:
            c = (1.099 * math.pow(cc, 0.45) - 0.099) / cc
        else:
            c = 1.099 * math.pow(c, 0.45) - 0.099
    else:
        c = math.pow(c, 1.0 / gamma)
    return c

def gamma_correct_rgb(cs, r, g, b):
    if False:
        while True:
            i = 10
    r = gamma_correct(cs, r)
    g = gamma_correct(cs, g)
    b = gamma_correct(cs, b)
    return (r, g, b)

def norm_rgb(r, g, b):
    if False:
        print('Hello World!')
    '\n    Normalise RGB components so the most intense (unless all\n    are zero) has a value of 1.\n    '
    greatest = max([r, g, b])
    if greatest > 0:
        r /= greatest
        g /= greatest
        b /= greatest
    return (r, g, b)

def spectrum_to_xyz(spec_intens, temp):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the CIE X, Y, and Z coordinates corresponding to\n    a light source with spectral distribution given by  the\n    function SPEC_INTENS, which is called with a series of\n    wavelengths between 380 and 780 nm (the argument is\n    expressed in meters), which returns emittance at  that\n    wavelength in arbitrary units.  The chromaticity\n    coordinates of the spectrum are returned in the x, y, and z\n    arguments which respect the identity:\n\n        x + y + z = 1.\n\n    CIE colour matching functions xBar, yBar, and zBar for\n    wavelengths from 380 through 780 nanometers, every 5\n    nanometers.  For a wavelength lambda in this range::\n\n        cie_colour_match[(lambda - 380) / 5][0] = xBar\n        cie_colour_match[(lambda - 380) / 5][1] = yBar\n        cie_colour_match[(lambda - 380) / 5][2] = zBar\n\n    AH Note 2011: This next bit is kind of irrelevant on modern\n    hardware. Unless you are desperate for speed.\n    In which case don\'t use the Python version!\n\n    To save memory, this table can be declared as floats\n    rather than doubles; (IEEE) float has enough\n    significant bits to represent the values. It\'s declared\n    as a double here to avoid warnings about "conversion\n    between floating-point types" from certain persnickety\n    compilers.\n    '
    cie_colour_match = [[0.0014, 0.0, 0.0065], [0.0022, 0.0001, 0.0105], [0.0042, 0.0001, 0.0201], [0.0076, 0.0002, 0.0362], [0.0143, 0.0004, 0.0679], [0.0232, 0.0006, 0.1102], [0.0435, 0.0012, 0.2074], [0.0776, 0.0022, 0.3713], [0.1344, 0.004, 0.6456], [0.2148, 0.0073, 1.0391], [0.2839, 0.0116, 1.3856], [0.3285, 0.0168, 1.623], [0.3483, 0.023, 1.7471], [0.3481, 0.0298, 1.7826], [0.3362, 0.038, 1.7721], [0.3187, 0.048, 1.7441], [0.2908, 0.06, 1.6692], [0.2511, 0.0739, 1.5281], [0.1954, 0.091, 1.2876], [0.1421, 0.1126, 1.0419], [0.0956, 0.139, 0.813], [0.058, 0.1693, 0.6162], [0.032, 0.208, 0.4652], [0.0147, 0.2586, 0.3533], [0.0049, 0.323, 0.272], [0.0024, 0.4073, 0.2123], [0.0093, 0.503, 0.1582], [0.0291, 0.6082, 0.1117], [0.0633, 0.71, 0.0782], [0.1096, 0.7932, 0.0573], [0.1655, 0.862, 0.0422], [0.2257, 0.9149, 0.0298], [0.2904, 0.954, 0.0203], [0.3597, 0.9803, 0.0134], [0.4334, 0.995, 0.0087], [0.5121, 1.0, 0.0057], [0.5945, 0.995, 0.0039], [0.6784, 0.9786, 0.0027], [0.7621, 0.952, 0.0021], [0.8425, 0.9154, 0.0018], [0.9163, 0.87, 0.0017], [0.9786, 0.8163, 0.0014], [1.0263, 0.757, 0.0011], [1.0567, 0.6949, 0.001], [1.0622, 0.631, 0.0008], [1.0456, 0.5668, 0.0006], [1.0026, 0.503, 0.0003], [0.9384, 0.4412, 0.0002], [0.8544, 0.381, 0.0002], [0.7514, 0.321, 0.0001], [0.6424, 0.265, 0.0], [0.5419, 0.217, 0.0], [0.4479, 0.175, 0.0], [0.3608, 0.1382, 0.0], [0.2835, 0.107, 0.0], [0.2187, 0.0816, 0.0], [0.1649, 0.061, 0.0], [0.1212, 0.0446, 0.0], [0.0874, 0.032, 0.0], [0.0636, 0.0232, 0.0], [0.0468, 0.017, 0.0], [0.0329, 0.0119, 0.0], [0.0227, 0.0082, 0.0], [0.0158, 0.0057, 0.0], [0.0114, 0.0041, 0.0], [0.0081, 0.0029, 0.0], [0.0058, 0.0021, 0.0], [0.0041, 0.0015, 0.0], [0.0029, 0.001, 0.0], [0.002, 0.0007, 0.0], [0.0014, 0.0005, 0.0], [0.001, 0.0004, 0.0], [0.0007, 0.0002, 0.0], [0.0005, 0.0002, 0.0], [0.0003, 0.0001, 0.0], [0.0002, 0.0001, 0.0], [0.0002, 0.0001, 0.0], [0.0001, 0.0, 0.0], [0.0001, 0.0, 0.0], [0.0001, 0.0, 0.0], [0.0, 0.0, 0.0]]
    X = 0
    Y = 0
    Z = 0
    for (i, lamb) in enumerate(range(380, 780, 5)):
        Me = spec_intens(lamb, temp)
        X += Me * cie_colour_match[i][0]
        Y += Me * cie_colour_match[i][1]
        Z += Me * cie_colour_match[i][2]
    XYZ = X + Y + Z
    x = X / XYZ
    y = Y / XYZ
    z = Z / XYZ
    return (x, y, z)

def bb_spectrum(wavelength, bbTemp=5000):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate, by Planck's radiation law, the emittance of a black body\n    of temperature bbTemp at the given wavelength (in metres).  */\n    "
    wlm = wavelength * 1e-09
    return 3.74183e-16 * math.pow(wlm, -5.0) / (math.exp(0.014388 / (wlm * bbTemp)) - 1.0)
    '  Built-in test program which displays the x, y, and Z and RGB\n    values for black body spectra from 1000 to 10000 degrees kelvin.\n    When run, this program should produce the following output:\n\n    Temperature       x      y      z       R     G     B\n    -----------    ------ ------ ------   ----- ----- -----\n       1000 K      0.6528 0.3444 0.0028   1.000 0.007 0.000 (Approximation)\n       1500 K      0.5857 0.3931 0.0212   1.000 0.126 0.000 (Approximation)\n       2000 K      0.5267 0.4133 0.0600   1.000 0.234 0.010\n       2500 K      0.4770 0.4137 0.1093   1.000 0.349 0.067\n       3000 K      0.4369 0.4041 0.1590   1.000 0.454 0.151\n       3500 K      0.4053 0.3907 0.2040   1.000 0.549 0.254\n       4000 K      0.3805 0.3768 0.2428   1.000 0.635 0.370\n       4500 K      0.3608 0.3636 0.2756   1.000 0.710 0.493\n       5000 K      0.3451 0.3516 0.3032   1.000 0.778 0.620\n       5500 K      0.3325 0.3411 0.3265   1.000 0.837 0.746\n       6000 K      0.3221 0.3318 0.3461   1.000 0.890 0.869\n       6500 K      0.3135 0.3237 0.3628   1.000 0.937 0.988\n       7000 K      0.3064 0.3166 0.3770   0.907 0.888 1.000\n       7500 K      0.3004 0.3103 0.3893   0.827 0.839 1.000\n       8000 K      0.2952 0.3048 0.4000   0.762 0.800 1.000\n       8500 K      0.2908 0.3000 0.4093   0.711 0.766 1.000\n       9000 K      0.2869 0.2956 0.4174   0.668 0.738 1.000\n       9500 K      0.2836 0.2918 0.4246   0.632 0.714 1.000\n      10000 K      0.2807 0.2884 0.4310   0.602 0.693 1.000\n'
if __name__ == '__main__':
    print('Temperature       x      y      z       R     G     B\n')
    print('-----------    ------ ------ ------   ----- ----- -----\n')
    for t in range(1000, 10000, 500):
        (x, y, z) = spectrum_to_xyz(bb_spectrum, t)
        (r, g, b) = xyz_to_rgb(SMPTEsystem, x, y, z)
        print('  %5.0f K      %.4f %.4f %.4f   ' % (t, x, y, z))
        (r, g, b) = constrain_rgb(r, g, b)
        (r, g, b) = norm_rgb(r, g, b)
        print('%.3f %.3f %.3f' % (r, g, b))