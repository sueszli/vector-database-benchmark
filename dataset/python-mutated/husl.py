import operator
import math
__version__ = '2.1.0'
m = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.204, 1.057]]
m_inv = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
refX = 0.95047
refY = 1.0
refZ = 1.08883
refU = 0.19784
refV = 0.46834
lab_e = 0.008856
lab_k = 903.3

def husl_to_rgb(h, s, l):
    if False:
        for i in range(10):
            print('nop')
    return lch_to_rgb(*husl_to_lch([h, s, l]))

def husl_to_hex(h, s, l):
    if False:
        while True:
            i = 10
    return rgb_to_hex(husl_to_rgb(h, s, l))

def rgb_to_husl(r, g, b):
    if False:
        return 10
    return lch_to_husl(rgb_to_lch(r, g, b))

def hex_to_husl(hex):
    if False:
        i = 10
        return i + 15
    return rgb_to_husl(*hex_to_rgb(hex))

def huslp_to_rgb(h, s, l):
    if False:
        return 10
    return lch_to_rgb(*huslp_to_lch([h, s, l]))

def huslp_to_hex(h, s, l):
    if False:
        i = 10
        return i + 15
    return rgb_to_hex(huslp_to_rgb(h, s, l))

def rgb_to_huslp(r, g, b):
    if False:
        for i in range(10):
            print('nop')
    return lch_to_huslp(rgb_to_lch(r, g, b))

def hex_to_huslp(hex):
    if False:
        while True:
            i = 10
    return rgb_to_huslp(*hex_to_rgb(hex))

def lch_to_rgb(l, c, h):
    if False:
        while True:
            i = 10
    return xyz_to_rgb(luv_to_xyz(lch_to_luv([l, c, h])))

def rgb_to_lch(r, g, b):
    if False:
        return 10
    return luv_to_lch(xyz_to_luv(rgb_to_xyz([r, g, b])))

def max_chroma(L, H):
    if False:
        for i in range(10):
            print('nop')
    hrad = math.radians(H)
    sinH = math.sin(hrad)
    cosH = math.cos(hrad)
    sub1 = math.pow(L + 16, 3.0) / 1560896.0
    sub2 = sub1 if sub1 > 0.008856 else L / 903.3
    result = float('inf')
    for row in m:
        m1 = row[0]
        m2 = row[1]
        m3 = row[2]
        top = (0.99915 * m1 + 1.05122 * m2 + 1.1446 * m3) * sub2
        rbottom = 0.8633 * m3 - 0.17266 * m2
        lbottom = 0.12949 * m3 - 0.38848 * m1
        bottom = (rbottom * sinH + lbottom * cosH) * sub2
        for t in (0.0, 1.0):
            C = L * (top - 1.05122 * t) / (bottom + 0.17266 * sinH * t)
            if C > 0.0 and C < result:
                result = C
    return result

def _hrad_extremum(L):
    if False:
        i = 10
        return i + 15
    lhs = (math.pow(L, 3.0) + 48.0 * math.pow(L, 2.0) + 768.0 * L + 4096.0) / 1560896.0
    rhs = 1107.0 / 125000.0
    sub = lhs if lhs > rhs else 10.0 * L / 9033.0
    chroma = float('inf')
    result = None
    for row in m:
        for limit in (0.0, 1.0):
            [m1, m2, m3] = row
            top = -3015466475.0 * m3 * sub + 603093295.0 * m2 * sub - 603093295.0 * limit
            bottom = 1356959916.0 * m1 * sub - 452319972.0 * m3 * sub
            hrad = math.atan2(top, bottom)
            if limit == 0.0:
                hrad += math.pi
            test = max_chroma(L, math.degrees(hrad))
            if test < chroma:
                chroma = test
                result = hrad
    return result

def max_chroma_pastel(L):
    if False:
        while True:
            i = 10
    H = math.degrees(_hrad_extremum(L))
    return max_chroma(L, H)

def dot_product(a, b):
    if False:
        for i in range(10):
            print('nop')
    return sum(map(operator.mul, a, b))

def f(t):
    if False:
        return 10
    if t > lab_e:
        return math.pow(t, 1.0 / 3.0)
    else:
        return 7.787 * t + 16.0 / 116.0

def f_inv(t):
    if False:
        i = 10
        return i + 15
    if math.pow(t, 3.0) > lab_e:
        return math.pow(t, 3.0)
    else:
        return (116.0 * t - 16.0) / lab_k

def from_linear(c):
    if False:
        while True:
            i = 10
    if c <= 0.0031308:
        return 12.92 * c
    else:
        return 1.055 * math.pow(c, 1.0 / 2.4) - 0.055

def to_linear(c):
    if False:
        return 10
    a = 0.055
    if c > 0.04045:
        return math.pow((c + a) / (1.0 + a), 2.4)
    else:
        return c / 12.92

def rgb_prepare(triple):
    if False:
        return 10
    ret = []
    for ch in triple:
        ch = round(ch, 3)
        if ch < -0.0001 or ch > 1.0001:
            raise Exception(f'Illegal RGB value {ch:f}')
        if ch < 0:
            ch = 0
        if ch > 1:
            ch = 1
        ret.append(int(round(ch * 255 + 0.001, 0)))
    return ret

def hex_to_rgb(hex):
    if False:
        i = 10
        return i + 15
    if hex.startswith('#'):
        hex = hex[1:]
    r = int(hex[0:2], 16) / 255.0
    g = int(hex[2:4], 16) / 255.0
    b = int(hex[4:6], 16) / 255.0
    return [r, g, b]

def rgb_to_hex(triple):
    if False:
        while True:
            i = 10
    [r, g, b] = triple
    return '#%02x%02x%02x' % tuple(rgb_prepare([r, g, b]))

def xyz_to_rgb(triple):
    if False:
        i = 10
        return i + 15
    xyz = map(lambda row: dot_product(row, triple), m)
    return list(map(from_linear, xyz))

def rgb_to_xyz(triple):
    if False:
        print('Hello World!')
    rgbl = list(map(to_linear, triple))
    return list(map(lambda row: dot_product(row, rgbl), m_inv))

def xyz_to_luv(triple):
    if False:
        while True:
            i = 10
    (X, Y, Z) = triple
    if X == Y == Z == 0.0:
        return [0.0, 0.0, 0.0]
    varU = 4.0 * X / (X + 15.0 * Y + 3.0 * Z)
    varV = 9.0 * Y / (X + 15.0 * Y + 3.0 * Z)
    L = 116.0 * f(Y / refY) - 16.0
    if L == 0.0:
        return [0.0, 0.0, 0.0]
    U = 13.0 * L * (varU - refU)
    V = 13.0 * L * (varV - refV)
    return [L, U, V]

def luv_to_xyz(triple):
    if False:
        i = 10
        return i + 15
    (L, U, V) = triple
    if L == 0:
        return [0.0, 0.0, 0.0]
    varY = f_inv((L + 16.0) / 116.0)
    varU = U / (13.0 * L) + refU
    varV = V / (13.0 * L) + refV
    Y = varY * refY
    X = 0.0 - 9.0 * Y * varU / ((varU - 4.0) * varV - varU * varV)
    Z = (9.0 * Y - 15.0 * varV * Y - varV * X) / (3.0 * varV)
    return [X, Y, Z]

def luv_to_lch(triple):
    if False:
        return 10
    (L, U, V) = triple
    C = math.pow(math.pow(U, 2) + math.pow(V, 2), 1.0 / 2.0)
    hrad = math.atan2(V, U)
    H = math.degrees(hrad)
    if H < 0.0:
        H = 360.0 + H
    return [L, C, H]

def lch_to_luv(triple):
    if False:
        return 10
    (L, C, H) = triple
    Hrad = math.radians(H)
    U = math.cos(Hrad) * C
    V = math.sin(Hrad) * C
    return [L, U, V]

def husl_to_lch(triple):
    if False:
        print('Hello World!')
    (H, S, L) = triple
    if L > 99.9999999:
        return [100, 0.0, H]
    if L < 1e-08:
        return [0.0, 0.0, H]
    mx = max_chroma(L, H)
    C = mx / 100.0 * S
    return [L, C, H]

def lch_to_husl(triple):
    if False:
        print('Hello World!')
    (L, C, H) = triple
    if L > 99.9999999:
        return [H, 0.0, 100.0]
    if L < 1e-08:
        return [H, 0.0, 0.0]
    mx = max_chroma(L, H)
    S = C / mx * 100.0
    return [H, S, L]

def huslp_to_lch(triple):
    if False:
        for i in range(10):
            print('nop')
    (H, S, L) = triple
    if L > 99.9999999:
        return [100, 0.0, H]
    if L < 1e-08:
        return [0.0, 0.0, H]
    mx = max_chroma_pastel(L)
    C = mx / 100.0 * S
    return [L, C, H]

def lch_to_huslp(triple):
    if False:
        while True:
            i = 10
    (L, C, H) = triple
    if L > 99.9999999:
        return [H, 0.0, 100.0]
    if L < 1e-08:
        return [H, 0.0, 0.0]
    mx = max_chroma_pastel(L)
    S = C / mx * 100.0
    return [H, S, L]