import numpy as np
LOWER_BOUND = np.sqrt(np.finfo(float).eps)

class HoltWintersArgs:

    def __init__(self, xi, p, bounds, y, m, n, transform=False):
        if False:
            return 10
        self._xi = xi
        self._p = p
        self._bounds = bounds
        self._y = y
        self._lvl = np.empty(n)
        self._b = np.empty(n)
        self._s = np.empty(n + m - 1)
        self._m = m
        self._n = n
        self._transform = transform

    @property
    def xi(self):
        if False:
            while True:
                i = 10
        return self._xi

    @xi.setter
    def xi(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._xi = value

    @property
    def p(self):
        if False:
            i = 10
            return i + 15
        return self._p

    @property
    def bounds(self):
        if False:
            i = 10
            return i + 15
        return self._bounds

    @property
    def y(self):
        if False:
            i = 10
            return i + 15
        return self._y

    @property
    def lvl(self):
        if False:
            return 10
        return self._lvl

    @property
    def b(self):
        if False:
            return 10
        return self._b

    @property
    def s(self):
        if False:
            while True:
                i = 10
        return self._s

    @property
    def m(self):
        if False:
            return 10
        return self._m

    @property
    def n(self):
        if False:
            for i in range(10):
                print('nop')
        return self._n

    @property
    def transform(self):
        if False:
            return 10
        return self._transform

    @transform.setter
    def transform(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._transform = value

def to_restricted(p, sel, bounds):
    if False:
        for i in range(10):
            print('nop')
    '\n    Transform parameters from the unrestricted [0,1] space\n    to satisfy both the bounds and the 2 constraints\n    beta <= alpha and gamma <= (1-alpha)\n\n    Parameters\n    ----------\n    p : ndarray\n        The parameters to transform\n    sel : ndarray\n        Array indicating whether a parameter is being estimated. If not\n        estimated, not transformed.\n    bounds : ndarray\n        2-d array of bounds where bound for element i is in row i\n        and stored as [lb, ub]\n\n    Returns\n    -------\n\n    '
    (a, b, g) = p[:3]
    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = lb + a * (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(a, bounds[1, 1])
        b = lb + b * (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - a, bounds[2, 1])
        g = lb + g * (ub - lb)
    return (a, b, g)

def to_unrestricted(p, sel, bounds):
    if False:
        print('Hello World!')
    '\n    Transform parameters to the unrestricted [0,1] space\n\n    Parameters\n    ----------\n    p : ndarray\n        Parameters that strictly satisfy the constraints\n\n    Returns\n    -------\n    ndarray\n        Parameters all in (0,1)\n    '
    (a, b, g) = p[:3]
    if sel[0]:
        lb = max(LOWER_BOUND, bounds[0, 0])
        ub = min(1 - LOWER_BOUND, bounds[0, 1])
        a = (a - lb) / (ub - lb)
    if sel[1]:
        lb = bounds[1, 0]
        ub = min(p[0], bounds[1, 1])
        b = (b - lb) / (ub - lb)
    if sel[2]:
        lb = bounds[2, 0]
        ub = min(1.0 - p[0], bounds[2, 1])
        g = (g - lb) / (ub - lb)
    return (a, b, g)

def holt_init(x, hw_args: HoltWintersArgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Initialization for the Holt Models\n    '
    hw_args.p[hw_args.xi.astype(bool)] = x
    if hw_args.transform:
        (alpha, beta, _) = to_restricted(hw_args.p, hw_args.xi, hw_args.bounds)
    else:
        (alpha, beta) = hw_args.p[:2]
    (l0, b0, phi) = hw_args.p[3:6]
    alphac = 1 - alpha
    betac = 1 - beta
    y_alpha = alpha * hw_args.y
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0
    return (alpha, beta, phi, alphac, betac, y_alpha)

def holt__(x, hw_args: HoltWintersArgs):
    if False:
        while True:
            i = 10
    '\n    Simple Exponential Smoothing\n    Minimization Function\n    (,)\n    '
    (_, _, _, alphac, _, y_alpha) = holt_init(x, hw_args)
    n = hw_args.n
    lvl = hw_args.lvl
    for i in range(1, n):
        lvl[i] = y_alpha[i - 1] + alphac * lvl[i - 1]
    return hw_args.y - lvl

def holt_mul_dam(x, hw_args: HoltWintersArgs):
    if False:
        while True:
            i = 10
    '\n    Multiplicative and Multiplicative Damped\n    Minimization Function\n    (M,) & (Md,)\n    '
    (_, beta, phi, alphac, betac, y_alpha) = holt_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] + alphac * (lvl[i - 1] * b[i - 1] ** phi)
        b[i] = beta * (lvl[i] / lvl[i - 1]) + betac * b[i - 1] ** phi
    return hw_args.y - lvl * b ** phi

def holt_add_dam(x, hw_args: HoltWintersArgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Additive and Additive Damped\n    Minimization Function\n    (A,) & (Ad,)\n    '
    (_, beta, phi, alphac, betac, y_alpha) = holt_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] + alphac * (lvl[i - 1] + phi * b[i - 1])
        b[i] = beta * (lvl[i] - lvl[i - 1]) + betac * phi * b[i - 1]
    return hw_args.y - (lvl + phi * b)

def holt_win_init(x, hw_args: HoltWintersArgs):
    if False:
        return 10
    'Initialization for the Holt Winters Seasonal Models'
    hw_args.p[hw_args.xi.astype(bool)] = x
    if hw_args.transform:
        (alpha, beta, gamma) = to_restricted(hw_args.p, hw_args.xi, hw_args.bounds)
    else:
        (alpha, beta, gamma) = hw_args.p[:3]
    (l0, b0, phi) = hw_args.p[3:6]
    s0 = hw_args.p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha * hw_args.y
    y_gamma = gamma * hw_args.y
    hw_args.lvl[:] = 0
    hw_args.b[:] = 0
    hw_args.s[:] = 0
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0
    hw_args.s[:hw_args.m] = s0
    return (alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma)

def holt_win__mul(x, hw_args: HoltWintersArgs):
    if False:
        return 10
    '\n    Multiplicative Seasonal\n    Minimization Function\n    (,M)\n    '
    (_, _, _, _, alphac, _, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * lvl[i - 1]
        s[i + m - 1] = y_gamma[i - 1] / lvl[i - 1] + gammac * s[i - 1]
    return hw_args.y - lvl * s[:-(m - 1)]

def holt_win__add(x, hw_args: HoltWintersArgs):
    if False:
        return 10
    '\n    Additive Seasonal\n    Minimization Function\n    (,A)\n    '
    (alpha, _, gamma, _, alphac, _, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] - alpha * s[i - 1] + alphac * lvl[i - 1]
        s[i + m - 1] = y_gamma[i - 1] - gamma * lvl[i - 1] + gammac * s[i - 1]
    return hw_args.y - lvl - s[:-(m - 1)]

def holt_win_add_mul_dam(x, hw_args: HoltWintersArgs):
    if False:
        i = 10
        return i + 15
    '\n    Additive and Additive Damped with Multiplicative Seasonal\n    Minimization Function\n    (A,M) & (Ad,M)\n    '
    (_, beta, _, phi, alphac, betac, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * (lvl[i - 1] + phi * b[i - 1])
        b[i] = beta * (lvl[i] - lvl[i - 1]) + betac * phi * b[i - 1]
        s[i + m - 1] = y_gamma[i - 1] / (lvl[i - 1] + phi * b[i - 1]) + gammac * s[i - 1]
    return hw_args.y - (lvl + phi * b) * s[:-(m - 1)]

def holt_win_mul_mul_dam(x, hw_args: HoltWintersArgs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Multiplicative and Multiplicative Damped with Multiplicative Seasonal\n    Minimization Function\n    (M,M) & (Md,M)\n    '
    (_, beta, _, phi, alphac, betac, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * (lvl[i - 1] * b[i - 1] ** phi)
        b[i] = beta * (lvl[i] / lvl[i - 1]) + betac * b[i - 1] ** phi
        s[i + m - 1] = y_gamma[i - 1] / (lvl[i - 1] * b[i - 1] ** phi) + gammac * s[i - 1]
    return hw_args.y - lvl * b ** phi * s[:-(m - 1)]

def holt_win_add_add_dam(x, hw_args: HoltWintersArgs):
    if False:
        while True:
            i = 10
    '\n    Additive and Additive Damped with Additive Seasonal\n    Minimization Function\n    (A,A) & (Ad,A)\n    '
    (alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] - alpha * s[i - 1] + alphac * (lvl[i - 1] + phi * b[i - 1])
        b[i] = beta * (lvl[i] - lvl[i - 1]) + betac * phi * b[i - 1]
        s[i + m - 1] = y_gamma[i - 1] - gamma * (lvl[i - 1] + phi * b[i - 1]) + gammac * s[i - 1]
    return hw_args.y - (lvl + phi * b + s[:-(m - 1)])

def holt_win_mul_add_dam(x, hw_args: HoltWintersArgs):
    if False:
        print('Hello World!')
    '\n    Multiplicative and Multiplicative Damped with Additive Seasonal\n    Minimization Function\n    (M,A) & (M,Ad)\n    '
    (alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma) = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    s = hw_args.s
    b = hw_args.b
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] - alpha * s[i - 1] + alphac * (lvl[i - 1] * b[i - 1] ** phi)
        b[i] = beta * (lvl[i] / lvl[i - 1]) + betac * b[i - 1] ** phi
        s[i + m - 1] = y_gamma[i - 1] - gamma * (lvl[i - 1] * b[i - 1] ** phi) + gammac * s[i - 1]
    return hw_args.y - (lvl * phi * b + s[:-(m - 1)])