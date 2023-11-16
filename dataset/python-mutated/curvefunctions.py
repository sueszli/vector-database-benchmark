"""
A family of functions used by CurvefittingAssessor
"""
import numpy as np
all_models = {}
model_para = {}
model_para_num = {}
curve_combination_models = ['vap', 'pow3', 'linear', 'logx_linear', 'dr_hill_zero_background', 'log_power', 'pow4', 'mmf', 'exp4', 'ilog2', 'weibull', 'janoschek']

def vap(x, a, b, c):
    if False:
        i = 10
        return i + 15
    'Vapor pressure model\n\n    Parameters\n    ----------\n    x : int\n    a : float\n    b : float\n    c : float\n\n    Returns\n    -------\n    float\n        np.exp(a+b/x+c*np.log(x))\n    '
    return np.exp(a + b / x + c * np.log(x))
all_models['vap'] = vap
model_para['vap'] = [-0.622028, -0.47005, 0.042322]
model_para_num['vap'] = 3

def pow3(x, c, a, alpha):
    if False:
        while True:
            i = 10
    'pow3\n\n    Parameters\n    ----------\n    x : int\n    c : float\n    a : float\n    alpha : float\n\n    Returns\n    -------\n    float\n        c - a * x**(-alpha)\n    '
    return c - a * x ** (-alpha)
all_models['pow3'] = pow3
model_para['pow3'] = [0.84, 0.52, 0.01]
model_para_num['pow3'] = 3

def linear(x, a, b):
    if False:
        return 10
    'linear\n\n    Parameters\n    ----------\n    x : int\n    a : float\n    b : float\n\n    Returns\n    -------\n    float\n        a*x + b\n    '
    return a * x + b
all_models['linear'] = linear
model_para['linear'] = [1.0, 0]
model_para_num['linear'] = 2

def logx_linear(x, a, b):
    if False:
        print('Hello World!')
    'logx linear\n\n    Parameters\n    ----------\n    x : int\n    a : float\n    b : float\n\n    Returns\n    -------\n    float\n        a * np.log(x) + b\n    '
    x = np.log(x)
    return a * x + b
all_models['logx_linear'] = logx_linear
model_para['logx_linear'] = [0.378106, 0.046506]
model_para_num['logx_linear'] = 2

def dr_hill_zero_background(x, theta, eta, kappa):
    if False:
        return 10
    'dr hill zero background\n\n    Parameters\n    ----------\n    x : int\n    theta : float\n    eta : float\n    kappa : float\n\n    Returns\n    -------\n    float\n        (theta* x**eta) / (kappa**eta + x**eta)\n    '
    return theta * x ** eta / (kappa ** eta + x ** eta)
all_models['dr_hill_zero_background'] = dr_hill_zero_background
model_para['dr_hill_zero_background'] = [0.77232, 0.586449, 2.460843]
model_para_num['dr_hill_zero_background'] = 3

def log_power(x, a, b, c):
    if False:
        i = 10
        return i + 15
    '"logistic power\n\n    Parameters\n    ----------\n    x : int\n    a : float\n    b : float\n    c : float\n\n    Returns\n    -------\n    float\n        a/(1.+(x/np.exp(b))**c)\n    '
    return a / (1.0 + (x / np.exp(b)) ** c)
all_models['log_power'] = log_power
model_para['log_power'] = [0.77, 2.98, -0.51]
model_para_num['log_power'] = 3

def pow4(x, alpha, a, b, c):
    if False:
        i = 10
        return i + 15
    'pow4\n\n    Parameters\n    ----------\n    x : int\n    alpha : float\n    a : float\n    b : float\n    c : float\n\n    Returns\n    -------\n    float\n        c - (a*x+b)**-alpha\n    '
    return c - (a * x + b) ** (-alpha)
all_models['pow4'] = pow4
model_para['pow4'] = [0.1, 200, 0.0, 0.8]
model_para_num['pow4'] = 4

def mmf(x, alpha, beta, kappa, delta):
    if False:
        return 10
    'Morgan-Mercer-Flodin\n    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm\n\n    Parameters\n    ----------\n    x : int\n    alpha : float\n    beta : float\n    kappa : float\n    delta : float\n\n    Returns\n    -------\n    float\n        alpha - (alpha - beta) / (1. + (kappa * x)**delta)\n    '
    return alpha - (alpha - beta) / (1.0 + (kappa * x) ** delta)
all_models['mmf'] = mmf
model_para['mmf'] = [0.7, 0.1, 0.01, 5]
model_para_num['mmf'] = 4

def exp4(x, c, a, b, alpha):
    if False:
        i = 10
        return i + 15
    'exp4\n\n    Parameters\n    ----------\n    x : int\n    c : float\n    a : float\n    b : float\n    alpha : float\n\n    Returns\n    -------\n    float\n        c - np.exp(-a*(x**alpha)+b)\n    '
    return c - np.exp(-a * x ** alpha + b)
all_models['exp4'] = exp4
model_para['exp4'] = [0.7, 0.8, -0.8, 0.3]
model_para_num['exp4'] = 4

def ilog2(x, c, a):
    if False:
        for i in range(10):
            print('nop')
    'ilog2\n\n    Parameters\n    ----------\n    x : int\n    c : float\n    a : float\n\n    Returns\n    -------\n    float\n        c - a / np.log(x)\n    '
    return c - a / np.log(x)
all_models['ilog2'] = ilog2
model_para['ilog2'] = [0.78, 0.43]
model_para_num['ilog2'] = 2

def weibull(x, alpha, beta, kappa, delta):
    if False:
        for i in range(10):
            print('nop')
    'Weibull model\n    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm\n\n    Parameters\n    ----------\n    x : int\n    alpha : float\n    beta : float\n    kappa : float\n    delta : float\n\n    Returns\n    -------\n    float\n        alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)\n    '
    return alpha - (alpha - beta) * np.exp(-(kappa * x) ** delta)
all_models['weibull'] = weibull
model_para['weibull'] = [0.7, 0.1, 0.01, 1]
model_para_num['weibull'] = 4

def janoschek(x, a, beta, k, delta):
    if False:
        return 10
    'http://www.pisces-conservation.com/growthhelp/janoschek.htm\n\n    Parameters\n    ----------\n    x : int\n    a : float\n    beta : float\n    k : float\n    delta : float\n\n    Returns\n    -------\n    float\n        a - (a - beta) * np.exp(-k*x**delta)\n    '
    return a - (a - beta) * np.exp(-k * x ** delta)
all_models['janoschek'] = janoschek
model_para['janoschek'] = [0.73, 0.07, 0.355, 0.46]
model_para_num['janoschek'] = 4