from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['compare_medians_ms', 'hdquantiles', 'hdmedian', 'hdquantiles_sd', 'idealfourths', 'median_cihs', 'mjci', 'mquantiles_cimj', 'rsh', 'trimmed_mean_ci', 'ma', 'MaskedArray', 'mstats', 'norm', 'beta', 't', 'binom']

def __dir__():
    if False:
        while True:
            i = 10
    return __all__

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    return _sub_module_deprecation(sub_package='stats', module='mstats_extras', private_modules=['_mstats_extras'], all=__all__, attribute=name, correct_module='mstats')