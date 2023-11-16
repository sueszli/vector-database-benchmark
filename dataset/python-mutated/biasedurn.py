from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['_PyFishersNCHypergeometric', '_PyWalleniusNCHypergeometric', '_PyStochasticLib3']

def __dir__():
    if False:
        print('Hello World!')
    return __all__

def __getattr__(name):
    if False:
        return 10
    return _sub_module_deprecation(sub_package='stats', module='biasedurn', private_modules=['_biasedurn'], all=__all__, attribute=name)