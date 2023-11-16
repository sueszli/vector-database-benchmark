from scipy._lib.deprecation import _sub_module_deprecation
__all__ = ['LineSearchWarning', 'line_search', 'line_search_BFGS', 'line_search_armijo', 'line_search_wolfe1', 'line_search_wolfe2', 'minpack2', 'scalar_search_armijo', 'scalar_search_wolfe1', 'scalar_search_wolfe2', 'warn']

def __dir__():
    if False:
        return 10
    return __all__

def __getattr__(name):
    if False:
        print('Hello World!')
    return _sub_module_deprecation(sub_package='optimize', module='linesearch', private_modules=['_linesearch'], all=__all__, attribute=name)