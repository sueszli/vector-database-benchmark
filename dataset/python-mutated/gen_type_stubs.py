import inspect
from operator import attrgetter
from textwrap import dedent
from zipline import api, TradingAlgorithm

def main():
    if False:
        i = 10
        return i + 15
    with open(api.__file__.rstrip('c') + 'i', 'w') as stub:
        stub.write(dedent('        import collections\n        from zipline.assets import Asset, Equity, Future\n        from zipline.assets.futures import FutureChain\n        from zipline.finance.asset_restrictions import Restrictions\n        from zipline.finance.cancel_policy import CancelPolicy\n        from zipline.pipeline import Pipeline\n        from zipline.protocol import Order\n        from zipline.utils.events import EventRule\n        from zipline.utils.security_list import SecurityList\n\n        '))
        for api_func in sorted(TradingAlgorithm.all_api_methods(), key=attrgetter('__name__')):
            stub.write('\n')
            sig = inspect._signature_bound_method(inspect.signature(api_func))
            indent = ' ' * 4
            stub.write(dedent('                def {func_name}{func_sig}:\n                    """'.format(func_name=api_func.__name__, func_sig=sig)))
            stub.write(dedent('{indent}{func_doc}'.format(func_doc=dedent(api_func.__doc__.lstrip()) or '\n', indent=indent)))
            stub.write('{indent}"""\n'.format(indent=indent))
if __name__ == '__main__':
    main()