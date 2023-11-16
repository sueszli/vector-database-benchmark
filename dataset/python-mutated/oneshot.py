from .base import Strategy
try:
    from nni.nas.oneshot.pytorch.strategy import DARTS, GumbelDARTS, Proxyless, ENAS, RandomOneShot
except ImportError as import_err:
    _import_err = import_err

    class ImportFailedStrategy(Strategy):

        def run(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise _import_err
    globals()['DARTS'] = ImportFailedStrategy
    globals()['GumbelDARTS'] = ImportFailedStrategy
    globals()['Proxyless'] = ImportFailedStrategy
    globals()['ENAS'] = ImportFailedStrategy
    globals()['RandomOneShot'] = ImportFailedStrategy