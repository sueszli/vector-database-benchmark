def ipython_magics_cpu_test():
    if False:
        for i in range(10):
            print('nop')
    import warnings
    from IPython.core.interactiveshell import InteractiveShell
    from traitlets.config import Config
    warnings.filterwarnings('ignore', category=UserWarning)
    c = Config()
    c.HistoryManager.hist_file = ':memory:'
    ip = InteractiveShell(config=c)
    ip.run_line_magic('load_ext', 'cudf.pandas')
    ip.run_cell('import pandas as pd; s = pd.Series(range(5))')
    result = ip.run_cell("assert not hasattr(s, '_fsproxy_state')")
    result.raise_error()
if __name__ == '__main__':
    ipython_magics_cpu_test()