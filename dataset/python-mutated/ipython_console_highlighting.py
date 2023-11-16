"""
reST directive for syntax-highlighting ipython interactive sessions.

"""
from sphinx import highlighting
from IPython.lib.lexers import IPyLexer

def setup(app):
    if False:
        return 10
    'Setup as a sphinx extension.'
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
ipy2 = IPyLexer(python3=False)
ipy3 = IPyLexer(python3=True)
highlighting.lexers['ipython'] = ipy2
highlighting.lexers['ipython2'] = ipy2
highlighting.lexers['ipython3'] = ipy3