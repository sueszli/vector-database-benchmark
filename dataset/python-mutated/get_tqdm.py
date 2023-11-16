import sys

def get_tqdm():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a tqdm appropriate for the situation\n\n    imports tqdm depending on if we're at a console, redir to a file, notebook, etc\n\n    from @tcrimi at https://github.com/tqdm/tqdm/issues/506\n\n    This replaces `import tqdm`, so for example, you do this:\n      from stanza.utils.get_tqdm import get_tqdm\n      tqdm = get_tqdm()\n    then do this when you want a scroll bar or regular iterator depending on context:\n      tqdm(list)\n\n    If there is no tty, the returned tqdm will always be disabled\n    unless disable=False is specifically set.\n    "
    ipy_str = ''
    try:
        from IPython import get_ipython
        ipy_str = str(type(get_ipython()))
    except ImportError:
        pass
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
        return tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
        return tqdm
    if sys.stderr is not None and sys.stderr.isatty():
        from tqdm import tqdm
        return tqdm
    from tqdm import tqdm

    def hidden_tqdm(*args, **kwargs):
        if False:
            while True:
                i = 10
        if 'disable' in kwargs:
            return tqdm(*args, **kwargs)
        kwargs['disable'] = True
        return tqdm(*args, **kwargs)
    return hidden_tqdm