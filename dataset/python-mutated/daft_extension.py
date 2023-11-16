"""
Useless IPython extension to test installing and loading extensions.
"""
some_vars = {'arq': 185}

def load_ipython_extension(ip):
    if False:
        for i in range(10):
            print('nop')
    ip.push(some_vars)

def unload_ipython_extension(ip):
    if False:
        print('Hello World!')
    ip.drop_by_id(some_vars)