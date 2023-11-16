import sys
import cupy

def who(vardict=None):
    if False:
        print('Hello World!')
    "Print the CuPy arrays in the given dictionary.\n\n    Prints out the name, shape, bytes and type of all of the ndarrays\n    present in `vardict`.\n\n    If there is no dictionary passed in or `vardict` is None then returns\n    CuPy arrays in the globals() dictionary (all CuPy arrays in the\n    namespace).\n\n    Args:\n        vardict : (None or dict)  A dictionary possibly containing ndarrays.\n                  Default is globals() if `None` specified\n\n\n    .. admonition:: Example\n\n        >>> a = cupy.arange(10)\n        >>> b = cupy.ones(20)\n        >>> cupy.who()\n        Name            Shape            Bytes            Type\n        ===========================================================\n        <BLANKLINE>\n        a               10               80               int64\n        b               20               160              float64\n        <BLANKLINE>\n        Upper bound on total bytes  =       240\n        >>> d = {'x': cupy.arange(2.0),\n        ... 'y': cupy.arange(3.0), 'txt': 'Some str',\n        ... 'idx':5}\n        >>> cupy.who(d)\n        Name            Shape            Bytes            Type\n        ===========================================================\n        <BLANKLINE>\n        x               2                16               float64\n        y               3                24               float64\n        <BLANKLINE>\n        Upper bound on total bytes  =       40\n\n    "
    if vardict is None:
        frame = sys._getframe().f_back
        vardict = frame.f_globals
    sta = []
    cache = {}
    for name in sorted(vardict.keys()):
        if isinstance(vardict[name], cupy.ndarray):
            var = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = '{} ({})'.format(name, cache[idv])
                original = 0
            else:
                cache[idv] = name
                namestr = name
                original = 1
            shapestr = ' x '.join(map(str, var.shape))
            bytestr = str(var.nbytes)
            sta.append([namestr, shapestr, bytestr, var.dtype.name, original])
    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])
    if len(sta) > 0:
        sp1 = max(10, maxname)
        sp2 = max(10, maxshape)
        sp3 = max(10, maxbyte)
        prval = 'Name {} Shape {} Bytes {} Type'.format(sp1 * ' ', sp2 * ' ', sp3 * ' ')
        print('{}\n{}\n'.format(prval, '=' * (len(prval) + 5)))
    for k in range(len(sta)):
        val = sta[k]
        print('{} {} {} {} {} {} {}'.format(val[0], ' ' * (sp1 - len(val[0]) + 4), val[1], ' ' * (sp2 - len(val[1]) + 5), val[2], ' ' * (sp3 - len(val[2]) + 5), val[3]))
    print('\nUpper bound on total bytes  =       {}'.format(totalbytes))