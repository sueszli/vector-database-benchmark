def _formatparam(param, value=None, quote=True):
    if False:
        while True:
            i = 10
    if value is not None and len(value) > 0:
        if isinstance(value, tuple):
            value = 'a'
        if quote or param:
            pass
        else:
            return '%s=%s' % (param, value)
    else:
        return param

def system_methodSignature(seflf, method_name):
    if False:
        for i in range(10):
            print('nop')
    return 'signatures not supported'