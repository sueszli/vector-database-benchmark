def kw_only_arg(*, kwo):
    if False:
        print('Hello World!')
    return kwo

def many_kw_only_args(*, first, second, third):
    if False:
        print('Hello World!')
    return first + second + third

def kw_only_arg_with_default(*, kwo='default', another='another'):
    if False:
        return 10
    return '{}-{}'.format(kwo, another)

def mandatory_after_defaults(*, default1='xxx', mandatory, default2='zzz'):
    if False:
        i = 10
        return i + 15
    return '{}-{}-{}'.format(default1, mandatory, default2)

def kw_only_arg_with_annotation(*, kwo: str):
    if False:
        while True:
            i = 10
    return kwo

def kw_only_arg_with_annotation_and_default(*, kwo: str='default'):
    if False:
        i = 10
        return i + 15
    return kwo

def kw_only_arg_with_varargs(*varargs, kwo):
    if False:
        i = 10
        return i + 15
    return '-'.join(varargs + (kwo,))

def all_arg_types(pos_req, pos_def='pd', *varargs, kwo_req, kwo_def='kd', **kwargs):
    if False:
        return 10
    varargs = list(varargs)
    kwargs = ['%s=%s' % item for item in sorted(kwargs.items())]
    return '-'.join([pos_req, pos_def] + varargs + [kwo_req, kwo_def] + kwargs)