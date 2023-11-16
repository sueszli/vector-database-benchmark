from robot.api.deco import keyword

class UnknownType:
    pass

@keyword(types={'integer': int, 'boolean': bool, 'string': str})
def A_basics(integer, boolean, string):
    if False:
        i = 10
        return i + 15
    pass

@keyword(types={'integer': int, 'list_': list})
def B_with_defaults(integer=42, list_=None):
    if False:
        return 10
    pass

@keyword(types={'varargs': int, 'kwargs': bool})
def C_varags_and_kwargs(*varargs, **kwargs):
    if False:
        return 10
    pass

@keyword(types={'unknown': UnknownType, 'unrecognized': Ellipsis})
def D_unknown_types(unknown, unrecognized):
    if False:
        print('Hello World!')
    pass

@keyword(types={'arg': 'One of the usages in PEP-3107', 'varargs': 'But surely feels odd...'})
def E_non_type_annotations(arg, *varargs):
    if False:
        for i in range(10):
            print('nop')
    pass

@keyword(types={'kwo': int, 'with_default': str})
def F_kw_only_args(*, kwo, with_default='value'):
    if False:
        for i in range(10):
            print('nop')
    pass