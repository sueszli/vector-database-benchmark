def resolver_kind_validator(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Resolver.Kind\n    '
    valid_types = ['UNIT', 'PIPELINE']
    if x not in valid_types:
        raise ValueError('Kind must be one of: %s' % ', '.join(valid_types))
    return x