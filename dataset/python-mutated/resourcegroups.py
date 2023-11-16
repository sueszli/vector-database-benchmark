def resourcequery_type(type):
    if False:
        i = 10
        return i + 15
    '\n    Property: ResourceQuery.Type\n    '
    valid_types = ['TAG_FILTERS_1_0', 'CLOUDFORMATION_STACK_1_0']
    if type not in valid_types:
        raise ValueError('Type must be one of: "%s"' % ', '.join(valid_types))
    return type