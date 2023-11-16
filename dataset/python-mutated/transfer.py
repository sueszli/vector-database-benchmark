def validate_homedirectory_type(homedirectory_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate HomeDirectoryType for User\n    Property: User.HomeDirectoryType\n    '
    VALID_HOMEDIRECTORY_TYPE = ('LOGICAL', 'PATH')
    if homedirectory_type not in VALID_HOMEDIRECTORY_TYPE:
        raise ValueError('User HomeDirectoryType must be one of: %s' % ', '.join(VALID_HOMEDIRECTORY_TYPE))
    return homedirectory_type