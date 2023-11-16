def validate_recoveryoption_name(recoveryoption_name):
    if False:
        print('Hello World!')
    '\n    Validate Name for RecoveryOption\n    Property: RecoveryOption.Name\n    '
    VALID_RECOVERYOPTION_NAME = ('admin_only', 'verified_email', 'verified_phone_number')
    if recoveryoption_name not in VALID_RECOVERYOPTION_NAME:
        raise ValueError('RecoveryOption Name must be one of: %s' % ', '.join(VALID_RECOVERYOPTION_NAME))
    return recoveryoption_name