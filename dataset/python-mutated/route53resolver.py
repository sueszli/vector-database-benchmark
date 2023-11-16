def validate_ruletype(ruletype):
    if False:
        while True:
            i = 10
    '\n    Validate RuleType for ResolverRule.\n    Property: ResolverRule.RuleType\n    '
    VALID_RULETYPES = ('SYSTEM', 'FORWARD')
    if ruletype not in VALID_RULETYPES:
        raise ValueError('Rule type must be one of: %s' % ', '.join(VALID_RULETYPES))
    return ruletype