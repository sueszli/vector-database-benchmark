from . import tags_or_list

def validate_tags_or_list(x):
    if False:
        print('Hello World!')
    '\n    Property: LifecyclePolicy.Tags\n    Property: PolicyDetails.TargetTags\n    Property: Schedule.TagsToAdd\n    '
    return tags_or_list(x)

def validate_interval(interval):
    if False:
        i = 10
        return i + 15
    '\n    Interval validation rule.\n    Property: CreateRule.Interval\n    '
    VALID_INTERVALS = (2, 3, 4, 6, 8, 12, 24)
    if interval not in VALID_INTERVALS:
        raise ValueError('Interval must be one of : %s' % ', '.join([str(i) for i in VALID_INTERVALS]))
    return interval

def validate_interval_unit(interval_unit):
    if False:
        print('Hello World!')
    '\n    Interval unit validation rule.\n    Property: CreateRule.IntervalUnit\n    '
    VALID_INTERVAL_UNITS = ('HOURS',)
    if interval_unit not in VALID_INTERVAL_UNITS:
        raise ValueError('Interval unit must be one of : %s' % ', '.join(VALID_INTERVAL_UNITS))
    return interval_unit

def validate_state(state):
    if False:
        while True:
            i = 10
    '\n    State validation rule.\n    Property: LifecyclePolicy.State\n    '
    VALID_STATES = ('ENABLED', 'DISABLED')
    if state not in VALID_STATES:
        raise ValueError('State must be one of : %s' % ', '.join(VALID_STATES))
    return state