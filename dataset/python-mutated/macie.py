def findingsfilter_action(action):
    if False:
        i = 10
        return i + 15
    '\n    Property: FindingsFilter.Action\n    '
    valid_actions = ['ARCHIVE', 'NOOP']
    if action not in valid_actions:
        raise ValueError('Action must be one of: "%s"' % ', '.join(valid_actions))
    return action

def session_findingpublishingfrequency(frequency):
    if False:
        return 10
    '\n    Property: Session.FindingPublishingFrequency\n    '
    valid_frequencies = ['FIFTEEN_MINUTES', 'ONE_HOUR', 'SIX_HOURS']
    if frequency not in valid_frequencies:
        raise ValueError('FindingPublishingFrequency must be one of: "%s"' % ', '.join(valid_frequencies))
    return frequency

def session_status(status):
    if False:
        print('Hello World!')
    '\n    Property: Session.Status\n    '
    valid_status = ['ENABLED', 'DISABLED']
    if status not in valid_status:
        raise ValueError('Status must be one of: "%s"' % ', '.join(valid_status))
    return status