from .. import AWSHelperFn

def validate_trigger(self):
    if False:
        i = 10
        return i + 15
    '\n    Class: Trigger\n    '
    valid = ['all', 'createReference', 'deleteReference', 'updateReference']
    events = self.properties.get('Events')
    if events and (not isinstance(events, AWSHelperFn)):
        if 'all' in events and len(events) != 1:
            raise ValueError('Trigger events: all must be used alone')
        else:
            for e in events:
                if e not in valid and (not isinstance(e, AWSHelperFn)):
                    raise ValueError('Trigger: invalid event %s' % e)