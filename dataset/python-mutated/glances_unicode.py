"""Manage unicode message for Glances output."""
_unicode_message = {'ARROW_LEFT': [u'←', u'<'], 'ARROW_RIGHT': [u'→', u'>'], 'ARROW_UP': [u'↑', u'^'], 'ARROW_DOWN': [u'↓', u'v'], 'CHECK': [u'✓', u''], 'PROCESS_SELECTOR': [u'>', u'>'], 'MEDIUM_LINE': [u'⎯', u'-'], 'LOW_LINE': [u'▁', u'_']}

def unicode_message(key, args=None):
    if False:
        return 10
    'Return the unicode message for the given key.'
    if args and hasattr(args, 'disable_unicode') and args.disable_unicode:
        return _unicode_message[key][1]
    else:
        return _unicode_message[key][0]