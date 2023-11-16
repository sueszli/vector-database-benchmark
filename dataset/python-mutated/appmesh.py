def validate_listenertls_mode(listenertls_mode):
    if False:
        i = 10
        return i + 15
    '\n    Validate Mode for ListernerTls\n    Property: ListenerTls.Mode\n    '
    VALID_LISTENERTLS_MODE = ('STRICT', 'PERMISSIVE', 'DISABLED')
    if listenertls_mode not in VALID_LISTENERTLS_MODE:
        raise ValueError('ListernerTls Mode must be one of: %s' % ', '.join(VALID_LISTENERTLS_MODE))
    return listenertls_mode