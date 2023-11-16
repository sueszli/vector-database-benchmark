def canary_runtime_version(runtime_version):
    if False:
        i = 10
        return i + 15
    '\n    Property: Canary.RuntimeVersion\n    '
    valid_runtime_versions = ['syn-nodejs-2.0', 'syn-nodejs-2.0-beta', 'syn-1.0']
    if runtime_version not in valid_runtime_versions:
        raise ValueError('RuntimeVersion must be one of: "%s"' % ', '.join(valid_runtime_versions))
    return runtime_version