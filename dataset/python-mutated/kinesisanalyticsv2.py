def validate_runtime_environment(runtime_environment):
    if False:
        while True:
            i = 10
    '\n    Validate RuntimeEnvironment for Application\n    Property: Application.RuntimeEnvironment\n    '
    VALID_RUNTIME_ENVIRONMENTS = ('FLINK-1_6', 'FLINK-1_8', 'FLINK-1_11', 'FLINK-1_13', 'FLINK-1_15', 'SQL-1_0', 'ZEPPELIN-FLINK-1_0', 'ZEPPELIN-FLINK-2_0')
    if runtime_environment not in VALID_RUNTIME_ENVIRONMENTS:
        raise ValueError('Application RuntimeEnvironment must be one of: %s' % ', '.join(VALID_RUNTIME_ENVIRONMENTS))
    return runtime_environment