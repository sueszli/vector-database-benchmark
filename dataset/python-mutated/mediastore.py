def containerlevelmetrics_status(status):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: MetricPolicy.ContainerLevelMetrics\n    '
    valid_status = ['DISABLED', 'ENABLED']
    if status not in valid_status:
        raise ValueError('ContainerLevelMetrics must be one of: "%s"' % ', '.join(valid_status))
    return status