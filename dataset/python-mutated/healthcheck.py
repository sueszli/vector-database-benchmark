from hypothesis.errors import FailedHealthCheck

def fail_health_check(settings, message, label):
    if False:
        return 10
    __tracebackhide__ = True
    if label in settings.suppress_health_check:
        return
    message += f'\nSee https://hypothesis.readthedocs.io/en/latest/healthchecks.html for more information about this. If you want to disable just this health check, add {label} to the suppress_health_check settings for this test.'
    raise FailedHealthCheck(message)