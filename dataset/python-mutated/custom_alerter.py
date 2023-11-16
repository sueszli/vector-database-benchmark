"""
.. module: security_monkey.alerters.custom_alerter
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey import app
alerter_registry = []

class AlerterType(type):

    def __init__(cls, name, bases, attrs):
        if False:
            while True:
                i = 10
        if getattr(cls, 'report_auditor_changes', None) and getattr(cls, 'report_watcher_changes', None):
            app.logger.debug('Registering alerter %s', cls.__name__)
            alerter_registry.append(cls)

def report_auditor_changes(auditor):
    if False:
        i = 10
        return i + 15
    for alerter_class in alerter_registry:
        alerter = alerter_class()
        alerter.report_auditor_changes(auditor)

def report_watcher_changes(watcher):
    if False:
        print('Hello World!')
    for alerter_class in alerter_registry:
        alerter = alerter_class()
        alerter.report_watcher_changes(watcher)