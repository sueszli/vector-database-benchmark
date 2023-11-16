"""
Thresholds classes: OK, CAREFUL, WARNING, CRITICAL
"""
import sys
from functools import total_ordering

class GlancesThresholds(object):
    """Class to manage thresholds dict for all Glances plugins:

    key: Glances stats (example: cpu_user)
    value: Threshold instance
    """
    threshold_list = ['OK', 'CAREFUL', 'WARNING', 'CRITICAL']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.current_module = sys.modules[__name__]
        self._thresholds = {}

    def get(self, stat_name=None):
        if False:
            while True:
                i = 10
        'Return the threshold dict.\n        If stat_name is None, return the threshold for all plugins (dict of Threshold*)\n        Else return the Threshold* instance for the given plugin\n        '
        if stat_name is None:
            return self._thresholds
        if stat_name in self._thresholds:
            return self._thresholds[stat_name]
        else:
            return {}

    def add(self, stat_name, threshold_description):
        if False:
            return 10
        'Add a new threshold to the dict (key = stat_name)'
        if threshold_description not in self.threshold_list:
            return False
        else:
            self._thresholds[stat_name] = getattr(self.current_module, 'GlancesThreshold' + threshold_description.capitalize())()
            return True
glances_thresholds = GlancesThresholds()

@total_ordering
class _GlancesThreshold(object):
    """Father class for all other Thresholds"""

    def description(self):
        if False:
            print('Hello World!')
        return self._threshold['description']

    def value(self):
        if False:
            while True:
                i = 10
        return self._threshold['value']

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self._threshold)

    def __str__(self):
        if False:
            return 10
        return self.description()

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self.value() < other.value()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value() == other.value()

class GlancesThresholdOk(_GlancesThreshold):
    """Ok Threshold class"""
    _threshold = {'description': 'OK', 'value': 0}

class GlancesThresholdCareful(_GlancesThreshold):
    """Careful Threshold class"""
    _threshold = {'description': 'CAREFUL', 'value': 1}

class GlancesThresholdWarning(_GlancesThreshold):
    """Warning Threshold class"""
    _threshold = {'description': 'WARNING', 'value': 2}

class GlancesThresholdCritical(_GlancesThreshold):
    """Warning Threshold class"""
    _threshold = {'description': 'CRITICAL', 'value': 3}