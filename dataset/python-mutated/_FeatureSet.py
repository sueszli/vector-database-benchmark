"""FeatureSet module.

Provides:
 - FeatureSet - container for Feature objects

For drawing capabilities, this module uses reportlab to draw and write
the diagram: http://www.reportlab.com
"""
from ._Feature import Feature
import re

class FeatureSet:
    """FeatureSet object."""

    def __init__(self, set_id=None, name=None, parent=None):
        if False:
            return 10
        'Create the object.\n\n        Arguments:\n         - set_id: Unique id for the set\n         - name: String identifying the feature set\n\n        '
        self.parent = parent
        self.id = id
        self.next_id = 0
        self.features = {}
        self.name = name

    def add_feature(self, feature, **kwargs):
        if False:
            i = 10
            return i + 15
        'Add a new feature.\n\n        Arguments:\n         - feature: Bio.SeqFeature object\n         - kwargs: Keyword arguments for Feature.  Named attributes\n           of the Feature\n\n        Add a Bio.SeqFeature object to the diagram (will be stored\n        internally in a Feature wrapper).\n        '
        id = self.next_id
        f = Feature(self, id, feature)
        self.features[id] = f
        for key in kwargs:
            if key == 'colour' or key == 'color':
                self.features[id].set_color(kwargs[key])
                continue
            setattr(self.features[id], key, kwargs[key])
        self.next_id += 1
        return f

    def del_feature(self, feature_id):
        if False:
            print('Hello World!')
        'Delete a feature.\n\n        Arguments:\n         - feature_id: Unique id of the feature to delete\n\n        Remove a feature from the set, indicated by its id.\n        '
        del self.features[feature_id]

    def set_all_features(self, attr, value):
        if False:
            return 10
        'Set an attribute of all the features.\n\n        Arguments:\n         - attr: An attribute of the Feature class\n         - value: The value to set that attribute to\n\n        Set the passed attribute of all features in the set to the\n        passed value.\n        '
        for feature in self.features.values():
            if hasattr(feature, attr):
                setattr(feature, attr, value)

    def get_features(self, attribute=None, value=None, comparator=None):
        if False:
            i = 10
            return i + 15
        "Retrieve features.\n\n        Arguments:\n         - attribute: String, attribute of a Feature object\n         - value: The value desired of the attribute\n         - comparator: String, how to compare the Feature attribute to the\n           passed value\n\n        If no attribute or value is given, return a list of all features in the\n        feature set.  If both an attribute and value are given, then depending\n        on the comparator, then a list of all features in the FeatureSet\n        matching (or not) the passed value will be returned.  Allowed comparators\n        are: 'startswith', 'not', 'like'.\n\n        The user is expected to make a responsible decision about which feature\n        attributes to use with which passed values and comparator settings.\n        "
        if attribute is None or value is None:
            return list(self.features.values())
        if comparator is None:
            return [feature for feature in self.features.values() if getattr(feature, attribute) == value]
        elif comparator == 'not':
            return [feature for feature in self.features.values() if getattr(feature, attribute) != value]
        elif comparator == 'startswith':
            return [feature for feature in self.features.values() if getattr(feature, attribute).startswith(value)]
        elif comparator == 'like':
            return [feature for feature in self.features.values() if re.search(value, getattr(feature, attribute))]
        return []

    def get_ids(self):
        if False:
            i = 10
            return i + 15
        'Return a list of all ids for the feature set.'
        return list(self.features.keys())

    def range(self):
        if False:
            i = 10
            return i + 15
        'Return the lowest and highest base (or mark) numbers as a tuple.'
        (lows, highs) = ([], [])
        for feature in self.features.values():
            for (start, end) in feature.locations:
                lows.append(start)
                highs.append(end)
        if len(lows) != 0 and len(highs) != 0:
            return (min(lows), max(highs))
        return (0, 0)

    def to_string(self, verbose=0):
        if False:
            return 10
        'Return a formatted string with information about the set.\n\n        Arguments:\n         - verbose: Boolean indicating whether a short (default) or\n           complete account of the set is required\n\n        '
        if not verbose:
            return f'{self}'
        else:
            outstr = [f'\n<{self.__class__}: {self.name}>']
            outstr.append('%d features' % len(self.features))
            for key in self.features:
                outstr.append(f'feature: {self.features[key]}')
            return '\n'.join(outstr)

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return the number of features in the set.'
        return len(self.features)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Return a feature, keyed by id.'
        return self.features[key]

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return a formatted string with information about the feature set.'
        outstr = ['\n<%s: %s %d features>' % (self.__class__, self.name, len(self.features))]
        return '\n'.join(outstr)