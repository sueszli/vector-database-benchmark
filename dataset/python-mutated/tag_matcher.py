"""
Contains classes and functionality to provide the active-tag mechanism.
Active-tags provide a skip-if logic based on tags in feature files.
"""
from __future__ import absolute_import, print_function
import logging
import operator
import re
import six
from ._types import Unknown
from .compat.collections import UserDict

class ValueObject(object):
    """Value object for active-tags that holds the current value for
    one activate-tag category and its comparison function.

    The :param:`compare_func(current_value, tag_value)` is a predicate function
    with two arguments that performs the comparison between the
    "current_value" and the "tag_value".

    EXAMPLE::

        # -- SIMPLIFIED EXAMPLE:
        from behave.tag_matcher import ValueObject
        import operator     # Module contains comparison functions.
        class NumberObject(ValueObject): ...  # Details left out here.

        xxx_current_value = 42
        active_tag_value_provider = {
            "xxx.value": ValueObject(xxx_current_value)  # USES: operator.eq (equals)
            "xxx.min_value": NumberValueObject(xxx_current_value, operator.ge),
            "xxx.max_value": NumberValueObject(xxx_current_value, operator.le),
        }

        # -- LATER WITHIN: ActivTag Logic
        # EXAMPLE TAG: @use.with_xxx.min_value=10  (schema: "@use.with_{category}={value}")
        tag_category = "xxx.min_value"
        current_value = active_tag_value_provider.get(tag_category)
        if not isinstance(current_value, ValueObject):
            current_value = ValueObject(current_value)
        ...
        tag_matches = current_value.matches(tag_value)
    """

    def __init__(self, value, compare=operator.eq):
        if False:
            return 10
        assert callable(compare)
        self._value = value
        self.compare = compare

    @property
    def value(self):
        if False:
            i = 10
            return i + 15
        if callable(self._value):
            return self._value()
        return self._value

    def matches(self, tag_value):
        if False:
            return 10
        'Comparison between current value and :param:`tag_value`.\n\n        :param tag_value: Tag value from active tag (as string).\n        :return: True, if comparison matches. False, otherwise.\n        '
        return bool(self.compare(self.value, tag_value))

    @staticmethod
    def on_type_conversion_error(tag_value, e):
        if False:
            while True:
                i = 10
        logger = logging.getLogger('behave.active_tags')
        logger.error("TYPE CONVERSION ERROR: active_tag.value='%s' (error: %s)" % (tag_value, str(e)))
        return False

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Conversion to string.'
        return str(self.value)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s: value=%s, compare=%s>' % (self.__class__.__name__, self.value, self.compare)

class NumberValueObject(ValueObject):

    def matches(self, tag_value):
        if False:
            print('Hello World!')
        try:
            tag_number = int(tag_value)
            return super(NumberValueObject, self).matches(tag_number)
        except ValueError as e:
            return self.on_type_conversion_error(tag_value, e)

    def __int__(self):
        if False:
            return 10
        'Convert into integer-number value.'
        return int(self.value)

class BoolValueObject(ValueObject):
    TRUE_STRINGS = set(['true', 'yes', 'on'])
    FALSE_STRINGS = set(['false', 'no', 'off'])

    def matches(self, tag_value):
        if False:
            while True:
                i = 10
        try:
            boolean_tag_value = self.to_bool(tag_value)
            return super(BoolValueObject, self).matches(boolean_tag_value)
        except ValueError as e:
            return self.on_type_conversion_error(tag_value, e)

    def __bool__(self):
        if False:
            while True:
                i = 10
        'Conversion to boolean value.'
        return bool(self.value)

    @classmethod
    def to_bool(cls, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, six.string_types):
            text = value.lower()
            if text in cls.TRUE_STRINGS:
                return True
            elif text in cls.FALSE_STRINGS:
                return False
            else:
                raise ValueError('NON-BOOL: %s' % value)
        return bool(value)

class TagMatcher(object):
    """Abstract base class that defines the TagMatcher protocol."""

    def should_run_with(self, tags):
        if False:
            return 10
        'Determines if a feature/scenario with these tags should run or not.\n\n        :param tags:    List of scenario/feature tags to check.\n        :return: True,  if scenario/feature should run.\n        :return: False, if scenario/feature should be excluded from the run-set.\n        '
        return not self.should_exclude_with(tags)

    def should_exclude_with(self, tags):
        if False:
            print('Hello World!')
        'Determines if a feature/scenario with these tags should be excluded\n        from the run-set.\n\n        :param tags:    List of scenario/feature tags to check.\n        :return: True, if scenario/feature should be excluded from the run-set.\n        :return: False, if scenario/feature should run.\n        '
        raise NotImplementedError

class ActiveTagMatcher(TagMatcher):
    """Provides an active tag matcher for many categories.

    TAG SCHEMA 1 (preferred):
      * use.with_{category}={value}
      * not.with_{category}={value}

    TAG SCHEMA 2:
      * active.with_{category}={value}
      * not_active.with_{category}={value}

    TAG LOGIC
    ----------

    Determine active-tag groups by grouping active-tags
    with same category together::

        active_group.enabled := enabled(group.tag1) or enabled(group.tag2) or ...
        active_tags.enabled  := enabled(group1) and enabled(group2) and ...

    All active-tag groups must be turned "on" (enabled).
    Otherwise, the model element should be excluded.

    CONCEPT: ValueProvider
    ------------------------------

    A ValueProvider provides the value of a category, used in active tags.
    A ValueProvider must provide a mapping-like protocol:

    .. code-block:: python

        class MyValueProvider(object):
            def get(self, category_name, default=None):
                ...
                return category_value   # OR: default, if category is unknown.

    EXAMPLE:
    --------

    Run some scenarios only when runtime conditions are met:

      * Run scenario Alice only on Windows OS
      * Run scenario Bob with all browsers except Chrome

    .. code-block:: gherkin

        # -- FILE: features/alice.feature
        Feature:

          @use.with_os=win32
          Scenario: Alice (Run only on Windows)
            Given I do something
            ...

          @not.with_browser=chrome
          Scenario: Bob (Excluded with Web-Browser Chrome)
            Given I do something else
            ...


    .. code-block:: python

        # -- FILE: features/environment.py
        from behave.tag_matcher import ActiveTagMatcher
        import sys

        # -- MATCHES ANY ACTIVE TAGS: @{prefix}.with_{category}={value}
        # NOTE: active_tag_value_provider provides current category values.
        active_tag_value_provider = {
            "browser": os.environ.get("BEHAVE_BROWSER", "chrome"),
            "os":      sys.platform,
        }
        active_tag_matcher = ActiveTagMatcher(active_tag_value_provider)

        def before_feature(context, feature):
            if active_tag_matcher.should_exclude_with(feature.tags):
                feature.skip()   #< LATE-EXCLUDE from run-set.

        def before_scenario(context, scenario):
            if active_tag_matcher.should_exclude_with(scenario.effective_tags):
                exclude_reason = active_tag_matcher.exclude_reason
                scenario.skip(exclude_reason)   #< LATE-EXCLUDE from run-set.
    """
    value_separator = '='
    tag_prefixes = ['use', 'not', 'active', 'not_active', 'only']
    tag_schema = '^(?P<prefix>%s)\\.with_(?P<category>\\w+(\\.\\w+)*)%s(?P<value>.*)$'
    ignore_unknown_categories = True
    use_exclude_reason = False

    def __init__(self, value_provider, tag_prefixes=None, value_separator=None, ignore_unknown_categories=None):
        if False:
            for i in range(10):
                print('nop')
        if value_provider is None:
            value_provider = {}
        if tag_prefixes is None:
            tag_prefixes = self.tag_prefixes
        if ignore_unknown_categories is None:
            ignore_unknown_categories = self.ignore_unknown_categories
        super(ActiveTagMatcher, self).__init__()
        self.value_provider = value_provider
        self.tag_pattern = self.make_tag_pattern(tag_prefixes, value_separator)
        self.tag_prefixes = tag_prefixes
        self.ignore_unknown_categories = ignore_unknown_categories
        self.exclude_reason = None

    @classmethod
    def make_tag_pattern(cls, tag_prefixes, value_separator=None):
        if False:
            return 10
        if value_separator is None:
            value_separator = cls.value_separator
        any_tag_prefix = '|'.join(tag_prefixes)
        expression = cls.tag_schema % (any_tag_prefix, value_separator)
        return re.compile(expression)

    @classmethod
    def make_category_tag(cls, category, value=None, tag_prefix=None, value_sep=None):
        if False:
            print('Hello World!')
        'Build category tag (mostly for testing purposes).\n        :return: Category tag as string (without leading AT char).\n        '
        if tag_prefix is None:
            tag_prefix = cls.tag_prefixes[0]
        if value_sep is None:
            value_sep = cls.value_separator
        value = value or ''
        return '%s.with_%s%s%s' % (tag_prefix, category, value_sep, value)

    def is_tag_negated(self, tag):
        if False:
            print('Hello World!')
        return tag.startswith('not')

    def is_tag_group_enabled(self, group_category, group_tag_pairs):
        if False:
            for i in range(10):
                print('nop')
        'Provides boolean logic to determine if all active-tags\n        which use the same category result in an enabled value.\n\n        .. code-block:: gherkin\n\n            @use.with_xxx=alice\n            @use.with_xxx=bob\n            @not.with_xxx=charly\n            @not.with_xxx=doro\n            Scenario:\n                Given a step passes\n                ...\n\n        Use LOGICAL expression for active-tags with same category::\n\n            category_tag_group.enabled := positive-tag-expression and not negative-tag-expression\n              positive-tag-expression  := enabled(tag1) or enabled(tag2) or ...\n              negative-tag-expression  := enabled(tag3) or enabled(tag4) or ...\n               tag1, tag2 are positive-tags, like @use.with_category=value\n               tag3, tag4 are negative-tags, like @not.with_category=value\n\n             xxx   | Only use parts: (xxx == "alice") or (xxx == "bob")\n            -------+-------------------\n            alice  | true\n            bob    | true\n            other  | false\n\n             xxx   | Only not parts:\n                   | (not xxx == "charly") and (not xxx == "doro")\n                   | = not((xxx == "charly") or (xxx == "doro"))\n            -------+-------------------\n            charly | false\n            doro   | false\n            other  | true\n\n             xxx   | Use and not parts:\n                   | ((xxx == "alice") or (xxx == "bob")) and not((xxx == "charly") or (xxx == "doro"))\n            -------+-------------------\n            alice  | true\n            bob    | true\n            charly | false\n            doro   | false\n            other  | false\n\n        :param group_category:      Category for this tag-group (as string).\n        :param category_tag_group:  List of active-tag match-pairs.\n        :return: True, if tag-group is enabled.\n        '
        if not group_tag_pairs:
            return True
        current_value = self.value_provider.get(group_category, Unknown)
        if current_value is Unknown and self.ignore_unknown_categories:
            return True
        elif not isinstance(current_value, ValueObject):
            current_value = ValueObject(current_value)
        positive_tags_matched = []
        negative_tags_matched = []
        for (category_tag, tag_match) in group_tag_pairs:
            tag_prefix = tag_match.group('prefix')
            category = tag_match.group('category')
            tag_value = tag_match.group('value')
            assert category == group_category
            if self.is_tag_negated(tag_prefix):
                tag_matched = current_value.matches(tag_value)
                negative_tags_matched.append(tag_matched)
            else:
                tag_matched = current_value.matches(tag_value)
                positive_tags_matched.append(tag_matched)
        tag_expression1 = any(positive_tags_matched)
        tag_expression2 = any(negative_tags_matched)
        if not positive_tags_matched:
            tag_expression1 = True
        tag_group_enabled = bool(tag_expression1 and (not tag_expression2))
        return tag_group_enabled

    def should_exclude_with(self, tags):
        if False:
            i = 10
            return i + 15
        group_categories = self.group_active_tags_by_category(tags)
        for (group_category, category_tag_pairs) in group_categories:
            if not self.is_tag_group_enabled(group_category, category_tag_pairs):
                if self.use_exclude_reason:
                    current_value = self.value_provider.get(group_category, None)
                    reason = '%s (but: %s)' % (group_category, current_value)
                    self.exclude_reason = reason
                return True
        return False

    def select_active_tags(self, tags):
        if False:
            while True:
                i = 10
        'Select all active tags that match the tag schema pattern.\n\n        :param tags: List of tags (as string).\n        :return: List of (tag, match_object) pairs (as generator).\n        '
        for tag in tags:
            match_object = self.tag_pattern.match(tag)
            if match_object:
                yield (tag, match_object)

    def group_active_tags_by_category(self, tags):
        if False:
            i = 10
            return i + 15
        'Select all active tags that match the tag schema pattern\n        and returns groups of active-tags, each group with tags\n        of the same category.\n\n        :param tags: List of tags (as string).\n        :return: List of tag-groups (as generator), each tag-group is a\n                list of (tag1, match1) pairs for the same category.\n        '
        category_tag_groups = {}
        for tag in tags:
            match_object = self.tag_pattern.match(tag)
            if match_object:
                category = match_object.group('category')
                category_tag_pairs = category_tag_groups.get(category, None)
                if category_tag_pairs is None:
                    category_tag_pairs = category_tag_groups[category] = []
                category_tag_pairs.append((tag, match_object))
        for (category, category_tag_pairs) in six.iteritems(category_tag_groups):
            yield (category, category_tag_pairs)

class PredicateTagMatcher(TagMatcher):

    def __init__(self, exclude_function):
        if False:
            print('Hello World!')
        assert callable(exclude_function)
        super(PredicateTagMatcher, self).__init__()
        self.predicate = exclude_function

    def should_exclude_with(self, tags):
        if False:
            return 10
        return self.predicate(tags)

class CompositeTagMatcher(TagMatcher):
    """Provides a composite tag matcher."""

    def __init__(self, tag_matchers=None):
        if False:
            while True:
                i = 10
        super(CompositeTagMatcher, self).__init__()
        self.tag_matchers = tag_matchers or []

    def should_exclude_with(self, tags):
        if False:
            for i in range(10):
                print('nop')
        for tag_matcher in self.tag_matchers:
            if tag_matcher.should_exclude_with(tags):
                return True
        return False

class IActiveTagValueProvider(object):
    """Protocol/Interface for active-tag value providers."""

    def get(self, category, default=None):
        if False:
            print('Hello World!')
        return NotImplemented

class ActiveTagValueProvider(UserDict):

    def __init__(self, data=None):
        if False:
            return 10
        if data is None:
            data = {}
        UserDict.__init__(self, data)

    @staticmethod
    def use_value(value):
        if False:
            print('Hello World!')
        if callable(value):
            value_func = value
            value = value_func()
        return value

    def __getitem__(self, name):
        if False:
            i = 10
            return i + 15
        value = self.data[name]
        return self.use_value(value)

    def get(self, category, default=None):
        if False:
            print('Hello World!')
        value = self.data.get(category, default)
        return self.use_value(value)

    def values(self):
        if False:
            i = 10
            return i + 15
        for value in self.data.values(self):
            yield self.use_value(value)

    def items(self):
        if False:
            return 10
        for (category, value) in self.data.items():
            yield (category, self.use_value(value))

    def categories(self):
        if False:
            return 10
        return self.keys()

class CompositeActiveTagValueProvider(ActiveTagValueProvider):
    """Provides a composite helper class to resolve active-tag values
    from a list of value-providers.
    """

    def __init__(self, value_providers=None):
        if False:
            return 10
        if value_providers is None:
            value_providers = []
        super(CompositeActiveTagValueProvider, self).__init__()
        self.value_providers = list(value_providers)

    def get(self, category, default=None):
        if False:
            while True:
                i = 10
        value = self.data.get(category, Unknown)
        if value is Unknown:
            for value_provider in self.value_providers:
                value = value_provider.get(category, Unknown)
                if value is Unknown:
                    continue
                self.data[category] = value
                break
            if value is Unknown:
                value = default
        return self.use_value(value)

    def keys(self):
        if False:
            return 10
        for value_provider in self.value_providers:
            try:
                for category in value_provider.keys():
                    yield category
            except AttributeError:
                pass

    def values(self):
        if False:
            for i in range(10):
                print('nop')
        for category in self.keys():
            value = self.get(category)
            yield value

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        for category in self.keys():
            value = self.get(category)
            yield (category, value)

def bool_to_string(value):
    if False:
        for i in range(10):
            print('nop')
    'Converts a boolean active-tag value into its normalized\n    string representation.\n\n    :param value:  Boolean value to use (or value converted into bool).\n    :returns: Boolean value converted into a normalized string.\n    '
    return str(bool(value)).lower()

def setup_active_tag_values(active_tag_values, data):
    if False:
        return 10
    'Setup/update active_tag values with dict-like data.\n    Only values for keys that are already present are updated.\n\n    :param active_tag_values:   Data storage for active_tag value (dict-like).\n    :param data:   Data that should be used for active_tag values (dict-like).\n    '
    for category in list(active_tag_values.keys()):
        if category in data:
            active_tag_values[category] = data[category]

def print_active_tags(active_tag_value_provider, categories=None):
    if False:
        for i in range(10):
            print('nop')
    'Print a summary of the current active-tag values.'
    if categories is None:
        try:
            categories = list(active_tag_value_provider.keys())
        except TypeError:
            categories = []
    active_tag_data = active_tag_value_provider
    print('ACTIVE-TAGS:')
    for category in categories:
        active_tag_value = active_tag_data.get(category)
        print('use.with_{category}={value}'.format(category=category, value=active_tag_value))
    print()