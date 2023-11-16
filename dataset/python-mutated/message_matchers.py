from hamcrest.core.base_matcher import BaseMatcher
IGNORED = object()

class MetricStructuredNameMatcher(BaseMatcher):
    """Matches a MetricStructuredName."""

    def __init__(self, name=IGNORED, origin=IGNORED, context=IGNORED):
        if False:
            while True:
                i = 10
        'Creates a MetricsStructuredNameMatcher.\n\n    Any property not passed in to the constructor will be ignored when matching.\n\n    Args:\n      name: A string with the metric name.\n      origin: A string with the metric namespace.\n      context: A key:value dictionary that will be matched to the\n        structured name.\n    '
        if context != IGNORED and (not isinstance(context, dict)):
            raise ValueError('context must be a Python dictionary.')
        self.name = name
        self.origin = origin
        self.context = context

    def _matches(self, item):
        if False:
            while True:
                i = 10
        if self.name != IGNORED and item.name != self.name:
            return False
        if self.origin != IGNORED and item.origin != self.origin:
            return False
        if self.context != IGNORED:
            for (key, name) in self.context.items():
                if key not in item.context:
                    return False
                if name != IGNORED and item.context[key] != name:
                    return False
        return True

    def describe_to(self, description):
        if False:
            i = 10
            return i + 15
        descriptors = []
        if self.name != IGNORED:
            descriptors.append('name is {}'.format(self.name))
        if self.origin != IGNORED:
            descriptors.append('origin is {}'.format(self.origin))
        if self.context != IGNORED:
            descriptors.append('context is ({})'.format(str(self.context)))
        item_description = ' and '.join(descriptors)
        description.append(item_description)

class MetricUpdateMatcher(BaseMatcher):
    """Matches a metrics update protocol buffer."""

    def __init__(self, cumulative=IGNORED, name=IGNORED, scalar=IGNORED, kind=IGNORED):
        if False:
            return 10
        'Creates a MetricUpdateMatcher.\n\n    Any property not passed in to the constructor will be ignored when matching.\n\n    Args:\n      cumulative: A boolean.\n      name: A MetricStructuredNameMatcher object that matches the name.\n      scalar: An integer with the metric update.\n      kind: A string defining the kind of counter.\n    '
        if name != IGNORED and (not isinstance(name, MetricStructuredNameMatcher)):
            raise ValueError('name must be a MetricStructuredNameMatcher.')
        self.cumulative = cumulative
        self.name = name
        self.scalar = scalar
        self.kind = kind

    def _matches(self, item):
        if False:
            return 10
        if self.cumulative != IGNORED and item.cumulative != self.cumulative:
            return False
        if self.name != IGNORED and (not self.name._matches(item.name)):
            return False
        if self.kind != IGNORED and item.kind != self.kind:
            return False
        if self.scalar != IGNORED:
            value_property = [p for p in item.scalar.object_value.properties if p.key == 'value']
            int_value = value_property[0].value.integer_value
            if self.scalar != int_value:
                return False
        return True

    def describe_to(self, description):
        if False:
            return 10
        descriptors = []
        if self.cumulative != IGNORED:
            descriptors.append('cumulative is {}'.format(self.cumulative))
        if self.name != IGNORED:
            descriptors.append('name is {}'.format(self.name))
        if self.scalar != IGNORED:
            descriptors.append('scalar is ({})'.format(str(self.scalar)))
        item_description = ' and '.join(descriptors)
        description.append(item_description)