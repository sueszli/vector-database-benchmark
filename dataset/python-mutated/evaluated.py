class Evaluated(object):

    def __init__(self, expected_type, default, name=None):
        if False:
            return 10
        self.expected_type = expected_type
        self.default = default
        self.name = name or 'evaled_property_{}'.format(id(self))
        self.eval_function = self.default_eval_func

    @property
    def name_raw(self):
        if False:
            print('Hello World!')
        return '_' + self.name

    def default_eval_func(self, instance):
        if False:
            while True:
                i = 10
        raw = getattr(instance, self.name_raw)
        try:
            value = instance.parent_block.evaluate(raw)
        except Exception as error:
            if raw:
                instance.add_error_message(f"Failed to eval '{raw}': ({type(error)}) {error}")
            return self.default
        if not isinstance(value, self.expected_type):
            instance.add_error_message("Can not cast evaluated value '{}' to type {}".format(value, self.expected_type))
            return self.default
        return value

    def __call__(self, func):
        if False:
            return 10
        self.name = func.__name__
        self.eval_function = func
        return self

    def __get__(self, instance, owner):
        if False:
            i = 10
            return i + 15
        if instance is None:
            return self
        attribs = instance.__dict__
        try:
            value = attribs[self.name]
        except KeyError:
            value = attribs[self.name] = self.eval_function(instance)
        return value

    def __set__(self, instance, value):
        if False:
            for i in range(10):
                print('nop')
        attribs = instance.__dict__
        value = value or self.default
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            attribs[self.name_raw] = value[2:-1].strip()
            attribs.pop(self.name, None)
        else:
            attribs[self.name] = type(self.default)(value)

    def __delete__(self, instance):
        if False:
            while True:
                i = 10
        attribs = instance.__dict__
        if self.name_raw in attribs:
            attribs.pop(self.name, None)

class EvaluatedEnum(Evaluated):

    def __init__(self, allowed_values, default=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(allowed_values, str):
            allowed_values = set(allowed_values.split())
        self.allowed_values = allowed_values
        default = default if default is not None else next(iter(self.allowed_values))
        super(EvaluatedEnum, self).__init__(str, default, name)

    def default_eval_func(self, instance):
        if False:
            i = 10
            return i + 15
        value = super(EvaluatedEnum, self).default_eval_func(instance)
        if value not in self.allowed_values:
            instance.add_error_message("Value '{}' not in allowed values".format(value))
            return self.default
        return value

class EvaluatedPInt(Evaluated):

    def __init__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        super(EvaluatedPInt, self).__init__(int, 1, name)

    def default_eval_func(self, instance):
        if False:
            return 10
        value = super(EvaluatedPInt, self).default_eval_func(instance)
        if value < 1:
            return self.default
        return value

class EvaluatedFlag(Evaluated):

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        super(EvaluatedFlag, self).__init__((bool, int), False, name)

def setup_names(cls):
    if False:
        for i in range(10):
            print('nop')
    for (name, attrib) in cls.__dict__.items():
        if isinstance(attrib, Evaluated):
            attrib.name = name
    return cls