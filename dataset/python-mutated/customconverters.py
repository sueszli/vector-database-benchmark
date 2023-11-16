from robot.utils import getdoc, seq2str, type_name

class CustomArgumentConverters:

    def __init__(self, converters):
        if False:
            return 10
        self.converters = converters

    @classmethod
    def from_dict(cls, converters, library=None):
        if False:
            while True:
                i = 10
        valid = []
        for (type_, conv) in converters.items():
            try:
                info = ConverterInfo.for_converter(type_, conv, library)
            except TypeError as err:
                if library is None:
                    raise
                library.report_error(str(err))
            else:
                valid.append(info)
        return cls(valid)

    def get_converter_info(self, type_):
        if False:
            print('Hello World!')
        if isinstance(type_, type):
            for conv in self.converters:
                if issubclass(type_, conv.type):
                    return conv
        return None

    def __iter__(self):
        if False:
            return 10
        return iter(self.converters)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.converters)

class ConverterInfo:

    def __init__(self, type, converter, value_types, library=None):
        if False:
            print('Hello World!')
        self.type = type
        self.converter = converter
        self.value_types = value_types
        self.library = library

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return type_name(self.type)

    @property
    def doc(self):
        if False:
            print('Hello World!')
        return getdoc(self.converter) or getdoc(self.type)

    @classmethod
    def for_converter(cls, type_, converter, library):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(type_, type):
            raise TypeError(f'Custom converters must be specified using types, got {type_name(type_)} {type_!r}.')
        if converter is None:

            def converter(arg):
                if False:
                    i = 10
                    return i + 15
                raise TypeError(f'Only {type_.__name__} instances are accepted, got {type_name(arg)}.')
        if not callable(converter):
            raise TypeError(f'Custom converters must be callable, converter for {type_name(type_)} is {type_name(converter)}.')
        spec = cls._get_arg_spec(converter)
        type_info = spec.types.get(spec.positional[0] if spec.positional else spec.var_positional)
        if type_info is None:
            accepts = ()
        elif type_info.is_union:
            accepts = type_info.nested
        else:
            accepts = (type_info,)
        accepts = tuple((info.type for info in accepts))
        pass_library = spec.minargs == 2 or spec.var_positional
        return cls(type_, converter, accepts, library if pass_library else None)

    @classmethod
    def _get_arg_spec(cls, converter):
        if False:
            i = 10
            return i + 15
        from .argumentparser import PythonArgumentParser
        spec = PythonArgumentParser(type='Converter').parse(converter)
        if spec.minargs > 2:
            required = seq2str([a for a in spec.positional if a not in spec.defaults])
            raise TypeError(f"Custom converters cannot have more than two mandatory arguments, '{converter.__name__}' has {required}.")
        if not spec.maxargs:
            raise TypeError(f"Custom converters must accept one positional argument, '{converter.__name__}' accepts none.")
        if spec.named_only and set(spec.named_only) - set(spec.defaults):
            required = seq2str(sorted(set(spec.named_only) - set(spec.defaults)))
            raise TypeError(f"Custom converters cannot have mandatory keyword-only arguments, '{converter.__name__}' has {required}.")
        return spec

    def convert(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not self.library:
            return self.converter(value)
        return self.converter(value, self.library.get_instance())