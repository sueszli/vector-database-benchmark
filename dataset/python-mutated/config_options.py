from typing import TypeVar, Generic
from pyflink.java_gateway import get_gateway
T = TypeVar('T')
__all__ = ['ConfigOptions', 'ConfigOption']

class ConfigOptions(object):
    """
    {@code ConfigOptions} are used to build a :class:`~pyflink.common.ConfigOption`. The option is
    typically built in one of the following patterns:

    Example:
    ::

        # simple string-valued option with a default value
        >>> ConfigOptions.key("tmp.dir").string_type().default_value("/tmp")
        # simple integer-valued option with a default value
        >>> ConfigOptions.key("application.parallelism").int_type().default_value(100)
        # option with no default value
        >>> ConfigOptions.key("user.name").string_type().no_default_value()
    """

    def __init__(self, j_config_options):
        if False:
            while True:
                i = 10
        self._j_config_options = j_config_options

    @staticmethod
    def key(key: str):
        if False:
            return 10
        '\n        Starts building a new ConfigOption.\n\n        :param key: The key for the config option.\n        :return: The builder for the config option with the given key.\n        '
        gateway = get_gateway()
        j_option_builder = gateway.jvm.org.apache.flink.configuration.ConfigOptions.key(key)
        return ConfigOptions.OptionBuilder(j_option_builder)

    class OptionBuilder(object):

        def __init__(self, j_option_builder):
            if False:
                print('Hello World!')
            self._j_option_builder = j_option_builder

        def boolean_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[bool]':
            if False:
                while True:
                    i = 10
            '\n            Defines that the value of the option should be of bool type.\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.booleanType())

        def int_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[int]':
            if False:
                print('Hello World!')
            '\n            Defines that the value of the option should be of int type\n            (from -2,147,483,648 to 2,147,483,647).\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.intType())

        def long_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[int]':
            if False:
                return 10
            '\n            Defines that the value of the option should be of int type\n            (from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807).\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.longType())

        def float_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[float]':
            if False:
                while True:
                    i = 10
            '\n            Defines that the value of the option should be of float type\n            (4-byte single precision floating point number).\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.floatType())

        def double_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[float]':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Defines that the value of the option should be of float Double} type\n            (8-byte double precision floating point number).\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.doubleType())

        def string_type(self) -> 'ConfigOptions.TypedConfigOptionBuilder[str]':
            if False:
                return 10
            '\n            Defines that the value of the option should be of str type.\n            '
            return ConfigOptions.TypedConfigOptionBuilder(self._j_option_builder.stringType())

    class TypedConfigOptionBuilder(Generic[T]):

        def __init__(self, j_typed_config_option_builder):
            if False:
                for i in range(10):
                    print('nop')
            self._j_typed_config_option_builder = j_typed_config_option_builder

        def default_value(self, value: T) -> 'ConfigOption[T]':
            if False:
                print('Hello World!')
            return ConfigOption(self._j_typed_config_option_builder.defaultValue(value))

        def no_default_value(self) -> 'ConfigOption[str]':
            if False:
                for i in range(10):
                    print('nop')
            return ConfigOption(self._j_typed_config_option_builder.noDefaultValue())

class ConfigOption(Generic[T]):
    """
    A {@code ConfigOption} describes a configuration parameter. It encapsulates the configuration
    key, deprecated older versions of the key, and an optional default value for the configuration
    parameter.

    {@code ConfigOptions} are built via the ConfigOptions class. Once created, a config
    option is immutable.
    """

    def __init__(self, j_config_option):
        if False:
            print('Hello World!')
        self._j_config_option = j_config_option