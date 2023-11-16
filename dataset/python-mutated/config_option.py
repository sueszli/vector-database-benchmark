"""Class to store a key-value pair for the config system."""
import datetime
import re
import textwrap
from typing import Any, Callable, Optional
from streamlit import util
from streamlit.case_converters import to_snake_case
from streamlit.errors import DeprecationError

class ConfigOption:
    '''Stores a Streamlit configuration option.

    A configuration option, like 'browser.serverPort', which indicates which port
    to use when connecting to the proxy. There are two ways to create a
    ConfigOption:

    Simple ConfigOptions are created as follows:

        ConfigOption('browser.serverPort',
            description = 'Connect to the proxy at this port.',
            default_val = 8501)

    More complex config options resolve their values at runtime as follows:

        @ConfigOption('browser.serverPort')
        def _proxy_port():
            """Connect to the proxy at this port.

            Defaults to 8501.
            """
            return 8501

    NOTE: For complex config options, the function is called each time the
    option.value is evaluated!

    Attributes
    ----------
    key : str
        The fully qualified section.name
    value : any
        The value for this option. If this is a complex config option then
        the callback is called EACH TIME value is evaluated.
    section : str
        The section of this option. Example: 'global'.
    name : str
        See __init__.
    description : str
        See __init__.
    where_defined : str
        Indicates which file set this config option.
        ConfigOption.DEFAULT_DEFINITION means this file.
    is_default: bool
        True if the config value is equal to its default value.
    visibility : {"visible", "hidden"}
        See __init__.
    scriptable : bool
        See __init__.
    deprecated: bool
        See __init__.
    deprecation_text : str or None
        See __init__.
    expiration_date : str or None
        See __init__.
    replaced_by : str or None
        See __init__.
    sensitive : bool
        See __init__.
    env_var: str
        The name of the environment variable that can be used to set the option.
    '''
    DEFAULT_DEFINITION = '<default>'
    STREAMLIT_DEFINITION = '<streamlit>'

    def __init__(self, key: str, description: Optional[str]=None, default_val: Optional[Any]=None, visibility: str='visible', scriptable: bool=False, deprecated: bool=False, deprecation_text: Optional[str]=None, expiration_date: Optional[str]=None, replaced_by: Optional[str]=None, type_: type=str, sensitive: bool=False):
        if False:
            while True:
                i = 10
        'Create a ConfigOption with the given name.\n\n        Parameters\n        ----------\n        key : str\n            Should be of the form "section.optionName"\n            Examples: server.name, deprecation.v1_0_featureName\n        description : str\n            Like a comment for the config option.\n        default_val : any\n            The value for this config option.\n        visibility : {"visible", "hidden"}\n            Whether this option should be shown to users.\n        scriptable : bool\n            Whether this config option can be set within a user script.\n        deprecated: bool\n            Whether this config option is deprecated.\n        deprecation_text : str or None\n            Required if deprecated == True. Set this to a string explaining\n            what to use instead.\n        expiration_date : str or None\n            Required if deprecated == True. set this to the date at which it\n            will no longer be accepted. Format: \'YYYY-MM-DD\'.\n        replaced_by : str or None\n            If this is option has been deprecated in favor or another option,\n            set this to the path to the new option. Example:\n            \'server.runOnSave\'. If this is set, the \'deprecated\' option\n            will automatically be set to True, and deprecation_text will have a\n            meaningful default (unless you override it).\n        type_ : one of str, int, float or bool\n            Useful to cast the config params sent by cmd option parameter.\n        sensitive: bool\n            Sensitive configuration options cannot be set by CLI parameter.\n        '
        self.key = key
        key_format = '(?P<section>\\_?[a-z][a-zA-Z0-9]*)\\.(?P<name>[a-z][a-zA-Z0-9]*)$'
        match = re.match(key_format, self.key)
        assert match, f'Key "{self.key}" has invalid format.'
        (self.section, self.name) = (match.group('section'), match.group('name'))
        self.description = description
        self.visibility = visibility
        self.scriptable = scriptable
        self.default_val = default_val
        self.deprecated = deprecated
        self.replaced_by = replaced_by
        self.is_default = True
        self._get_val_func: Optional[Callable[[], Any]] = None
        self.where_defined = ConfigOption.DEFAULT_DEFINITION
        self.type = type_
        self.sensitive = sensitive
        if self.replaced_by:
            self.deprecated = True
            if deprecation_text is None:
                deprecation_text = 'Replaced by %s.' % self.replaced_by
        if self.deprecated:
            assert expiration_date, 'expiration_date is required for deprecated items'
            assert deprecation_text, 'deprecation_text is required for deprecated items'
            self.expiration_date = expiration_date
            self.deprecation_text = textwrap.dedent(deprecation_text)
        self.set_value(default_val)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return util.repr_(self)

    def __call__(self, get_val_func: Callable[[], Any]) -> 'ConfigOption':
        if False:
            while True:
                i = 10
        'Assign a function to compute the value for this option.\n\n        This method is called when ConfigOption is used as a decorator.\n\n        Parameters\n        ----------\n        get_val_func : function\n            A function which will be called to get the value of this parameter.\n            We will use its docString as the description.\n\n        Returns\n        -------\n        ConfigOption\n            Returns self, which makes testing easier. See config_test.py.\n\n        '
        assert get_val_func.__doc__, 'Complex config options require doc strings for their description.'
        self.description = get_val_func.__doc__
        self._get_val_func = get_val_func
        return self

    @property
    def value(self) -> Any:
        if False:
            i = 10
            return i + 15
        'Get the value of this config option.'
        if self._get_val_func is None:
            return None
        return self._get_val_func()

    def set_value(self, value: Any, where_defined: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Set the value of this option.\n\n        Parameters\n        ----------\n        value\n            The new value for this parameter.\n        where_defined : str\n            New value to remember where this parameter was set.\n\n        '
        self._get_val_func = lambda : value
        if where_defined is None:
            self.where_defined = ConfigOption.DEFAULT_DEFINITION
        else:
            self.where_defined = where_defined
        self.is_default = value == self.default_val
        if self.deprecated and self.where_defined != ConfigOption.DEFAULT_DEFINITION:
            details = {'key': self.key, 'file': self.where_defined, 'explanation': self.deprecation_text, 'date': self.expiration_date}
            if self.is_expired():
                raise DeprecationError(textwrap.dedent('\n                    ════════════════════════════════════════════════\n                    %(key)s IS NO LONGER SUPPORTED.\n\n                    %(explanation)s\n\n                    Please update %(file)s.\n                    ════════════════════════════════════════════════\n                    ') % details)
            else:
                from streamlit.logger import get_logger
                LOGGER = get_logger(__name__)
                LOGGER.warning(textwrap.dedent('\n                    ════════════════════════════════════════════════\n                    %(key)s IS DEPRECATED.\n                    %(explanation)s\n\n                    This option will be removed on or after %(date)s.\n\n                    Please update %(file)s.\n                    ════════════════════════════════════════════════\n                    ') % details)

    def is_expired(self) -> bool:
        if False:
            return 10
        'Returns true if expiration_date is in the past.'
        if not self.deprecated:
            return False
        expiration_date = _parse_yyyymmdd_str(self.expiration_date)
        now = datetime.datetime.now()
        return now > expiration_date

    @property
    def env_var(self):
        if False:
            return 10
        '\n        Get the name of the environment variable that can be used to set the option.\n        '
        name = self.key.replace('.', '_')
        return f'STREAMLIT_{to_snake_case(name).upper()}'

def _parse_yyyymmdd_str(date_str: str) -> datetime.datetime:
    if False:
        print('Hello World!')
    (year, month, day) = [int(token) for token in date_str.split('-', 2)]
    return datetime.datetime(year, month, day)