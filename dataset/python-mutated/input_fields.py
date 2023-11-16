import click
from archivy.click_web.web_click_types import EmailParamType, PasswordParamType

class FieldId:
    """
    Extract/serialize information from the encoded form input field name
    the parts:
        [command_index].[opt_or_arg_index].[click_type].[html_input_type].[opt_or_arg_name]
    e.g.
        "0.0.option.text.text.--an-option"
        "0.1.argument.file[rb].text.an-argument"
    """
    SEPARATOR = '.'

    def __init__(self, command_index, param_index, param_type, click_type, nargs, form_type, name, key=None):
        if False:
            while True:
                i = 10
        'the int index of the command it belongs to'
        self.command_index = int(command_index)
        'the int index for the ordering of paramters/arguments'
        self.param_index = int(param_index)
        'Type of option (argument, option, flag)'
        self.param_type = param_type
        'Type of option (file, text)'
        self.click_type = click_type
        'nargs value (-1 is variardic)'
        self.nargs = int(nargs)
        'Type of html input type'
        self.form_type = form_type
        'The actual command line option (--debug)'
        self.name = name
        'The actual form id'
        self.key = key if key else str(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.SEPARATOR.join((str(p) for p in (self.command_index, self.param_index, self.param_type, self.click_type, self.nargs, self.form_type, self.name)))

    @classmethod
    def from_string(cls, field_info_as_string) -> 'FieldId':
        if False:
            while True:
                i = 10
        args = field_info_as_string.split(cls.SEPARATOR) + [field_info_as_string]
        return cls(*args)

class NotSupported(ValueError):
    pass

class BaseInput:
    param_type_cls = None

    def __init__(self, ctx, param: click.Parameter, command_index, param_index):
        if False:
            while True:
                i = 10
        self.ctx = ctx
        self.param = param
        self.command_index = command_index
        self.param_index = param_index
        if not self.is_supported():
            raise NotSupported()

    def is_supported(self):
        if False:
            while True:
                i = 10
        return isinstance(self.param.type, self.param_type_cls)

    @property
    def fields(self) -> dict:
        if False:
            return 10
        field = {}
        param = self.param
        field['param'] = param.param_type_name
        if param.param_type_name == 'option':
            name = '--{}'.format(self._to_cmd_line_name(param.name))
            field['value'] = param.default if param.default else ''
            field['checked'] = 'checked="checked"' if param.default else ''
            field['desc'] = param.help
            field['help'] = param.get_help_record(self.ctx)
        elif param.param_type_name == 'argument':
            name = self._to_cmd_line_name(param.name)
            field['value'] = param.default
            field['checked'] = ''
            field['help'] = ''
        field['name'] = self._build_name(name)
        field['required'] = param.required
        field['nargs'] = param.nargs
        field['human_readable_name'] = param.human_readable_name.replace('_', ' ')
        field.update(self.type_attrs)
        return field

    @property
    def type_attrs(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Here the input type and type specific information should be retuned as a dict\n        :return:\n        '
        raise NotImplementedError()

    def _to_cmd_line_name(self, name: str) -> str:
        if False:
            print('Hello World!')
        return name.replace('_', '-')

    def _build_name(self, name: str):
        if False:
            print('Hello World!')
        '\n        Construct a name to use for field in form that have information about\n        what sub-command it belongs order index (for later sorting) and type of parameter.\n        '
        if self.param.param_type_name == 'option':
            param_type = 'flag' if self.param.is_bool_flag else 'option'
        else:
            param_type = self.param.param_type_name
        click_type = self.type_attrs['click_type']
        form_type = self.type_attrs['type']
        return str(FieldId(self.command_index, self.param_index, param_type, click_type, self.param.nargs, form_type, name))

class ChoiceInput(BaseInput):
    param_type_cls = click.Choice

    @property
    def type_attrs(self):
        if False:
            while True:
                i = 10
        type_attrs = {}
        type_attrs['type'] = 'option'
        type_attrs['options'] = self.param.type.choices
        type_attrs['default'] = self.param.default
        type_attrs['click_type'] = 'choice'
        return type_attrs

class FlagInput(BaseInput):

    def is_supported(self):
        if False:
            i = 10
            return i + 15
        return self.param.param_type_name == 'option' and self.param.is_bool_flag

    @property
    def type_attrs(self):
        if False:
            while True:
                i = 10
        type_attrs = {}
        type_attrs['type'] = 'checkbox'
        type_attrs['click_type'] = 'bool_flag'
        type_attrs['value'] = self.param.opts[0]
        type_attrs['on_flag'] = self.param.opts[0]
        if self.param.secondary_opts:
            type_attrs['off_flag'] = self.param.secondary_opts[0]
        return type_attrs

class IntInput(BaseInput):
    param_type_cls = click.types.IntParamType

    @property
    def type_attrs(self):
        if False:
            return 10
        type_attrs = {}
        type_attrs['type'] = 'number'
        type_attrs['step'] = '1'
        type_attrs['click_type'] = 'int'
        return type_attrs

class FloatInput(BaseInput):
    param_type_cls = click.types.FloatParamType

    @property
    def type_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        type_attrs = {}
        type_attrs['type'] = 'number'
        type_attrs['step'] = 'any'
        type_attrs['click_type'] = 'float'
        return type_attrs

class FolderInput(BaseInput):

    def is_supported(self):
        if False:
            return 10
        if isinstance(self.param.type, click.Path):
            if self.param.type.dir_okay:
                return True
        return False

    @property
    def type_attrs(self):
        if False:
            i = 10
            return i + 15
        type_attrs = {}
        mode = 'r' if self.param.type.exists else 'w'
        type_attrs['click_type'] = f'path[{mode}]'
        if self.param.type.exists:
            type_attrs['accept'] = 'application/zip'
            type_attrs['type'] = 'file'
        else:
            type_attrs['type'] = 'hidden'
        return type_attrs

class FileInput(BaseInput):

    def is_supported(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.param.type, click.File):
            return True
        elif isinstance(self.param.type, click.Path):
            if self.param.type.file_okay:
                return True
        return False

    @property
    def type_attrs(self):
        if False:
            i = 10
            return i + 15
        type_attrs = {}
        if isinstance(self.param.type, click.File):
            mode = self.param.type.mode
        elif isinstance(self.param.type, click.Path):
            mode = 'w' if self.param.type.writable else ''
            mode += 'r' if self.param.type.readable else ''
        else:
            raise NotSupported(f'Illegal param type. Got type: {self.param.type}')
        type_attrs['click_type'] = f'file[{mode}]'
        if 'r' not in mode:
            if self.param.required:
                type_attrs['type'] = 'hidden'
            else:
                type_attrs['type'] = 'text'
        else:
            type_attrs['type'] = 'file'
        return type_attrs

class EmailInput(BaseInput):
    param_type_cls = EmailParamType

    @property
    def type_attrs(self):
        if False:
            print('Hello World!')
        type_attrs = {}
        type_attrs['type'] = 'email'
        type_attrs['click_type'] = 'email'
        return type_attrs

class PasswordInput(BaseInput):
    param_type_cls = PasswordParamType

    @property
    def type_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        type_attrs = {}
        type_attrs['type'] = 'password'
        type_attrs['click_type'] = 'password'
        return type_attrs

class DefaultInput(BaseInput):
    param_type_cls = click.ParamType

    @property
    def type_attrs(self):
        if False:
            print('Hello World!')
        type_attrs = {}
        type_attrs['type'] = 'text'
        type_attrs['click_type'] = 'text'
        return type_attrs
'\nThe types of inputs we support form inputs listed in priority order\n(first that matches will be selected).\nTo add new Input handling for html forms for custom Parameter types\njust Subclass BaseInput and insert the class in the list.\n'
INPUT_TYPES = [ChoiceInput, FlagInput, IntInput, FloatInput, FolderInput, FileInput, EmailInput, PasswordInput]
_DEFAULT_INPUT = [DefaultInput]

def get_input_field(ctx: click.Context, param: click.Parameter, command_index, param_index) -> dict:
    if False:
        while True:
            i = 10
    '\n    Convert a click.Parameter into a dict structure describing a html form option\n    '
    for input_cls in INPUT_TYPES + _DEFAULT_INPUT:
        try:
            input_type = input_cls(ctx, param, command_index, param_index)
        except NotSupported:
            pass
        else:
            fields = input_type.fields
            return fields
    raise NotSupported(f'No Form input type not supported: {param}')