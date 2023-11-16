"""Enhanced YAML parsing for Spack.

- ``load()`` preserves YAML Marks on returned objects -- this allows
  us to access file and line information later.

- ``Our load methods use ``OrderedDict`` class instead of YAML's
  default unorderd dict.

"""
import collections
import collections.abc
import copy
import ctypes
import enum
import functools
import io
import re
from typing import IO, List, Optional
import ruamel.yaml
from ruamel.yaml import comments, constructor, emitter, error, representer
from llnl.util.tty.color import cextra, clen, colorize
import spack.error
__all__ = ['load', 'dump', 'SpackYAMLError']

class syaml_dict(collections.OrderedDict):

    def __repr__(self):
        if False:
            while True:
                i = 10
        mappings = (f'{k!r}: {v!r}' for (k, v) in self.items())
        return '{%s}' % ', '.join(mappings)

class syaml_list(list):
    __repr__ = list.__repr__

class syaml_str(str):
    __repr__ = str.__repr__

class syaml_int(int):
    __repr__ = int.__repr__
syaml_types = {syaml_str: str, syaml_int: int, syaml_dict: dict, syaml_list: list}
markable_types = set(syaml_types) | {comments.CommentedSeq, comments.CommentedMap}

def syaml_type(obj):
    if False:
        for i in range(10):
            print('nop')
    'Get the corresponding syaml wrapper type for a primitive type.\n\n    Return:\n        (object): syaml-typed copy of object, or the obj if no wrapper\n    '
    for (syaml_t, t) in syaml_types.items():
        if type(obj) is not bool and isinstance(obj, t):
            return syaml_t(obj) if type(obj) is not syaml_t else obj
    return obj

def markable(obj):
    if False:
        for i in range(10):
            print('nop')
    'Whether an object can be marked.'
    return type(obj) in markable_types

def mark(obj, node):
    if False:
        i = 10
        return i + 15
    'Add start and end markers to an object.'
    if hasattr(node, 'start_mark'):
        obj._start_mark = node.start_mark
    elif hasattr(node, '_start_mark'):
        obj._start_mark = node._start_mark
    if hasattr(node, 'end_mark'):
        obj._end_mark = node.end_mark
    elif hasattr(node, '_end_mark'):
        obj._end_mark = node._end_mark

def marked(obj):
    if False:
        while True:
            i = 10
    'Whether an object has been marked by spack_yaml.'
    return hasattr(obj, '_start_mark') and obj._start_mark or (hasattr(obj, '_end_mark') and obj._end_mark)

class OrderedLineConstructor(constructor.RoundTripConstructor):
    """YAML loader specifically intended for reading Spack configuration
    files. It preserves order and line numbers. It also has special-purpose
    logic for handling dictionary keys that indicate a Spack config
    override: namely any key that contains an "extra" ':' character.

    Mappings read in by this loader behave like an ordered dict.
    Sequences, mappings, and strings also have new attributes,
    ``_start_mark`` and ``_end_mark``, that preserve YAML line
    information in the output data.

    """

    def construct_yaml_str(self, node):
        if False:
            while True:
                i = 10
        value = super().construct_yaml_str(node)
        if value and value.endswith(':') and ('@' not in value):
            value = syaml_str(value[:-1])
            value.override = True
        else:
            value = syaml_str(value)
        mark(value, node)
        return value

    def construct_yaml_seq(self, node):
        if False:
            return 10
        gen = super().construct_yaml_seq(node)
        data = next(gen)
        if markable(data):
            mark(data, node)
        yield data
        for x in gen:
            pass

    def construct_yaml_map(self, node):
        if False:
            while True:
                i = 10
        gen = super().construct_yaml_map(node)
        data = next(gen)
        if markable(data):
            mark(data, node)
        yield data
        for x in gen:
            pass
OrderedLineConstructor.add_constructor('tag:yaml.org,2002:map', OrderedLineConstructor.construct_yaml_map)
OrderedLineConstructor.add_constructor('tag:yaml.org,2002:seq', OrderedLineConstructor.construct_yaml_seq)
OrderedLineConstructor.add_constructor('tag:yaml.org,2002:str', OrderedLineConstructor.construct_yaml_str)

class OrderedLineRepresenter(representer.RoundTripRepresenter):
    """Representer that preserves ordering and formats ``syaml_*`` objects.

    This representer preserves insertion ordering ``syaml_dict`` objects
    when they're written out.  It also has some custom formatters
    for ``syaml_*`` objects so that they are formatted like their
    regular Python equivalents, instead of ugly YAML pyobjects.
    """

    def ignore_aliases(self, _data):
        if False:
            return 10
        'Make the dumper NEVER print YAML aliases.'
        return True

    def represent_data(self, data):
        if False:
            while True:
                i = 10
        result = super().represent_data(data)
        if data is None:
            result.value = syaml_str('null')
        return result

    def represent_str(self, data):
        if False:
            return 10
        if hasattr(data, 'override') and data.override:
            data = data + ':'
        return super().represent_str(data)

class SafeRepresenter(representer.RoundTripRepresenter):

    def ignore_aliases(self, _data):
        if False:
            i = 10
            return i + 15
        'Make the dumper NEVER print YAML aliases.'
        return True
representer.RoundTripRepresenter.add_representer(syaml_dict, representer.RoundTripRepresenter.represent_dict)
representer.RoundTripRepresenter.add_representer(syaml_list, representer.RoundTripRepresenter.represent_list)
representer.RoundTripRepresenter.add_representer(syaml_int, representer.RoundTripRepresenter.represent_int)
representer.RoundTripRepresenter.add_representer(syaml_str, representer.RoundTripRepresenter.represent_str)
OrderedLineRepresenter.add_representer(syaml_str, OrderedLineRepresenter.represent_str)
maxint = 2 ** (ctypes.sizeof(ctypes.c_int) * 8 - 1) - 1

def return_string_when_no_stream(func):
    if False:
        i = 10
        return i + 15

    @functools.wraps(func)
    def wrapper(data, stream=None, **kwargs):
        if False:
            print('Hello World!')
        if stream:
            return func(data, stream=stream, **kwargs)
        stream = io.StringIO()
        func(data, stream=stream, **kwargs)
        return stream.getvalue()
    return wrapper

@return_string_when_no_stream
def dump(data, stream=None, default_flow_style=False):
    if False:
        print('Hello World!')
    handler = ConfigYAML(yaml_type=YAMLType.GENERIC_YAML)
    handler.default_flow_style = default_flow_style
    handler.width = maxint
    return handler.dump(data, stream=stream)

def file_line(mark):
    if False:
        print('Hello World!')
    'Format a mark as <file>:<line> information.'
    result = mark.name
    if mark.line:
        result += ':' + str(mark.line)
    return result
_ANNOTATIONS: List[str] = []

class LineAnnotationRepresenter(OrderedLineRepresenter):
    """Representer that generates per-line annotations.

    Annotations are stored in the ``_annotations`` global.  After one
    dump pass, the strings in ``_annotations`` will correspond one-to-one
    with the lines output by the dumper.

    LineAnnotationDumper records blame information after each line is
    generated. As each line is parsed, it saves file/line info for each
    object printed. At the end of each line, it creates an annotation
    based on the saved mark and stores it in ``_annotations``.

    For an example of how to use this, see ``dump_annotated()``, which
    writes to a ``StringIO`` then joins the lines from that with
    annotations.
    """

    def represent_data(self, data):
        if False:
            i = 10
            return i + 15
        'Force syaml_str to be passed through with marks.'
        result = super().represent_data(data)
        if data is None:
            result.value = syaml_str('null')
        elif isinstance(result.value, str):
            result.value = syaml_str(data)
        if markable(result.value):
            mark(result.value, data)
        return result

class LineAnnotationEmitter(emitter.Emitter):
    saved = None

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        del _ANNOTATIONS[:]
        self.colors = 'KgrbmcyGRBMCY'
        self.filename_colors = {}

    def process_scalar(self):
        if False:
            while True:
                i = 10
        super().process_scalar()
        if marked(self.event.value):
            self.saved = self.event.value

    def write_line_break(self):
        if False:
            while True:
                i = 10
        super().write_line_break()
        if self.saved is None:
            _ANNOTATIONS.append(colorize('@K{---}'))
            return
        if self.saved:
            mark = self.saved._start_mark
            color = self.filename_colors.get(mark.name)
            if not color:
                ncolors = len(self.colors)
                color = self.colors[len(self.filename_colors) % ncolors]
                self.filename_colors[mark.name] = color
            fmt = '@%s{%%s}' % color
            ann = fmt % mark.name
            if mark.line is not None:
                ann += ':@c{%s}' % (mark.line + 1)
            _ANNOTATIONS.append(colorize(ann))
        else:
            _ANNOTATIONS.append('')

    def write_comment(self, comment, pre=False):
        if False:
            while True:
                i = 10
        pass

class YAMLType(enum.Enum):
    """YAML configurations handled by Spack"""
    GENERIC_YAML = enum.auto()
    SPACK_CONFIG_FILE = enum.auto()
    ANNOTATED_SPACK_CONFIG_FILE = enum.auto()

class ConfigYAML:
    """Handles the loading and dumping of Spack's YAML files."""

    def __init__(self, yaml_type: YAMLType) -> None:
        if False:
            print('Hello World!')
        self.yaml = ruamel.yaml.YAML(typ='rt', pure=True)
        if yaml_type == YAMLType.GENERIC_YAML:
            self.yaml.Representer = SafeRepresenter
        elif yaml_type == YAMLType.ANNOTATED_SPACK_CONFIG_FILE:
            self.yaml.Representer = LineAnnotationRepresenter
            self.yaml.Emitter = LineAnnotationEmitter
            self.yaml.Constructor = OrderedLineConstructor
        else:
            self.yaml.Representer = OrderedLineRepresenter
            self.yaml.Constructor = OrderedLineConstructor

    def load(self, stream: IO):
        if False:
            while True:
                i = 10
        'Loads the YAML data from a stream and returns it.\n\n        Args:\n            stream: stream to load from.\n\n        Raises:\n            SpackYAMLError: if anything goes wrong while loading\n        '
        try:
            return self.yaml.load(stream)
        except error.MarkedYAMLError as e:
            msg = 'error parsing YAML'
            error_mark = e.context_mark if e.context_mark else e.problem_mark
            if error_mark:
                (line, column) = (error_mark.line, error_mark.column)
                msg += f': near {error_mark.name}, {str(line)}, {str(column)}'
            else:
                msg += f': {stream.name}'
            msg += f': {e.problem}'
            raise SpackYAMLError(msg, e) from e
        except Exception as e:
            msg = 'cannot load Spack YAML configuration'
            raise SpackYAMLError(msg, e) from e

    def dump(self, data, stream: Optional[IO]=None, *, transform=None) -> None:
        if False:
            print('Hello World!')
        'Dumps the YAML data to a stream.\n\n        Args:\n            data: data to be dumped\n            stream: stream to dump the data into.\n\n        Raises:\n            SpackYAMLError: if anything goes wrong while dumping\n        '
        try:
            return self.yaml.dump(data, stream=stream, transform=transform)
        except Exception as e:
            msg = 'cannot dump Spack YAML configuration'
            raise SpackYAMLError(msg, str(e)) from e

    def as_string(self, data) -> str:
        if False:
            return 10
        'Returns a string representing the YAML data passed as input.'
        result = io.StringIO()
        self.dump(data, stream=result)
        return result.getvalue()

def deepcopy(data):
    if False:
        i = 10
        return i + 15
    'Returns a deepcopy of the input YAML data.'
    result = copy.deepcopy(data)
    if isinstance(result, comments.CommentedMap):
        extracted_comments = extract_comments(data)
        if extracted_comments:
            set_comments(result, data_comments=extracted_comments)
    return result

def load_config(str_or_file):
    if False:
        return 10
    'Load but modify the loader instance so that it will add __line__\n    attributes to the returned object.'
    handler = ConfigYAML(yaml_type=YAMLType.SPACK_CONFIG_FILE)
    return handler.load(str_or_file)

def load(*args, **kwargs):
    if False:
        while True:
            i = 10
    handler = ConfigYAML(yaml_type=YAMLType.GENERIC_YAML)
    return handler.load(*args, **kwargs)

@return_string_when_no_stream
def dump_config(data, stream, *, default_flow_style=False, blame=False):
    if False:
        i = 10
        return i + 15
    if blame:
        handler = ConfigYAML(yaml_type=YAMLType.ANNOTATED_SPACK_CONFIG_FILE)
        handler.yaml.default_flow_style = default_flow_style
        return _dump_annotated(handler, data, stream)
    handler = ConfigYAML(yaml_type=YAMLType.SPACK_CONFIG_FILE)
    handler.yaml.default_flow_style = default_flow_style
    return handler.dump(data, stream)

def _dump_annotated(handler, data, stream=None):
    if False:
        i = 10
        return i + 15
    sio = io.StringIO()
    handler.dump(data, sio)
    lines = re.split('\\n+', sio.getvalue().rstrip())
    getvalue = None
    if stream is None:
        stream = io.StringIO()
        getvalue = stream.getvalue
    width = max((clen(a) for a in _ANNOTATIONS))
    formats = ['%%-%ds  %%s\n' % (width + cextra(a)) for a in _ANNOTATIONS]
    for (f, a, l) in zip(formats, _ANNOTATIONS, lines):
        stream.write(f % (a, l))
    if getvalue:
        return getvalue()

def sorted_dict(dict_like):
    if False:
        while True:
            i = 10
    'Return an ordered dict with all the fields sorted recursively.\n\n    Args:\n        dict_like (dict): dictionary to be sorted\n\n    Returns:\n        dictionary sorted recursively\n    '
    result = syaml_dict(sorted(dict_like.items()))
    for (key, value) in result.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = sorted_dict(value)
    return result

def extract_comments(data):
    if False:
        return 10
    'Extract and returns comments from some YAML data'
    return getattr(data, comments.Comment.attrib, None)

def set_comments(data, *, data_comments):
    if False:
        print('Hello World!')
    'Set comments on some YAML data'
    return setattr(data, comments.Comment.attrib, data_comments)

def name_mark(name):
    if False:
        while True:
            i = 10
    'Returns a mark with just a name'
    return error.StringMark(name, None, None, None, None, None)

class SpackYAMLError(spack.error.SpackError):
    """Raised when there are issues with YAML parsing."""

    def __init__(self, msg, yaml_error):
        if False:
            while True:
                i = 10
        super().__init__(msg, str(yaml_error))