"""Contains routines for printing protocol messages in text format.

Simple usage example:

  # Create a proto object and serialize it to a text proto string.
  message = my_proto_pb2.MyMessage(foo='bar')
  text_proto = text_format.MessageToString(message)

  # Parse a text proto string.
  message = text_format.Parse(text_proto, my_proto_pb2.MyMessage())
"""
__author__ = 'kenton@google.com (Kenton Varda)'
import io
import re
import six
if six.PY3:
    long = int
from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import text_encoding
__all__ = ['MessageToString', 'PrintMessage', 'PrintField', 'PrintFieldValue', 'Merge']
_INTEGER_CHECKERS = (type_checkers.Uint32ValueChecker(), type_checkers.Int32ValueChecker(), type_checkers.Uint64ValueChecker(), type_checkers.Int64ValueChecker())
_FLOAT_INFINITY = re.compile('-?inf(?:inity)?f?', re.IGNORECASE)
_FLOAT_NAN = re.compile('nanf?', re.IGNORECASE)
_FLOAT_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_FLOAT, descriptor.FieldDescriptor.CPPTYPE_DOUBLE])
_QUOTES = frozenset(("'", '"'))

class Error(Exception):
    """Top-level module error for text_format."""

class ParseError(Error):
    """Thrown in case of text parsing error."""

class TextWriter(object):

    def __init__(self, as_utf8):
        if False:
            return 10
        if six.PY2:
            self._writer = io.BytesIO()
        else:
            self._writer = io.StringIO()

    def write(self, val):
        if False:
            for i in range(10):
                print('nop')
        if six.PY2:
            if isinstance(val, six.text_type):
                val = val.encode('utf-8')
        return self._writer.write(val)

    def close(self):
        if False:
            print('Hello World!')
        return self._writer.close()

    def getvalue(self):
        if False:
            for i in range(10):
                print('nop')
        return self._writer.getvalue()

def MessageToString(message, as_utf8=False, as_one_line=False, pointy_brackets=False, use_index_order=False, float_format=None, use_field_number=False):
    if False:
        print('Hello World!')
    'Convert protobuf message to text format.\n\n  Floating point values can be formatted compactly with 15 digits of\n  precision (which is the most that IEEE 754 "double" can guarantee)\n  using float_format=\'.15g\'. To ensure that converting to text and back to a\n  proto will result in an identical value, float_format=\'.17g\' should be used.\n\n  Args:\n    message: The protocol buffers message.\n    as_utf8: Produce text output in UTF8 format.\n    as_one_line: Don\'t introduce newlines between fields.\n    pointy_brackets: If True, use angle brackets instead of curly braces for\n      nesting.\n    use_index_order: If True, print fields of a proto message using the order\n      defined in source code instead of the field number. By default, use the\n      field number order.\n    float_format: If set, use this to specify floating point number formatting\n      (per the "Format Specification Mini-Language"); otherwise, str() is used.\n    use_field_number: If True, print field numbers instead of names.\n\n  Returns:\n    A string of the text formatted protocol buffer message.\n  '
    out = TextWriter(as_utf8)
    printer = _Printer(out, 0, as_utf8, as_one_line, pointy_brackets, use_index_order, float_format, use_field_number)
    printer.PrintMessage(message)
    result = out.getvalue()
    out.close()
    if as_one_line:
        return result.rstrip()
    return result

def _IsMapEntry(field):
    if False:
        print('Hello World!')
    return field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry

def PrintMessage(message, out, indent=0, as_utf8=False, as_one_line=False, pointy_brackets=False, use_index_order=False, float_format=None, use_field_number=False):
    if False:
        for i in range(10):
            print('nop')
    printer = _Printer(out, indent, as_utf8, as_one_line, pointy_brackets, use_index_order, float_format, use_field_number)
    printer.PrintMessage(message)

def PrintField(field, value, out, indent=0, as_utf8=False, as_one_line=False, pointy_brackets=False, use_index_order=False, float_format=None):
    if False:
        return 10
    'Print a single field name/value pair.'
    printer = _Printer(out, indent, as_utf8, as_one_line, pointy_brackets, use_index_order, float_format)
    printer.PrintField(field, value)

def PrintFieldValue(field, value, out, indent=0, as_utf8=False, as_one_line=False, pointy_brackets=False, use_index_order=False, float_format=None):
    if False:
        while True:
            i = 10
    'Print a single field value (not including name).'
    printer = _Printer(out, indent, as_utf8, as_one_line, pointy_brackets, use_index_order, float_format)
    printer.PrintFieldValue(field, value)

class _Printer(object):
    """Text format printer for protocol message."""

    def __init__(self, out, indent=0, as_utf8=False, as_one_line=False, pointy_brackets=False, use_index_order=False, float_format=None, use_field_number=False):
        if False:
            return 10
        'Initialize the Printer.\n\n    Floating point values can be formatted compactly with 15 digits of\n    precision (which is the most that IEEE 754 "double" can guarantee)\n    using float_format=\'.15g\'. To ensure that converting to text and back to a\n    proto will result in an identical value, float_format=\'.17g\' should be used.\n\n    Args:\n      out: To record the text format result.\n      indent: The indent level for pretty print.\n      as_utf8: Produce text output in UTF8 format.\n      as_one_line: Don\'t introduce newlines between fields.\n      pointy_brackets: If True, use angle brackets instead of curly braces for\n        nesting.\n      use_index_order: If True, print fields of a proto message using the order\n        defined in source code instead of the field number. By default, use the\n        field number order.\n      float_format: If set, use this to specify floating point number formatting\n        (per the "Format Specification Mini-Language"); otherwise, str() is\n        used.\n      use_field_number: If True, print field numbers instead of names.\n    '
        self.out = out
        self.indent = indent
        self.as_utf8 = as_utf8
        self.as_one_line = as_one_line
        self.pointy_brackets = pointy_brackets
        self.use_index_order = use_index_order
        self.float_format = float_format
        self.use_field_number = use_field_number

    def PrintMessage(self, message):
        if False:
            return 10
        'Convert protobuf message to text format.\n\n    Args:\n      message: The protocol buffers message.\n    '
        fields = message.ListFields()
        if self.use_index_order:
            fields.sort(key=lambda x: x[0].index)
        for (field, value) in fields:
            if _IsMapEntry(field):
                for key in sorted(value):
                    entry_submsg = field.message_type._concrete_class(key=key, value=value[key])
                    self.PrintField(field, entry_submsg)
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                for element in value:
                    self.PrintField(field, element)
            else:
                self.PrintField(field, value)

    def PrintField(self, field, value):
        if False:
            return 10
        'Print a single field name/value pair.'
        out = self.out
        out.write(' ' * self.indent)
        if self.use_field_number:
            out.write(str(field.number))
        elif field.is_extension:
            out.write('[')
            if field.containing_type.GetOptions().message_set_wire_format and field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and (field.label == descriptor.FieldDescriptor.LABEL_OPTIONAL):
                out.write(field.message_type.full_name)
            else:
                out.write(field.full_name)
            out.write(']')
        elif field.type == descriptor.FieldDescriptor.TYPE_GROUP:
            out.write(field.message_type.name)
        else:
            out.write(field.name)
        if field.cpp_type != descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            out.write(': ')
        self.PrintFieldValue(field, value)
        if self.as_one_line:
            out.write(' ')
        else:
            out.write('\n')

    def PrintFieldValue(self, field, value):
        if False:
            print('Hello World!')
        'Print a single field value (not including name).\n\n    For repeated fields, the value should be a single element.\n\n    Args:\n      field: The descriptor of the field to be printed.\n      value: The value of the field.\n    '
        out = self.out
        if self.pointy_brackets:
            openb = '<'
            closeb = '>'
        else:
            openb = '{'
            closeb = '}'
        if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            if self.as_one_line:
                out.write(' %s ' % openb)
                self.PrintMessage(value)
                out.write(closeb)
            else:
                out.write(' %s\n' % openb)
                self.indent += 2
                self.PrintMessage(value)
                self.indent -= 2
                out.write(' ' * self.indent + closeb)
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
            enum_value = field.enum_type.values_by_number.get(value, None)
            if enum_value is not None:
                out.write(enum_value.name)
            else:
                out.write(str(value))
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
            out.write('"')
            if isinstance(value, six.text_type):
                out_value = value.encode('utf-8')
            else:
                out_value = value
            if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
                out_as_utf8 = False
            else:
                out_as_utf8 = self.as_utf8
            out.write(text_encoding.CEscape(out_value, out_as_utf8))
            out.write('"')
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
            if value:
                out.write('true')
            else:
                out.write('false')
        elif field.cpp_type in _FLOAT_TYPES and self.float_format is not None:
            out.write('{1:{0}}'.format(self.float_format, value))
        else:
            out.write(str(value))

def Parse(text, message, allow_unknown_extension=False, allow_field_number=False):
    if False:
        for i in range(10):
            print('nop')
    'Parses an text representation of a protocol message into a message.\n\n  Args:\n    text: Message text representation.\n    message: A protocol buffer message to merge into.\n    allow_unknown_extension: if True, skip over missing extensions and keep\n      parsing\n    allow_field_number: if True, both field number and field name are allowed.\n\n  Returns:\n    The same message passed as argument.\n\n  Raises:\n    ParseError: On text parsing problems.\n  '
    if not isinstance(text, str):
        text = text.decode('utf-8')
    return ParseLines(text.split('\n'), message, allow_unknown_extension, allow_field_number)

def Merge(text, message, allow_unknown_extension=False, allow_field_number=False):
    if False:
        i = 10
        return i + 15
    'Parses an text representation of a protocol message into a message.\n\n  Like Parse(), but allows repeated values for a non-repeated field, and uses\n  the last one.\n\n  Args:\n    text: Message text representation.\n    message: A protocol buffer message to merge into.\n    allow_unknown_extension: if True, skip over missing extensions and keep\n      parsing\n    allow_field_number: if True, both field number and field name are allowed.\n\n  Returns:\n    The same message passed as argument.\n\n  Raises:\n    ParseError: On text parsing problems.\n  '
    return MergeLines(text.split('\n'), message, allow_unknown_extension, allow_field_number)

def ParseLines(lines, message, allow_unknown_extension=False, allow_field_number=False):
    if False:
        for i in range(10):
            print('nop')
    "Parses an text representation of a protocol message into a message.\n\n  Args:\n    lines: An iterable of lines of a message's text representation.\n    message: A protocol buffer message to merge into.\n    allow_unknown_extension: if True, skip over missing extensions and keep\n      parsing\n    allow_field_number: if True, both field number and field name are allowed.\n\n  Returns:\n    The same message passed as argument.\n\n  Raises:\n    ParseError: On text parsing problems.\n  "
    parser = _Parser(allow_unknown_extension, allow_field_number)
    return parser.ParseLines(lines, message)

def MergeLines(lines, message, allow_unknown_extension=False, allow_field_number=False):
    if False:
        return 10
    "Parses an text representation of a protocol message into a message.\n\n  Args:\n    lines: An iterable of lines of a message's text representation.\n    message: A protocol buffer message to merge into.\n    allow_unknown_extension: if True, skip over missing extensions and keep\n      parsing\n    allow_field_number: if True, both field number and field name are allowed.\n\n  Returns:\n    The same message passed as argument.\n\n  Raises:\n    ParseError: On text parsing problems.\n  "
    parser = _Parser(allow_unknown_extension, allow_field_number)
    return parser.MergeLines(lines, message)

class _Parser(object):
    """Text format parser for protocol message."""

    def __init__(self, allow_unknown_extension=False, allow_field_number=False):
        if False:
            while True:
                i = 10
        self.allow_unknown_extension = allow_unknown_extension
        self.allow_field_number = allow_field_number

    def ParseFromString(self, text, message):
        if False:
            for i in range(10):
                print('nop')
        'Parses an text representation of a protocol message into a message.'
        if not isinstance(text, str):
            text = text.decode('utf-8')
        return self.ParseLines(text.split('\n'), message)

    def ParseLines(self, lines, message):
        if False:
            i = 10
            return i + 15
        'Parses an text representation of a protocol message into a message.'
        self._allow_multiple_scalars = False
        self._ParseOrMerge(lines, message)
        return message

    def MergeFromString(self, text, message):
        if False:
            i = 10
            return i + 15
        'Merges an text representation of a protocol message into a message.'
        return self._MergeLines(text.split('\n'), message)

    def MergeLines(self, lines, message):
        if False:
            return 10
        'Merges an text representation of a protocol message into a message.'
        self._allow_multiple_scalars = True
        self._ParseOrMerge(lines, message)
        return message

    def _ParseOrMerge(self, lines, message):
        if False:
            for i in range(10):
                print('nop')
        "Converts an text representation of a protocol message into a message.\n\n    Args:\n      lines: Lines of a message's text representation.\n      message: A protocol buffer message to merge into.\n\n    Raises:\n      ParseError: On text parsing problems.\n    "
        tokenizer = _Tokenizer(lines)
        while not tokenizer.AtEnd():
            self._MergeField(tokenizer, message)

    def _MergeField(self, tokenizer, message):
        if False:
            while True:
                i = 10
        'Merges a single protocol message field into a message.\n\n    Args:\n      tokenizer: A tokenizer to parse the field name and values.\n      message: A protocol message to record the data.\n\n    Raises:\n      ParseError: In case of text parsing problems.\n    '
        message_descriptor = message.DESCRIPTOR
        if hasattr(message_descriptor, 'syntax') and message_descriptor.syntax == 'proto3':
            self._allow_multiple_scalars = True
        if tokenizer.TryConsume('['):
            name = [tokenizer.ConsumeIdentifier()]
            while tokenizer.TryConsume('.'):
                name.append(tokenizer.ConsumeIdentifier())
            name = '.'.join(name)
            if not message_descriptor.is_extendable:
                raise tokenizer.ParseErrorPreviousToken('Message type "%s" does not have extensions.' % message_descriptor.full_name)
            field = message.Extensions._FindExtensionByName(name)
            if not field:
                if self.allow_unknown_extension:
                    field = None
                else:
                    raise tokenizer.ParseErrorPreviousToken('Extension "%s" not registered.' % name)
            elif message_descriptor != field.containing_type:
                raise tokenizer.ParseErrorPreviousToken('Extension "%s" does not extend message type "%s".' % (name, message_descriptor.full_name))
            tokenizer.Consume(']')
        else:
            name = tokenizer.ConsumeIdentifier()
            if self.allow_field_number and name.isdigit():
                number = ParseInteger(name, True, True)
                field = message_descriptor.fields_by_number.get(number, None)
                if not field and message_descriptor.is_extendable:
                    field = message.Extensions._FindExtensionByNumber(number)
            else:
                field = message_descriptor.fields_by_name.get(name, None)
                if not field:
                    field = message_descriptor.fields_by_name.get(name.lower(), None)
                    if field and field.type != descriptor.FieldDescriptor.TYPE_GROUP:
                        field = None
                if field and field.type == descriptor.FieldDescriptor.TYPE_GROUP and (field.message_type.name != name):
                    field = None
            if not field:
                raise tokenizer.ParseErrorPreviousToken('Message type "%s" has no field named "%s".' % (message_descriptor.full_name, name))
        if field:
            if not self._allow_multiple_scalars and field.containing_oneof:
                which_oneof = message.WhichOneof(field.containing_oneof.name)
                if which_oneof is not None and which_oneof != field.name:
                    raise tokenizer.ParseErrorPreviousToken('Field "%s" is specified along with field "%s", another member of oneof "%s" for message type "%s".' % (field.name, which_oneof, field.containing_oneof.name, message_descriptor.full_name))
            if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                tokenizer.TryConsume(':')
                merger = self._MergeMessageField
            else:
                tokenizer.Consume(':')
                merger = self._MergeScalarField
            if field.label == descriptor.FieldDescriptor.LABEL_REPEATED and tokenizer.TryConsume('['):
                while True:
                    merger(tokenizer, message, field)
                    if tokenizer.TryConsume(']'):
                        break
                    tokenizer.Consume(',')
            else:
                merger(tokenizer, message, field)
        else:
            assert self.allow_unknown_extension
            _SkipFieldContents(tokenizer)
        if not tokenizer.TryConsume(','):
            tokenizer.TryConsume(';')

    def _MergeMessageField(self, tokenizer, message, field):
        if False:
            print('Hello World!')
        'Merges a single scalar field into a message.\n\n    Args:\n      tokenizer: A tokenizer to parse the field value.\n      message: The message of which field is a member.\n      field: The descriptor of the field to be merged.\n\n    Raises:\n      ParseError: In case of text parsing problems.\n    '
        is_map_entry = _IsMapEntry(field)
        if tokenizer.TryConsume('<'):
            end_token = '>'
        else:
            tokenizer.Consume('{')
            end_token = '}'
        if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            if field.is_extension:
                sub_message = message.Extensions[field].add()
            elif is_map_entry:
                sub_message = field.message_type._concrete_class()
            else:
                sub_message = getattr(message, field.name).add()
        else:
            if field.is_extension:
                sub_message = message.Extensions[field]
            else:
                sub_message = getattr(message, field.name)
            sub_message.SetInParent()
        while not tokenizer.TryConsume(end_token):
            if tokenizer.AtEnd():
                raise tokenizer.ParseErrorPreviousToken('Expected "%s".' % (end_token,))
            self._MergeField(tokenizer, sub_message)
        if is_map_entry:
            value_cpptype = field.message_type.fields_by_name['value'].cpp_type
            if value_cpptype == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                value = getattr(message, field.name)[sub_message.key]
                value.MergeFrom(sub_message.value)
            else:
                getattr(message, field.name)[sub_message.key] = sub_message.value

    def _MergeScalarField(self, tokenizer, message, field):
        if False:
            print('Hello World!')
        'Merges a single scalar field into a message.\n\n    Args:\n      tokenizer: A tokenizer to parse the field value.\n      message: A protocol message to record the data.\n      field: The descriptor of the field to be merged.\n\n    Raises:\n      ParseError: In case of text parsing problems.\n      RuntimeError: On runtime errors.\n    '
        _ = self.allow_unknown_extension
        value = None
        if field.type in (descriptor.FieldDescriptor.TYPE_INT32, descriptor.FieldDescriptor.TYPE_SINT32, descriptor.FieldDescriptor.TYPE_SFIXED32):
            value = tokenizer.ConsumeInt32()
        elif field.type in (descriptor.FieldDescriptor.TYPE_INT64, descriptor.FieldDescriptor.TYPE_SINT64, descriptor.FieldDescriptor.TYPE_SFIXED64):
            value = tokenizer.ConsumeInt64()
        elif field.type in (descriptor.FieldDescriptor.TYPE_UINT32, descriptor.FieldDescriptor.TYPE_FIXED32):
            value = tokenizer.ConsumeUint32()
        elif field.type in (descriptor.FieldDescriptor.TYPE_UINT64, descriptor.FieldDescriptor.TYPE_FIXED64):
            value = tokenizer.ConsumeUint64()
        elif field.type in (descriptor.FieldDescriptor.TYPE_FLOAT, descriptor.FieldDescriptor.TYPE_DOUBLE):
            value = tokenizer.ConsumeFloat()
        elif field.type == descriptor.FieldDescriptor.TYPE_BOOL:
            value = tokenizer.ConsumeBool()
        elif field.type == descriptor.FieldDescriptor.TYPE_STRING:
            value = tokenizer.ConsumeString()
        elif field.type == descriptor.FieldDescriptor.TYPE_BYTES:
            value = tokenizer.ConsumeByteString()
        elif field.type == descriptor.FieldDescriptor.TYPE_ENUM:
            value = tokenizer.ConsumeEnum(field)
        else:
            raise RuntimeError('Unknown field type %d' % field.type)
        if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            if field.is_extension:
                message.Extensions[field].append(value)
            else:
                getattr(message, field.name).append(value)
        elif field.is_extension:
            if not self._allow_multiple_scalars and message.HasExtension(field):
                raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" extensions.' % (message.DESCRIPTOR.full_name, field.full_name))
            else:
                message.Extensions[field] = value
        elif not self._allow_multiple_scalars and message.HasField(field.name):
            raise tokenizer.ParseErrorPreviousToken('Message type "%s" should not have multiple "%s" fields.' % (message.DESCRIPTOR.full_name, field.name))
        else:
            setattr(message, field.name, value)

def _SkipFieldContents(tokenizer):
    if False:
        return 10
    'Skips over contents (value or message) of a field.\n\n  Args:\n    tokenizer: A tokenizer to parse the field name and values.\n  '
    if tokenizer.TryConsume(':') and (not tokenizer.LookingAt('{')) and (not tokenizer.LookingAt('<')):
        _SkipFieldValue(tokenizer)
    else:
        _SkipFieldMessage(tokenizer)

def _SkipField(tokenizer):
    if False:
        return 10
    'Skips over a complete field (name and value/message).\n\n  Args:\n    tokenizer: A tokenizer to parse the field name and values.\n  '
    if tokenizer.TryConsume('['):
        tokenizer.ConsumeIdentifier()
        while tokenizer.TryConsume('.'):
            tokenizer.ConsumeIdentifier()
        tokenizer.Consume(']')
    else:
        tokenizer.ConsumeIdentifier()
    _SkipFieldContents(tokenizer)
    if not tokenizer.TryConsume(','):
        tokenizer.TryConsume(';')

def _SkipFieldMessage(tokenizer):
    if False:
        i = 10
        return i + 15
    'Skips over a field message.\n\n  Args:\n    tokenizer: A tokenizer to parse the field name and values.\n  '
    if tokenizer.TryConsume('<'):
        delimiter = '>'
    else:
        tokenizer.Consume('{')
        delimiter = '}'
    while not tokenizer.LookingAt('>') and (not tokenizer.LookingAt('}')):
        _SkipField(tokenizer)
    tokenizer.Consume(delimiter)

def _SkipFieldValue(tokenizer):
    if False:
        i = 10
        return i + 15
    'Skips over a field value.\n\n  Args:\n    tokenizer: A tokenizer to parse the field name and values.\n\n  Raises:\n    ParseError: In case an invalid field value is found.\n  '
    if tokenizer.TryConsumeByteString():
        while tokenizer.TryConsumeByteString():
            pass
        return
    if not tokenizer.TryConsumeIdentifier() and (not tokenizer.TryConsumeInt64()) and (not tokenizer.TryConsumeUint64()) and (not tokenizer.TryConsumeFloat()):
        raise ParseError('Invalid field value: ' + tokenizer.token)

class _Tokenizer(object):
    """Protocol buffer text representation tokenizer.

  This class handles the lower level string parsing by splitting it into
  meaningful tokens.

  It was directly ported from the Java protocol buffer API.
  """
    _WHITESPACE = re.compile('(\\s|(#.*$))+', re.MULTILINE)
    _TOKEN = re.compile('|'.join(['[a-zA-Z_][0-9a-zA-Z_+-]*', '([0-9+-]|(\\.[0-9]))[0-9a-zA-Z_.+-]*'] + ['{qt}([^{qt}\\n\\\\]|\\\\.)*({qt}|\\\\?$)'.format(qt=mark) for mark in _QUOTES]))
    _IDENTIFIER = re.compile('\\w+')

    def __init__(self, lines):
        if False:
            return 10
        self._position = 0
        self._line = -1
        self._column = 0
        self._token_start = None
        self.token = ''
        self._lines = iter(lines)
        self._current_line = ''
        self._previous_line = 0
        self._previous_column = 0
        self._more_lines = True
        self._SkipWhitespace()
        self.NextToken()

    def LookingAt(self, token):
        if False:
            return 10
        return self.token == token

    def AtEnd(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks the end of the text was reached.\n\n    Returns:\n      True iff the end was reached.\n    '
        return not self.token

    def _PopLine(self):
        if False:
            print('Hello World!')
        while len(self._current_line) <= self._column:
            try:
                self._current_line = next(self._lines)
            except StopIteration:
                self._current_line = ''
                self._more_lines = False
                return
            else:
                self._line += 1
                self._column = 0

    def _SkipWhitespace(self):
        if False:
            i = 10
            return i + 15
        while True:
            self._PopLine()
            match = self._WHITESPACE.match(self._current_line, self._column)
            if not match:
                break
            length = len(match.group(0))
            self._column += length

    def TryConsume(self, token):
        if False:
            for i in range(10):
                print('nop')
        'Tries to consume a given piece of text.\n\n    Args:\n      token: Text to consume.\n\n    Returns:\n      True iff the text was consumed.\n    '
        if self.token == token:
            self.NextToken()
            return True
        return False

    def Consume(self, token):
        if False:
            while True:
                i = 10
        "Consumes a piece of text.\n\n    Args:\n      token: Text to consume.\n\n    Raises:\n      ParseError: If the text couldn't be consumed.\n    "
        if not self.TryConsume(token):
            raise self._ParseError('Expected "%s".' % token)

    def TryConsumeIdentifier(self):
        if False:
            i = 10
            return i + 15
        try:
            self.ConsumeIdentifier()
            return True
        except ParseError:
            return False

    def ConsumeIdentifier(self):
        if False:
            return 10
        "Consumes protocol message field identifier.\n\n    Returns:\n      Identifier string.\n\n    Raises:\n      ParseError: If an identifier couldn't be consumed.\n    "
        result = self.token
        if not self._IDENTIFIER.match(result):
            raise self._ParseError('Expected identifier.')
        self.NextToken()
        return result

    def ConsumeInt32(self):
        if False:
            print('Hello World!')
        "Consumes a signed 32bit integer number.\n\n    Returns:\n      The integer parsed.\n\n    Raises:\n      ParseError: If a signed 32bit integer couldn't be consumed.\n    "
        try:
            result = ParseInteger(self.token, is_signed=True, is_long=False)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def ConsumeUint32(self):
        if False:
            return 10
        "Consumes an unsigned 32bit integer number.\n\n    Returns:\n      The integer parsed.\n\n    Raises:\n      ParseError: If an unsigned 32bit integer couldn't be consumed.\n    "
        try:
            result = ParseInteger(self.token, is_signed=False, is_long=False)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def TryConsumeInt64(self):
        if False:
            print('Hello World!')
        try:
            self.ConsumeInt64()
            return True
        except ParseError:
            return False

    def ConsumeInt64(self):
        if False:
            return 10
        "Consumes a signed 64bit integer number.\n\n    Returns:\n      The integer parsed.\n\n    Raises:\n      ParseError: If a signed 64bit integer couldn't be consumed.\n    "
        try:
            result = ParseInteger(self.token, is_signed=True, is_long=True)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def TryConsumeUint64(self):
        if False:
            return 10
        try:
            self.ConsumeUint64()
            return True
        except ParseError:
            return False

    def ConsumeUint64(self):
        if False:
            return 10
        "Consumes an unsigned 64bit integer number.\n\n    Returns:\n      The integer parsed.\n\n    Raises:\n      ParseError: If an unsigned 64bit integer couldn't be consumed.\n    "
        try:
            result = ParseInteger(self.token, is_signed=False, is_long=True)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def TryConsumeFloat(self):
        if False:
            print('Hello World!')
        try:
            self.ConsumeFloat()
            return True
        except ParseError:
            return False

    def ConsumeFloat(self):
        if False:
            for i in range(10):
                print('nop')
        "Consumes an floating point number.\n\n    Returns:\n      The number parsed.\n\n    Raises:\n      ParseError: If a floating point number couldn't be consumed.\n    "
        try:
            result = ParseFloat(self.token)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def ConsumeBool(self):
        if False:
            i = 10
            return i + 15
        "Consumes a boolean value.\n\n    Returns:\n      The bool parsed.\n\n    Raises:\n      ParseError: If a boolean value couldn't be consumed.\n    "
        try:
            result = ParseBool(self.token)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def TryConsumeByteString(self):
        if False:
            return 10
        try:
            self.ConsumeByteString()
            return True
        except ParseError:
            return False

    def ConsumeString(self):
        if False:
            i = 10
            return i + 15
        "Consumes a string value.\n\n    Returns:\n      The string parsed.\n\n    Raises:\n      ParseError: If a string value couldn't be consumed.\n    "
        the_bytes = self.ConsumeByteString()
        try:
            return six.text_type(the_bytes, 'utf-8')
        except UnicodeDecodeError as e:
            raise self._StringParseError(e)

    def ConsumeByteString(self):
        if False:
            for i in range(10):
                print('nop')
        "Consumes a byte array value.\n\n    Returns:\n      The array parsed (as a string).\n\n    Raises:\n      ParseError: If a byte array value couldn't be consumed.\n    "
        the_list = [self._ConsumeSingleByteString()]
        while self.token and self.token[0] in _QUOTES:
            the_list.append(self._ConsumeSingleByteString())
        return b''.join(the_list)

    def _ConsumeSingleByteString(self):
        if False:
            i = 10
            return i + 15
        'Consume one token of a string literal.\n\n    String literals (whether bytes or text) can come in multiple adjacent\n    tokens which are automatically concatenated, like in C or Python.  This\n    method only consumes one token.\n\n    Returns:\n      The token parsed.\n    Raises:\n      ParseError: When the wrong format data is found.\n    '
        text = self.token
        if len(text) < 1 or text[0] not in _QUOTES:
            raise self._ParseError('Expected string but found: %r' % (text,))
        if len(text) < 2 or text[-1] != text[0]:
            raise self._ParseError('String missing ending quote: %r' % (text,))
        try:
            result = text_encoding.CUnescape(text[1:-1])
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def ConsumeEnum(self, field):
        if False:
            return 10
        try:
            result = ParseEnum(field, self.token)
        except ValueError as e:
            raise self._ParseError(str(e))
        self.NextToken()
        return result

    def ParseErrorPreviousToken(self, message):
        if False:
            print('Hello World!')
        'Creates and *returns* a ParseError for the previously read token.\n\n    Args:\n      message: A message to set for the exception.\n\n    Returns:\n      A ParseError instance.\n    '
        return ParseError('%d:%d : %s' % (self._previous_line + 1, self._previous_column + 1, message))

    def _ParseError(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Creates and *returns* a ParseError for the current token.'
        return ParseError('%d:%d : %s' % (self._line + 1, self._column + 1, message))

    def _StringParseError(self, e):
        if False:
            for i in range(10):
                print('nop')
        return self._ParseError("Couldn't parse string: " + str(e))

    def NextToken(self):
        if False:
            print('Hello World!')
        'Reads the next meaningful token.'
        self._previous_line = self._line
        self._previous_column = self._column
        self._column += len(self.token)
        self._SkipWhitespace()
        if not self._more_lines:
            self.token = ''
            return
        match = self._TOKEN.match(self._current_line, self._column)
        if match:
            token = match.group(0)
            self.token = token
        else:
            self.token = self._current_line[self._column]

def ParseInteger(text, is_signed=False, is_long=False):
    if False:
        for i in range(10):
            print('nop')
    'Parses an integer.\n\n  Args:\n    text: The text to parse.\n    is_signed: True if a signed integer must be parsed.\n    is_long: True if a long integer must be parsed.\n\n  Returns:\n    The integer value.\n\n  Raises:\n    ValueError: Thrown Iff the text is not a valid integer.\n  '
    try:
        if is_long:
            result = long(text, 0)
        else:
            result = int(text, 0)
    except ValueError:
        raise ValueError("Couldn't parse integer: %s" % text)
    checker = _INTEGER_CHECKERS[2 * int(is_long) + int(is_signed)]
    checker.CheckValue(result)
    return result

def ParseFloat(text):
    if False:
        print('Hello World!')
    "Parse a floating point number.\n\n  Args:\n    text: Text to parse.\n\n  Returns:\n    The number parsed.\n\n  Raises:\n    ValueError: If a floating point number couldn't be parsed.\n  "
    try:
        return float(text)
    except ValueError:
        if _FLOAT_INFINITY.match(text):
            if text[0] == '-':
                return float('-inf')
            else:
                return float('inf')
        elif _FLOAT_NAN.match(text):
            return float('nan')
        else:
            try:
                return float(text.rstrip('f'))
            except ValueError:
                raise ValueError("Couldn't parse float: %s" % text)

def ParseBool(text):
    if False:
        print('Hello World!')
    'Parse a boolean value.\n\n  Args:\n    text: Text to parse.\n\n  Returns:\n    Boolean values parsed\n\n  Raises:\n    ValueError: If text is not a valid boolean.\n  '
    if text in ('true', 't', '1'):
        return True
    elif text in ('false', 'f', '0'):
        return False
    else:
        raise ValueError('Expected "true" or "false".')

def ParseEnum(field, value):
    if False:
        i = 10
        return i + 15
    'Parse an enum value.\n\n  The value can be specified by a number (the enum value), or by\n  a string literal (the enum name).\n\n  Args:\n    field: Enum field descriptor.\n    value: String value.\n\n  Returns:\n    Enum value number.\n\n  Raises:\n    ValueError: If the enum value could not be parsed.\n  '
    enum_descriptor = field.enum_type
    try:
        number = int(value, 0)
    except ValueError:
        enum_value = enum_descriptor.values_by_name.get(value, None)
        if enum_value is None:
            raise ValueError('Enum type "%s" has no value named %s.' % (enum_descriptor.full_name, value))
    else:
        enum_value = enum_descriptor.values_by_number.get(number, None)
        if enum_value is None:
            raise ValueError('Enum type "%s" has no value with number %d.' % (enum_descriptor.full_name, number))
    return enum_value.number