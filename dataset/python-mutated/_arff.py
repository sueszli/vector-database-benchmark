"""
The liac-arff module implements functions to read and write ARFF files in
Python. It was created in the Connectionist Artificial Intelligence Laboratory
(LIAC), which takes place at the Federal University of Rio Grande do Sul
(UFRGS), in Brazil.

ARFF (Attribute-Relation File Format) is an file format specially created for
describe datasets which are commonly used for machine learning experiments and
software. This file format was created to be used in Weka, the best
representative software for machine learning automated experiments.

An ARFF file can be divided into two sections: header and data. The Header
describes the metadata of the dataset, including a general description of the
dataset, its name and its attributes. The source below is an example of a
header section in a XOR dataset::

    %
    % XOR Dataset
    %
    % Created by Renato Pereira
    %            rppereira@inf.ufrgs.br
    %            http://inf.ufrgs.br/~rppereira
    %
    %
    @RELATION XOR

    @ATTRIBUTE input1 REAL
    @ATTRIBUTE input2 REAL
    @ATTRIBUTE y REAL

The Data section of an ARFF file describes the observations of the dataset, in
the case of XOR dataset::

    @DATA
    0.0,0.0,0.0
    0.0,1.0,1.0
    1.0,0.0,1.0
    1.0,1.0,0.0
    %
    %
    %

Notice that several lines are starting with an ``%`` symbol, denoting a
comment, thus, lines with ``%`` at the beginning will be ignored, except by the
description part at the beginning of the file. The declarations ``@RELATION``,
``@ATTRIBUTE``, and ``@DATA`` are all case insensitive and obligatory.

For more information and details about the ARFF file description, consult
http://www.cs.waikato.ac.nz/~ml/weka/arff.html


ARFF Files in Python
~~~~~~~~~~~~~~~~~~~~

This module uses built-ins python objects to represent a deserialized ARFF
file. A dictionary is used as the container of the data and metadata of ARFF,
and have the following keys:

- **description**: (OPTIONAL) a string with the description of the dataset.
- **relation**: (OBLIGATORY) a string with the name of the dataset.
- **attributes**: (OBLIGATORY) a list of attributes with the following
  template::

    (attribute_name, attribute_type)

  the attribute_name is a string, and attribute_type must be an string
  or a list of strings.
- **data**: (OBLIGATORY) a list of data instances. Each data instance must be
  a list with values, depending on the attributes.

The above keys must follow the case which were described, i.e., the keys are
case sensitive. The attribute type ``attribute_type`` must be one of these
strings (they are not case sensitive): ``NUMERIC``, ``INTEGER``, ``REAL`` or
``STRING``. For nominal attributes, the ``atribute_type`` must be a list of
strings.

In this format, the XOR dataset presented above can be represented as a python
object as::

    xor_dataset = {
        'description': 'XOR Dataset',
        'relation': 'XOR',
        'attributes': [
            ('input1', 'REAL'),
            ('input2', 'REAL'),
            ('y', 'REAL'),
        ],
        'data': [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ]
    }


Features
~~~~~~~~

This module provides several features, including:

- Read and write ARFF files using python built-in structures, such dictionaries
  and lists;
- Supports `scipy.sparse.coo <http://docs.scipy
  .org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html#scipy.sparse.coo_matrix>`_
  and lists of dictionaries as used by SVMLight
- Supports the following attribute types: NUMERIC, REAL, INTEGER, STRING, and
  NOMINAL;
- Has an interface similar to other built-in modules such as ``json``, or
  ``zipfile``;
- Supports read and write the descriptions of files;
- Supports missing values and names with spaces;
- Supports unicode values and names;
- Fully compatible with Python 2.7+, Python 3.5+, pypy and pypy3;
- Under `MIT License <http://opensource.org/licenses/MIT>`_

"""
__author__ = 'Renato de Pontes Pereira, Matthias Feurer, Joel Nothman'
__author_email__ = 'renato.ppontes@gmail.com, feurerm@informatik.uni-freiburg.de, joel.nothman@gmail.com'
__version__ = '2.4.0'
import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
_SIMPLE_TYPES = ['NUMERIC', 'REAL', 'INTEGER', 'STRING']
_TK_DESCRIPTION = '%'
_TK_COMMENT = '%'
_TK_RELATION = '@RELATION'
_TK_ATTRIBUTE = '@ATTRIBUTE'
_TK_DATA = '@DATA'
_RE_RELATION = re.compile('^([^\\{\\}%,\\s]*|\\".*\\"|\\\'.*\\\')$', re.UNICODE)
_RE_ATTRIBUTE = re.compile('^(\\".*\\"|\\\'.*\\\'|[^\\{\\}%,\\s]*)\\s+(.+)$', re.UNICODE)
_RE_QUOTE_CHARS = re.compile('["\\\'\\\\\\s%,\\000-\\031]', re.UNICODE)
_RE_ESCAPE_CHARS = re.compile('(?=["\\\'\\\\%])|[\\n\\r\\t\\000-\\031]')
_RE_SPARSE_LINE = re.compile('^\\s*\\{.*\\}\\s*$', re.UNICODE)
_RE_NONTRIVIAL_DATA = re.compile('["\'{}\\s]', re.UNICODE)
ArffDenseDataType = Iterator[List]
ArffSparseDataType = Tuple[List, ...]
if TYPE_CHECKING:
    from typing_extensions import TypedDict

    class ArffContainerType(TypedDict):
        description: str
        relation: str
        attributes: List
        data: Union[ArffDenseDataType, ArffSparseDataType]
else:
    ArffContainerType = Dict[str, Any]

def _build_re_values():
    if False:
        i = 10
        return i + 15
    quoted_re = '\n                    "      # open quote followed by zero or more of:\n                    (?:\n                        (?<!\\\\)    # no additional backslash\n                        (?:\\\\\\\\)*  # maybe escaped backslashes\n                        \\\\"        # escaped quote\n                    |\n                        \\\\[^"]     # escaping a non-quote\n                    |\n                        [^"\\\\]     # non-quote char\n                    )*\n                    "      # close quote\n                    '
    value_re = '(?:\n        %s|          # a value may be surrounded by "\n        %s|          # or by \'\n        [^,\\s"\'{}]+  # or may contain no characters requiring quoting\n        )' % (quoted_re, quoted_re.replace('"', "'"))
    dense = re.compile("(?x)\n        ,                # may follow ','\n        \\s*\n        ((?=,)|$|{value_re})  # empty or value\n        |\n        (\\S.*)           # error\n        ".format(value_re=value_re))
    sparse = re.compile("(?x)\n        (?:^\\s*\\{|,)   # may follow ',', or '{' at line start\n        \\s*\n        (\\d+)          # attribute key\n        \\s+\n        (%(value_re)s) # value\n        |\n        (?!}\\s*$)      # not an error if it's }$\n        (?!^\\s*{\\s*}\\s*$)  # not an error if it's ^{}$\n        \\S.*           # error\n        " % {'value_re': value_re})
    return (dense, sparse)
(_RE_DENSE_VALUES, _RE_SPARSE_KEY_VALUES) = _build_re_values()
_ESCAPE_SUB_MAP = {'\\\\': '\\', '\\"': '"', "\\'": "'", '\\t': '\t', '\\n': '\n', '\\r': '\r', '\\b': '\x08', '\\f': '\x0c', '\\%': '%'}
_UNESCAPE_SUB_MAP = {chr(i): '\\%03o' % i for i in range(32)}
_UNESCAPE_SUB_MAP.update({v: k for (k, v) in _ESCAPE_SUB_MAP.items()})
_UNESCAPE_SUB_MAP[''] = '\\'
_ESCAPE_SUB_MAP.update({'\\%d' % i: chr(i) for i in range(10)})

def _escape_sub_callback(match):
    if False:
        print('Hello World!')
    s = match.group()
    if len(s) == 2:
        try:
            return _ESCAPE_SUB_MAP[s]
        except KeyError:
            raise ValueError('Unsupported escape sequence: %s' % s)
    if s[1] == 'u':
        return chr(int(s[2:], 16))
    else:
        return chr(int(s[1:], 8))

def _unquote(v):
    if False:
        i = 10
        return i + 15
    if v[:1] in ('"', "'"):
        return re.sub('\\\\([0-9]{1,3}|u[0-9a-f]{4}|.)', _escape_sub_callback, v[1:-1])
    elif v in ('?', ''):
        return None
    else:
        return v

def _parse_values(s):
    if False:
        while True:
            i = 10
    '(INTERNAL) Split a line into a list of values'
    if not _RE_NONTRIVIAL_DATA.search(s):
        return [None if s in ('?', '') else s for s in next(csv.reader([s]))]
    (values, errors) = zip(*_RE_DENSE_VALUES.findall(',' + s))
    if not any(errors):
        return [_unquote(v) for v in values]
    if _RE_SPARSE_LINE.match(s):
        try:
            return {int(k): _unquote(v) for (k, v) in _RE_SPARSE_KEY_VALUES.findall(s)}
        except ValueError:
            for match in _RE_SPARSE_KEY_VALUES.finditer(s):
                if not match.group(1):
                    raise BadLayout('Error parsing %r' % match.group())
            raise BadLayout('Unknown parsing error')
    else:
        for match in _RE_DENSE_VALUES.finditer(s):
            if match.group(2):
                raise BadLayout('Error parsing %r' % match.group())
        raise BadLayout('Unknown parsing error')
DENSE = 0
COO = 1
LOD = 2
DENSE_GEN = 3
LOD_GEN = 4
_SUPPORTED_DATA_STRUCTURES = [DENSE, COO, LOD, DENSE_GEN, LOD_GEN]

class ArffException(Exception):
    message: Optional[str] = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.line = -1

    def __str__(self):
        if False:
            return 10
        return self.message % self.line

class BadRelationFormat(ArffException):
    """Error raised when the relation declaration is in an invalid format."""
    message = 'Bad @RELATION format, at line %d.'

class BadAttributeFormat(ArffException):
    """Error raised when some attribute declaration is in an invalid format."""
    message = 'Bad @ATTRIBUTE format, at line %d.'

class BadDataFormat(ArffException):
    """Error raised when some data instance is in an invalid format."""

    def __init__(self, value):
        if False:
            return 10
        super().__init__()
        self.message = 'Bad @DATA instance format in line %d: ' + '%s' % value

class BadAttributeType(ArffException):
    """Error raised when some invalid type is provided into the attribute
    declaration."""
    message = 'Bad @ATTRIBUTE type, at line %d.'

class BadAttributeName(ArffException):
    """Error raised when an attribute name is provided twice the attribute
    declaration."""

    def __init__(self, value, value2):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.message = 'Bad @ATTRIBUTE name %s at line' % value + ' %d, this name is already in use in line' + ' %d.' % value2

class BadNominalValue(ArffException):
    """Error raised when a value in used in some data instance but is not
    declared into it respective attribute declaration."""

    def __init__(self, value):
        if False:
            while True:
                i = 10
        super().__init__()
        self.message = 'Data value %s not found in nominal declaration, ' % value + 'at line %d.'

class BadNominalFormatting(ArffException):
    """Error raised when a nominal value with space is not properly quoted."""

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.message = 'Nominal data value "%s" not properly quoted in line ' % value + '%d.'

class BadNumericalValue(ArffException):
    """Error raised when and invalid numerical value is used in some data
    instance."""
    message = 'Invalid numerical value, at line %d.'

class BadStringValue(ArffException):
    """Error raise when a string contains space but is not quoted."""
    message = 'Invalid string value at line %d.'

class BadLayout(ArffException):
    """Error raised when the layout of the ARFF file has something wrong."""
    message = 'Invalid layout of the ARFF file, at line %d.'

    def __init__(self, msg=''):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if msg:
            self.message = BadLayout.message + ' ' + msg.replace('%', '%%')

class BadObject(ArffException):
    """Error raised when the object representing the ARFF file has something
    wrong."""

    def __init__(self, msg='Invalid object.'):
        if False:
            for i in range(10):
                print('nop')
        self.msg = msg

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s' % self.msg

def _unescape_sub_callback(match):
    if False:
        return 10
    return _UNESCAPE_SUB_MAP[match.group()]

def encode_string(s):
    if False:
        for i in range(10):
            print('nop')
    if _RE_QUOTE_CHARS.search(s):
        return "'%s'" % _RE_ESCAPE_CHARS.sub(_unescape_sub_callback, s)
    return s

class EncodedNominalConversor:

    def __init__(self, values):
        if False:
            return 10
        self.values = {v: i for (i, v) in enumerate(values)}
        self.values[0] = 0

    def __call__(self, value):
        if False:
            print('Hello World!')
        try:
            return self.values[value]
        except KeyError:
            raise BadNominalValue(value)

class NominalConversor:

    def __init__(self, values):
        if False:
            return 10
        self.values = set(values)
        self.zero_value = values[0]

    def __call__(self, value):
        if False:
            return 10
        if value not in self.values:
            if value == 0:
                return self.zero_value
            raise BadNominalValue(value)
        return str(value)

class DenseGeneratorData:
    """Internal helper class to allow for different matrix types without
    making the code a huge collection of if statements."""

    def decode_rows(self, stream, conversors):
        if False:
            print('Hello World!')
        for row in stream:
            values = _parse_values(row)
            if isinstance(values, dict):
                if values and max(values) >= len(conversors):
                    raise BadDataFormat(row)
                values = [values[i] if i in values else 0 for i in range(len(conversors))]
            elif len(values) != len(conversors):
                raise BadDataFormat(row)
            yield self._decode_values(values, conversors)

    @staticmethod
    def _decode_values(values, conversors):
        if False:
            for i in range(10):
                print('nop')
        try:
            values = [None if value is None else conversor(value) for (conversor, value) in zip(conversors, values)]
        except ValueError as exc:
            if 'float: ' in str(exc):
                raise BadNumericalValue()
        return values

    def encode_data(self, data, attributes):
        if False:
            return 10
        '(INTERNAL) Encodes a line of data.\n\n        Data instances follow the csv format, i.e, attribute values are\n        delimited by commas. After converted from csv.\n\n        :param data: a list of values.\n        :param attributes: a list of attributes. Used to check if data is valid.\n        :return: a string with the encoded data line.\n        '
        current_row = 0
        for inst in data:
            if len(inst) != len(attributes):
                raise BadObject('Instance %d has %d attributes, expected %d' % (current_row, len(inst), len(attributes)))
            new_data = []
            for value in inst:
                if value is None or value == '' or value != value:
                    s = '?'
                else:
                    s = encode_string(str(value))
                new_data.append(s)
            current_row += 1
            yield ','.join(new_data)

class _DataListMixin:
    """Mixin to return a list from decode_rows instead of a generator"""

    def decode_rows(self, stream, conversors):
        if False:
            return 10
        return list(super().decode_rows(stream, conversors))

class Data(_DataListMixin, DenseGeneratorData):
    pass

class COOData:

    def decode_rows(self, stream, conversors):
        if False:
            while True:
                i = 10
        (data, rows, cols) = ([], [], [])
        for (i, row) in enumerate(stream):
            values = _parse_values(row)
            if not isinstance(values, dict):
                raise BadLayout()
            if not values:
                continue
            (row_cols, values) = zip(*sorted(values.items()))
            try:
                values = [value if value is None else conversors[key](value) for (key, value) in zip(row_cols, values)]
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                raise BadDataFormat(row)
            data.extend(values)
            rows.extend([i] * len(values))
            cols.extend(row_cols)
        return (data, rows, cols)

    def encode_data(self, data, attributes):
        if False:
            print('Hello World!')
        num_attributes = len(attributes)
        new_data = []
        current_row = 0
        row = data.row
        col = data.col
        data = data.data
        if not all((row[i] <= row[i + 1] for i in range(len(row) - 1))):
            raise ValueError('liac-arff can only output COO matrices with sorted rows.')
        for (v, col, row) in zip(data, col, row):
            if row > current_row:
                while current_row < row:
                    yield ' '.join(['{', ','.join(new_data), '}'])
                    new_data = []
                    current_row += 1
            if col >= num_attributes:
                raise BadObject('Instance %d has at least %d attributes, expected %d' % (current_row, col + 1, num_attributes))
            if v is None or v == '' or v != v:
                s = '?'
            else:
                s = encode_string(str(v))
            new_data.append('%d %s' % (col, s))
        yield ' '.join(['{', ','.join(new_data), '}'])

class LODGeneratorData:

    def decode_rows(self, stream, conversors):
        if False:
            for i in range(10):
                print('nop')
        for row in stream:
            values = _parse_values(row)
            if not isinstance(values, dict):
                raise BadLayout()
            try:
                yield {key: None if value is None else conversors[key](value) for (key, value) in values.items()}
            except ValueError as exc:
                if 'float: ' in str(exc):
                    raise BadNumericalValue()
                raise
            except IndexError:
                raise BadDataFormat(row)

    def encode_data(self, data, attributes):
        if False:
            i = 10
            return i + 15
        current_row = 0
        num_attributes = len(attributes)
        for row in data:
            new_data = []
            if len(row) > 0 and max(row) >= num_attributes:
                raise BadObject('Instance %d has %d attributes, expected %d' % (current_row, max(row) + 1, num_attributes))
            for col in sorted(row):
                v = row[col]
                if v is None or v == '' or v != v:
                    s = '?'
                else:
                    s = encode_string(str(v))
                new_data.append('%d %s' % (col, s))
            current_row += 1
            yield ' '.join(['{', ','.join(new_data), '}'])

class LODData(_DataListMixin, LODGeneratorData):
    pass

def _get_data_object_for_decoding(matrix_type):
    if False:
        for i in range(10):
            print('nop')
    if matrix_type == DENSE:
        return Data()
    elif matrix_type == COO:
        return COOData()
    elif matrix_type == LOD:
        return LODData()
    elif matrix_type == DENSE_GEN:
        return DenseGeneratorData()
    elif matrix_type == LOD_GEN:
        return LODGeneratorData()
    else:
        raise ValueError('Matrix type %s not supported.' % str(matrix_type))

def _get_data_object_for_encoding(matrix):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(matrix, 'format'):
        if matrix.format == 'coo':
            return COOData()
        else:
            raise ValueError('Cannot guess matrix format!')
    elif isinstance(matrix[0], dict):
        return LODData()
    else:
        return Data()

class ArffDecoder:
    """An ARFF decoder."""

    def __init__(self):
        if False:
            return 10
        'Constructor.'
        self._conversors = []
        self._current_line = 0

    def _decode_comment(self, s):
        if False:
            return 10
        '(INTERNAL) Decodes a comment line.\n\n        Comments are single line strings starting, obligatorily, with the ``%``\n        character, and can have any symbol, including whitespaces or special\n        characters.\n\n        This method must receive a normalized string, i.e., a string without\n        padding, including the "\r\n" characters.\n\n        :param s: a normalized string.\n        :return: a string with the decoded comment.\n        '
        res = re.sub('^\\%( )?', '', s)
        return res

    def _decode_relation(self, s):
        if False:
            print('Hello World!')
        '(INTERNAL) Decodes a relation line.\n\n        The relation declaration is a line with the format ``@RELATION\n        <relation-name>``, where ``relation-name`` is a string. The string must\n        start with alphabetic character and must be quoted if the name includes\n        spaces, otherwise this method will raise a `BadRelationFormat` exception.\n\n        This method must receive a normalized string, i.e., a string without\n        padding, including the "\r\n" characters.\n\n        :param s: a normalized string.\n        :return: a string with the decoded relation name.\n        '
        (_, v) = s.split(' ', 1)
        v = v.strip()
        if not _RE_RELATION.match(v):
            raise BadRelationFormat()
        res = str(v.strip('"\''))
        return res

    def _decode_attribute(self, s):
        if False:
            for i in range(10):
                print('nop')
        '(INTERNAL) Decodes an attribute line.\n\n        The attribute is the most complex declaration in an arff file. All\n        attributes must follow the template::\n\n             @attribute <attribute-name> <datatype>\n\n        where ``attribute-name`` is a string, quoted if the name contains any\n        whitespace, and ``datatype`` can be:\n\n        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.\n        - Strings as ``STRING``.\n        - Dates (NOT IMPLEMENTED).\n        - Nominal attributes with format:\n\n            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}\n\n        The nominal names follow the rules for the attribute names, i.e., they\n        must be quoted if the name contains whitespaces.\n\n        This method must receive a normalized string, i.e., a string without\n        padding, including the "\r\n" characters.\n\n        :param s: a normalized string.\n        :return: a tuple (ATTRIBUTE_NAME, TYPE_OR_VALUES).\n        '
        (_, v) = s.split(' ', 1)
        v = v.strip()
        m = _RE_ATTRIBUTE.match(v)
        if not m:
            raise BadAttributeFormat()
        (name, type_) = m.groups()
        name = str(name.strip('"\''))
        if type_[:1] == '{' and type_[-1:] == '}':
            try:
                type_ = _parse_values(type_.strip('{} '))
            except Exception:
                raise BadAttributeType()
            if isinstance(type_, dict):
                raise BadAttributeType()
        else:
            type_ = str(type_).upper()
            if type_ not in ['NUMERIC', 'REAL', 'INTEGER', 'STRING']:
                raise BadAttributeType()
        return (name, type_)

    def _decode(self, s, encode_nominal=False, matrix_type=DENSE):
        if False:
            print('Hello World!')
        'Do the job the ``encode``.'
        self._current_line = 0
        if isinstance(s, str):
            s = s.strip('\r\n ').replace('\r\n', '\n').split('\n')
        obj: ArffContainerType = {'description': '', 'relation': '', 'attributes': [], 'data': []}
        attribute_names = {}
        data = _get_data_object_for_decoding(matrix_type)
        STATE = _TK_DESCRIPTION
        s = iter(s)
        for row in s:
            self._current_line += 1
            row = row.strip(' \r\n')
            if not row:
                continue
            u_row = row.upper()
            if u_row.startswith(_TK_DESCRIPTION) and STATE == _TK_DESCRIPTION:
                obj['description'] += self._decode_comment(row) + '\n'
            elif u_row.startswith(_TK_RELATION):
                if STATE != _TK_DESCRIPTION:
                    raise BadLayout()
                STATE = _TK_RELATION
                obj['relation'] = self._decode_relation(row)
            elif u_row.startswith(_TK_ATTRIBUTE):
                if STATE != _TK_RELATION and STATE != _TK_ATTRIBUTE:
                    raise BadLayout()
                STATE = _TK_ATTRIBUTE
                attr = self._decode_attribute(row)
                if attr[0] in attribute_names:
                    raise BadAttributeName(attr[0], attribute_names[attr[0]])
                else:
                    attribute_names[attr[0]] = self._current_line
                obj['attributes'].append(attr)
                if isinstance(attr[1], (list, tuple)):
                    if encode_nominal:
                        conversor = EncodedNominalConversor(attr[1])
                    else:
                        conversor = NominalConversor(attr[1])
                else:
                    CONVERSOR_MAP = {'STRING': str, 'INTEGER': lambda x: int(float(x)), 'NUMERIC': float, 'REAL': float}
                    conversor = CONVERSOR_MAP[attr[1]]
                self._conversors.append(conversor)
            elif u_row.startswith(_TK_DATA):
                if STATE != _TK_ATTRIBUTE:
                    raise BadLayout()
                break
            elif u_row.startswith(_TK_COMMENT):
                pass
        else:
            raise BadLayout()

        def stream():
            if False:
                print('Hello World!')
            for row in s:
                self._current_line += 1
                row = row.strip()
                if row and (not row.startswith(_TK_COMMENT)):
                    yield row
        obj['data'] = data.decode_rows(stream(), self._conversors)
        if obj['description'].endswith('\n'):
            obj['description'] = obj['description'][:-1]
        return obj

    def decode(self, s, encode_nominal=False, return_type=DENSE):
        if False:
            i = 10
            return i + 15
        'Returns the Python representation of a given ARFF file.\n\n        When a file object is passed as an argument, this method reads lines\n        iteratively, avoiding to load unnecessary information to the memory.\n\n        :param s: a string or file object with the ARFF file.\n        :param encode_nominal: boolean, if True perform a label encoding\n            while reading the .arff file.\n        :param return_type: determines the data structure used to store the\n            dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,\n            `arff.DENSE_GEN` or `arff.LOD_GEN`.\n            Consult the sections on `working with sparse data`_ and `loading\n            progressively`_.\n        '
        try:
            return self._decode(s, encode_nominal=encode_nominal, matrix_type=return_type)
        except ArffException as e:
            e.line = self._current_line
            raise e

class ArffEncoder:
    """An ARFF encoder."""

    def _encode_comment(self, s=''):
        if False:
            for i in range(10):
                print('nop')
        '(INTERNAL) Encodes a comment line.\n\n        Comments are single line strings starting, obligatorily, with the ``%``\n        character, and can have any symbol, including whitespaces or special\n        characters.\n\n        If ``s`` is None, this method will simply return an empty comment.\n\n        :param s: (OPTIONAL) string.\n        :return: a string with the encoded comment line.\n        '
        if s:
            return '%s %s' % (_TK_COMMENT, s)
        else:
            return '%s' % _TK_COMMENT

    def _encode_relation(self, name):
        if False:
            for i in range(10):
                print('nop')
        '(INTERNAL) Decodes a relation line.\n\n        The relation declaration is a line with the format ``@RELATION\n        <relation-name>``, where ``relation-name`` is a string.\n\n        :param name: a string.\n        :return: a string with the encoded relation declaration.\n        '
        for char in ' %{},':
            if char in name:
                name = '"%s"' % name
                break
        return '%s %s' % (_TK_RELATION, name)

    def _encode_attribute(self, name, type_):
        if False:
            while True:
                i = 10
        '(INTERNAL) Encodes an attribute line.\n\n        The attribute follow the template::\n\n             @attribute <attribute-name> <datatype>\n\n        where ``attribute-name`` is a string, and ``datatype`` can be:\n\n        - Numerical attributes as ``NUMERIC``, ``INTEGER`` or ``REAL``.\n        - Strings as ``STRING``.\n        - Dates (NOT IMPLEMENTED).\n        - Nominal attributes with format:\n\n            {<nominal-name1>, <nominal-name2>, <nominal-name3>, ...}\n\n        This method must receive a the name of the attribute and its type, if\n        the attribute type is nominal, ``type`` must be a list of values.\n\n        :param name: a string.\n        :param type_: a string or a list of string.\n        :return: a string with the encoded attribute declaration.\n        '
        for char in ' %{},':
            if char in name:
                name = '"%s"' % name
                break
        if isinstance(type_, (tuple, list)):
            type_tmp = ['%s' % encode_string(type_k) for type_k in type_]
            type_ = '{%s}' % ', '.join(type_tmp)
        return '%s %s %s' % (_TK_ATTRIBUTE, name, type_)

    def encode(self, obj):
        if False:
            i = 10
            return i + 15
        'Encodes a given object to an ARFF file.\n\n        :param obj: the object containing the ARFF information.\n        :return: the ARFF file as an string.\n        '
        data = [row for row in self.iter_encode(obj)]
        return '\n'.join(data)

    def iter_encode(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'The iterative version of `arff.ArffEncoder.encode`.\n\n        This encodes iteratively a given object and return, one-by-one, the\n        lines of the ARFF file.\n\n        :param obj: the object containing the ARFF information.\n        :return: (yields) the ARFF file as strings.\n        '
        if obj.get('description', None):
            for row in obj['description'].split('\n'):
                yield self._encode_comment(row)
        if not obj.get('relation'):
            raise BadObject('Relation name not found or with invalid value.')
        yield self._encode_relation(obj['relation'])
        yield ''
        if not obj.get('attributes'):
            raise BadObject('Attributes not found.')
        attribute_names = set()
        for attr in obj['attributes']:
            if not isinstance(attr, (tuple, list)) or len(attr) != 2 or (not isinstance(attr[0], str)):
                raise BadObject('Invalid attribute declaration "%s"' % str(attr))
            if isinstance(attr[1], str):
                if attr[1] not in _SIMPLE_TYPES:
                    raise BadObject('Invalid attribute type "%s"' % str(attr))
            elif not isinstance(attr[1], (tuple, list)):
                raise BadObject('Invalid attribute type "%s"' % str(attr))
            if attr[0] in attribute_names:
                raise BadObject('Trying to use attribute name "%s" for the second time.' % str(attr[0]))
            else:
                attribute_names.add(attr[0])
            yield self._encode_attribute(attr[0], attr[1])
        yield ''
        attributes = obj['attributes']
        yield _TK_DATA
        if 'data' in obj:
            data = _get_data_object_for_encoding(obj.get('data'))
            yield from data.encode_data(obj.get('data'), attributes)
        yield ''

def load(fp, encode_nominal=False, return_type=DENSE):
    if False:
        i = 10
        return i + 15
    'Load a file-like object containing the ARFF document and convert it into\n    a Python object.\n\n    :param fp: a file-like object.\n    :param encode_nominal: boolean, if True perform a label encoding\n        while reading the .arff file.\n    :param return_type: determines the data structure used to store the\n        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,\n        `arff.DENSE_GEN` or `arff.LOD_GEN`.\n        Consult the sections on `working with sparse data`_ and `loading\n        progressively`_.\n    :return: a dictionary.\n     '
    decoder = ArffDecoder()
    return decoder.decode(fp, encode_nominal=encode_nominal, return_type=return_type)

def loads(s, encode_nominal=False, return_type=DENSE):
    if False:
        i = 10
        return i + 15
    'Convert a string instance containing the ARFF document into a Python\n    object.\n\n    :param s: a string object.\n    :param encode_nominal: boolean, if True perform a label encoding\n        while reading the .arff file.\n    :param return_type: determines the data structure used to store the\n        dataset. Can be one of `arff.DENSE`, `arff.COO`, `arff.LOD`,\n        `arff.DENSE_GEN` or `arff.LOD_GEN`.\n        Consult the sections on `working with sparse data`_ and `loading\n        progressively`_.\n    :return: a dictionary.\n    '
    decoder = ArffDecoder()
    return decoder.decode(s, encode_nominal=encode_nominal, return_type=return_type)

def dump(obj, fp):
    if False:
        return 10
    'Serialize an object representing the ARFF document to a given file-like\n    object.\n\n    :param obj: a dictionary.\n    :param fp: a file-like object.\n    '
    encoder = ArffEncoder()
    generator = encoder.iter_encode(obj)
    last_row = next(generator)
    for row in generator:
        fp.write(last_row + '\n')
        last_row = row
    fp.write(last_row)
    return fp

def dumps(obj):
    if False:
        for i in range(10):
            print('nop')
    'Serialize an object representing the ARFF document, returning a string.\n\n    :param obj: a dictionary.\n    :return: a string with the ARFF document.\n    '
    encoder = ArffEncoder()
    return encoder.encode(obj)