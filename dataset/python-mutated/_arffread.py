import re
import datetime
import numpy as np
import csv
import ctypes
'A module to read arff files.'
__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']
r_meta = re.compile('^\\s*@')
r_comment = re.compile('^%')
r_empty = re.compile('^\\s+$')
r_headerline = re.compile('^\\s*@\\S*')
r_datameta = re.compile('^@[Dd][Aa][Tt][Aa]')
r_relation = re.compile('^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\\s*(\\S*)')
r_attribute = re.compile('^\\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\\s*(..*$)')
r_nominal = re.compile('{(.+)}')
r_date = re.compile('[Dd][Aa][Tt][Ee]\\s+[\\"\']?(.+?)[\\"\']?$')
r_comattrval = re.compile("'(..+)'\\s+(..+$)")
r_wcomattrval = re.compile('(\\S+)\\s+(..+$)')

class ArffError(OSError):
    pass

class ParseArffError(ArffError):
    pass

class Attribute:
    type_name = None

    def __init__(self, name):
        if False:
            return 10
        self.name = name
        self.range = None
        self.dtype = np.object_

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            i = 10
            return i + 15
        '\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n        '
        return None

    def parse_data(self, data_str):
        if False:
            while True:
                i = 10
        '\n        Parse a value of this type.\n        '
        return None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse a value of this type.\n        '
        return self.name + ',' + self.type_name

class NominalAttribute(Attribute):
    type_name = 'nominal'

    def __init__(self, name, values):
        if False:
            return 10
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.bytes_, max((len(i) for i in values)))

    @staticmethod
    def _get_nom_val(atrv):
        if False:
            while True:
                i = 10
        'Given a string containing a nominal type, returns a tuple of the\n        possible values.\n\n        A nominal type is defined as something framed between braces ({}).\n\n        Parameters\n        ----------\n        atrv : str\n           Nominal type definition\n\n        Returns\n        -------\n        poss_vals : tuple\n           possible values\n\n        Examples\n        --------\n        >>> from scipy.io.arff._arffread import NominalAttribute\n        >>> NominalAttribute._get_nom_val("{floup, bouga, fl, ratata}")\n        (\'floup\', \'bouga\', \'fl\', \'ratata\')\n        '
        m = r_nominal.match(atrv)
        if m:
            (attrs, _) = split_data_line(m.group(1))
            return tuple(attrs)
        else:
            raise ValueError('This does not look like a nominal string')

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            while True:
                i = 10
        "\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n\n        For nominal attributes, the attribute string would be like '{<attr_1>,\n         <attr2>, <attr_3>}'.\n        "
        if attr_string[0] == '{':
            values = cls._get_nom_val(attr_string)
            return cls(name, values)
        else:
            return None

    def parse_data(self, data_str):
        if False:
            while True:
                i = 10
        '\n        Parse a value of this type.\n        '
        if data_str in self.values:
            return data_str
        elif data_str == '?':
            return data_str
        else:
            raise ValueError('{} value not in {}'.format(str(data_str), str(self.values)))

    def __str__(self):
        if False:
            return 10
        msg = self.name + ',{'
        for i in range(len(self.values) - 1):
            msg += self.values[i] + ','
        msg += self.values[-1]
        msg += '}'
        return msg

class NumericAttribute(Attribute):

    def __init__(self, name):
        if False:
            print('Hello World!')
        super().__init__(name)
        self.type_name = 'numeric'
        self.dtype = np.float64

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n\n        For numeric attributes, the attribute string would be like\n        'numeric' or 'int' or 'real'.\n        "
        attr_string = attr_string.lower().strip()
        if attr_string[:len('numeric')] == 'numeric' or attr_string[:len('int')] == 'int' or attr_string[:len('real')] == 'real':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        if False:
            return 10
        "\n        Parse a value of this type.\n\n        Parameters\n        ----------\n        data_str : str\n           string to convert\n\n        Returns\n        -------\n        f : float\n           where float can be nan\n\n        Examples\n        --------\n        >>> from scipy.io.arff._arffread import NumericAttribute\n        >>> atr = NumericAttribute('atr')\n        >>> atr.parse_data('1')\n        1.0\n        >>> atr.parse_data('1\\n')\n        1.0\n        >>> atr.parse_data('?\\n')\n        nan\n        "
        if '?' in data_str:
            return np.nan
        else:
            return float(data_str)

    def _basic_stats(self, data):
        if False:
            while True:
                i = 10
        nbfac = data.size * 1.0 / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac)

class StringAttribute(Attribute):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self.type_name = 'string'

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            print('Hello World!')
        "\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n\n        For string attributes, the attribute string would be like\n        'string'.\n        "
        attr_string = attr_string.lower().strip()
        if attr_string[:len('string')] == 'string':
            return cls(name)
        else:
            return None

class DateAttribute(Attribute):

    def __init__(self, name, date_format, datetime_unit):
        if False:
            print('Hello World!')
        super().__init__(name)
        self.date_format = date_format
        self.datetime_unit = datetime_unit
        self.type_name = 'date'
        self.range = date_format
        self.dtype = np.datetime64(0, self.datetime_unit)

    @staticmethod
    def _get_date_format(atrv):
        if False:
            i = 10
            return i + 15
        m = r_date.match(atrv)
        if m:
            pattern = m.group(1).strip()
            datetime_unit = None
            if 'yyyy' in pattern:
                pattern = pattern.replace('yyyy', '%Y')
                datetime_unit = 'Y'
            elif 'yy':
                pattern = pattern.replace('yy', '%y')
                datetime_unit = 'Y'
            if 'MM' in pattern:
                pattern = pattern.replace('MM', '%m')
                datetime_unit = 'M'
            if 'dd' in pattern:
                pattern = pattern.replace('dd', '%d')
                datetime_unit = 'D'
            if 'HH' in pattern:
                pattern = pattern.replace('HH', '%H')
                datetime_unit = 'h'
            if 'mm' in pattern:
                pattern = pattern.replace('mm', '%M')
                datetime_unit = 'm'
            if 'ss' in pattern:
                pattern = pattern.replace('ss', '%S')
                datetime_unit = 's'
            if 'z' in pattern or 'Z' in pattern:
                raise ValueError('Date type attributes with time zone not supported, yet')
            if datetime_unit is None:
                raise ValueError('Invalid or unsupported date format')
            return (pattern, datetime_unit)
        else:
            raise ValueError('Invalid or no date format')

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            return 10
        "\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n\n        For date attributes, the attribute string would be like\n        'date <format>'.\n        "
        attr_string_lower = attr_string.lower().strip()
        if attr_string_lower[:len('date')] == 'date':
            (date_format, datetime_unit) = cls._get_date_format(attr_string)
            return cls(name, date_format, datetime_unit)
        else:
            return None

    def parse_data(self, data_str):
        if False:
            while True:
                i = 10
        '\n        Parse a value of this type.\n        '
        date_str = data_str.strip().strip("'").strip('"')
        if date_str == '?':
            return np.datetime64('NaT', self.datetime_unit)
        else:
            dt = datetime.datetime.strptime(date_str, self.date_format)
            return np.datetime64(dt).astype('datetime64[%s]' % self.datetime_unit)

    def __str__(self):
        if False:
            return 10
        return super().__str__() + ',' + self.date_format

class RelationalAttribute(Attribute):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name)
        self.type_name = 'relational'
        self.dtype = np.object_
        self.attributes = []
        self.dialect = None

    @classmethod
    def parse_attribute(cls, name, attr_string):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parse the attribute line if it knows how. Returns the parsed\n        attribute, or None.\n\n        For date attributes, the attribute string would be like\n        'date <format>'.\n        "
        attr_string_lower = attr_string.lower().strip()
        if attr_string_lower[:len('relational')] == 'relational':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        if False:
            i = 10
            return i + 15
        elems = list(range(len(self.attributes)))
        escaped_string = data_str.encode().decode('unicode-escape')
        row_tuples = []
        for raw in escaped_string.split('\n'):
            (row, self.dialect) = split_data_line(raw, self.dialect)
            row_tuples.append(tuple([self.attributes[i].parse_data(row[i]) for i in elems]))
        return np.array(row_tuples, [(a.name, a.dtype) for a in self.attributes])

    def __str__(self):
        if False:
            print('Hello World!')
        return super().__str__() + '\n\t' + '\n\t'.join((str(a) for a in self.attributes))

def to_attribute(name, attr_string):
    if False:
        return 10
    attr_classes = (NominalAttribute, NumericAttribute, DateAttribute, StringAttribute, RelationalAttribute)
    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)
        if attr is not None:
            return attr
    raise ParseArffError('unknown attribute %s' % attr_string)

def csv_sniffer_has_bug_last_field():
    if False:
        return 10
    '\n    Checks if the bug https://bugs.python.org/issue30157 is unpatched.\n    '
    has_bug = getattr(csv_sniffer_has_bug_last_field, 'has_bug', None)
    if has_bug is None:
        dialect = csv.Sniffer().sniff("3, 'a'")
        csv_sniffer_has_bug_last_field.has_bug = dialect.quotechar != "'"
        has_bug = csv_sniffer_has_bug_last_field.has_bug
    return has_bug

def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
    if False:
        return 10
    '\n    Workaround for the bug https://bugs.python.org/issue30157 if is unpatched.\n    '
    if csv_sniffer_has_bug_last_field():
        right_regex = '(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)'
        for restr in ('(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)(?P<quote>["\\\']).*?(?P=quote)(?P=delim)', '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?P<delim>[^\\w\\n"\\\'])(?P<space> ?)', right_regex, '(?:^|\\n)(?P<quote>["\\\']).*?(?P=quote)(?:$|\\n)'):
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            matches = regexp.findall(sniff_line)
            if matches:
                break
        if restr != right_regex:
            return
        groupindex = regexp.groupindex
        assert len(matches) == 1
        m = matches[0]
        n = groupindex['quote'] - 1
        quote = m[n]
        n = groupindex['delim'] - 1
        delim = m[n]
        n = groupindex['space'] - 1
        space = bool(m[n])
        dq_regexp = re.compile('((%(delim)s)|^)\\W*%(quote)s[^%(delim)s\\n]*%(quote)s[^%(delim)s\\n]*%(quote)s\\W*((%(delim)s)|$)' % {'delim': re.escape(delim), 'quote': quote}, re.MULTILINE)
        doublequote = bool(dq_regexp.search(sniff_line))
        dialect.quotechar = quote
        if delim in delimiters:
            dialect.delimiter = delim
        dialect.doublequote = doublequote
        dialect.skipinitialspace = space

def split_data_line(line, dialect=None):
    if False:
        while True:
            i = 10
    delimiters = ',\t'
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
    if line[-1] == '\n':
        line = line[:-1]
    line = line.strip()
    sniff_line = line
    if not any((d in line for d in delimiters)):
        sniff_line += ','
    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
        workaround_csv_sniffer_bug_last_field(sniff_line=sniff_line, dialect=dialect, delimiters=delimiters)
    row = next(csv.reader([line], dialect))
    return (row, dialect)

def tokenize_attribute(iterable, attribute):
    if False:
        for i in range(10):
            print('nop')
    'Parse a raw string in header (e.g., starts by @attribute).\n\n    Given a raw string attribute, try to get the name and type of the\n    attribute. Constraints:\n\n    * The first line must start with @attribute (case insensitive, and\n      space like characters before @attribute are allowed)\n    * Works also if the attribute is spread on multilines.\n    * Works if empty lines or comments are in between\n\n    Parameters\n    ----------\n    attribute : str\n       the attribute string.\n\n    Returns\n    -------\n    name : str\n       name of the attribute\n    value : str\n       value of the attribute\n    next : str\n       next line to be parsed\n\n    Examples\n    --------\n    If attribute is a string defined in python as r"floupi real", will\n    return floupi as name, and real as value.\n\n    >>> from scipy.io.arff._arffread import tokenize_attribute\n    >>> iterable = iter([0] * 10) # dummy iterator\n    >>> tokenize_attribute(iterable, r"@attribute floupi real")\n    (\'floupi\', \'real\', 0)\n\n    If attribute is r"\'floupi 2\' real", will return \'floupi 2\' as name,\n    and real as value.\n\n    >>> tokenize_attribute(iterable, r"  @attribute \'floupi 2\' real   ")\n    (\'floupi 2\', \'real\', 0)\n\n    '
    sattr = attribute.strip()
    mattr = r_attribute.match(sattr)
    if mattr:
        atrv = mattr.group(1)
        if r_comattrval.match(atrv):
            (name, type) = tokenize_single_comma(atrv)
            next_item = next(iterable)
        elif r_wcomattrval.match(atrv):
            (name, type) = tokenize_single_wcomma(atrv)
            next_item = next(iterable)
        else:
            raise ValueError('multi line not supported yet')
    else:
        raise ValueError('First line unparsable: %s' % sattr)
    attribute = to_attribute(name, type)
    if type.lower() == 'relational':
        next_item = read_relational_attribute(iterable, attribute, next_item)
    return (attribute, next_item)

def tokenize_single_comma(val):
    if False:
        print('Hello World!')
    m = r_comattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError('Error while tokenizing attribute') from e
    else:
        raise ValueError('Error while tokenizing single %s' % val)
    return (name, type)

def tokenize_single_wcomma(val):
    if False:
        i = 10
        return i + 15
    m = r_wcomattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError('Error while tokenizing attribute') from e
    else:
        raise ValueError('Error while tokenizing single %s' % val)
    return (name, type)

def read_relational_attribute(ofile, relational_attribute, i):
    if False:
        for i in range(10):
            print('nop')
    'Read the nested attributes of a relational attribute'
    r_end_relational = re.compile('^@[Ee][Nn][Dd]\\s*' + relational_attribute.name + '\\s*$')
    while not r_end_relational.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                (attr, i) = tokenize_attribute(ofile, i)
                relational_attribute.attributes.append(attr)
            else:
                raise ValueError('Error parsing line %s' % i)
        else:
            i = next(ofile)
    i = next(ofile)
    return i

def read_header(ofile):
    if False:
        return 10
    'Read the header of the iterable ofile.'
    i = next(ofile)
    while r_comment.match(i):
        i = next(ofile)
    relation = None
    attributes = []
    while not r_datameta.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                (attr, i) = tokenize_attribute(ofile, i)
                attributes.append(attr)
            else:
                isrel = r_relation.match(i)
                if isrel:
                    relation = isrel.group(1)
                else:
                    raise ValueError('Error parsing line %s' % i)
                i = next(ofile)
        else:
            i = next(ofile)
    return (relation, attributes)

class MetaData:
    """Small container to keep useful information on a ARFF dataset.

    Knows about attributes names and types.

    Examples
    --------
    ::

        data, meta = loadarff('iris.arff')
        # This will print the attributes names of the iris.arff dataset
        for i in meta:
            print(i)
        # This works too
        meta.names()
        # Getting attribute type
        types = meta.types()

    Methods
    -------
    names
    types

    Notes
    -----
    Also maintains the list of attributes in order, i.e., doing for i in
    meta, where meta is an instance of MetaData, will return the
    different attribute names in the order they were defined.
    """

    def __init__(self, rel, attr):
        if False:
            for i in range(10):
                print('nop')
        self.name = rel
        self._attributes = {a.name: a for a in attr}

    def __repr__(self):
        if False:
            return 10
        msg = ''
        msg += 'Dataset: %s\n' % self.name
        for i in self._attributes:
            msg += f"\t{i}'s type is {self._attributes[i].type_name}"
            if self._attributes[i].range:
                msg += ', range is %s' % str(self._attributes[i].range)
            msg += '\n'
        return msg

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self._attributes)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        attr = self._attributes[key]
        return (attr.type_name, attr.range)

    def names(self):
        if False:
            print('Hello World!')
        'Return the list of attribute names.\n\n        Returns\n        -------\n        attrnames : list of str\n            The attribute names.\n        '
        return list(self._attributes)

    def types(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the list of attribute types.\n\n        Returns\n        -------\n        attr_types : list of str\n            The attribute types.\n        '
        attr_types = [self._attributes[name].type_name for name in self._attributes]
        return attr_types

def loadarff(f):
    if False:
        while True:
            i = 10
    '\n    Read an arff file.\n\n    The data is returned as a record array, which can be accessed much like\n    a dictionary of NumPy arrays. For example, if one of the attributes is\n    called \'pressure\', then its first 10 data points can be accessed from the\n    ``data`` record array like so: ``data[\'pressure\'][0:10]``\n\n\n    Parameters\n    ----------\n    f : file-like or str\n       File-like object to read from, or filename to open.\n\n    Returns\n    -------\n    data : record array\n       The data of the arff file, accessible by attribute names.\n    meta : `MetaData`\n       Contains information about the arff file such as name and\n       type of attributes, the relation (name of the dataset), etc.\n\n    Raises\n    ------\n    ParseArffError\n        This is raised if the given file is not ARFF-formatted.\n    NotImplementedError\n        The ARFF file has an attribute which is not supported yet.\n\n    Notes\n    -----\n\n    This function should be able to read most arff files. Not\n    implemented functionality include:\n\n    * date type attributes\n    * string type attributes\n\n    It can read files with numeric and nominal attributes. It cannot read\n    files with sparse data ({} in the file). However, this function can\n    read files with missing data (? in the file), representing the data\n    points as NaNs.\n\n    Examples\n    --------\n    >>> from scipy.io import arff\n    >>> from io import StringIO\n    >>> content = """\n    ... @relation foo\n    ... @attribute width  numeric\n    ... @attribute height numeric\n    ... @attribute color  {red,green,blue,yellow,black}\n    ... @data\n    ... 5.0,3.25,blue\n    ... 4.5,3.75,green\n    ... 3.0,4.00,red\n    ... """\n    >>> f = StringIO(content)\n    >>> data, meta = arff.loadarff(f)\n    >>> data\n    array([(5.0, 3.25, \'blue\'), (4.5, 3.75, \'green\'), (3.0, 4.0, \'red\')],\n          dtype=[(\'width\', \'<f8\'), (\'height\', \'<f8\'), (\'color\', \'|S6\')])\n    >>> meta\n    Dataset: foo\n    \twidth\'s type is numeric\n    \theight\'s type is numeric\n    \tcolor\'s type is nominal, range is (\'red\', \'green\', \'blue\', \'yellow\', \'black\')\n\n    '
    if hasattr(f, 'read'):
        ofile = f
    else:
        ofile = open(f)
    try:
        return _loadarff(ofile)
    finally:
        if ofile is not f:
            ofile.close()

def _loadarff(ofile):
    if False:
        i = 10
        return i + 15
    try:
        (rel, attr) = read_header(ofile)
    except ValueError as e:
        msg = 'Error while parsing header, error was: ' + str(e)
        raise ParseArffError(msg) from e
    hasstr = False
    for a in attr:
        if isinstance(a, StringAttribute):
            hasstr = True
    meta = MetaData(rel, attr)
    if hasstr:
        raise NotImplementedError('String attributes not supported yet, sorry')
    ni = len(attr)

    def generator(row_iter, delim=','):
        if False:
            for i in range(10):
                print('nop')
        elems = list(range(ni))
        dialect = None
        for raw in row_iter:
            if r_comment.match(raw) or r_empty.match(raw):
                continue
            (row, dialect) = split_data_line(raw, dialect)
            yield tuple([attr[i].parse_data(row[i]) for i in elems])
    a = list(generator(ofile))
    data = np.array(a, [(a.name, a.dtype) for a in attr])
    return (data, meta)

def basic_stats(data):
    if False:
        return 10
    nbfac = data.size * 1.0 / (data.size - 1)
    return (np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac)

def print_attribute(name, tp, data):
    if False:
        return 10
    type = tp.type_name
    if type == 'numeric' or type == 'real' or type == 'integer':
        (min, max, mean, std) = basic_stats(data)
        print(f'{name},{type},{min:f},{max:f},{mean:f},{std:f}')
    else:
        print(str(tp))

def test_weka(filename):
    if False:
        for i in range(10):
            print('nop')
    (data, meta) = loadarff(filename)
    print(len(data.dtype))
    print(data.size)
    for i in meta:
        print_attribute(i, meta[i], data[i])
test_weka.__test__ = False
if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    test_weka(filename)