import uuid
from os.path import relpath
from coala_utils.decorators import enforce_signature, generate_ordering, generate_repr, get_public_members
from coalib.bearlib.aspects import aspectbase
from coalib.results.RESULT_SEVERITY import RESULT_SEVERITY
from coalib.results.SourceRange import SourceRange

@generate_repr(('id', hex), 'origin', 'affected_code', ('severity', RESULT_SEVERITY.reverse.get), 'confidence', 'message', ('aspect', lambda aspect: type(aspect).__qualname__), 'applied_actions')
@generate_ordering('affected_code', 'severity', 'confidence', 'origin', 'message_base', 'message_arguments', 'aspect', 'additional_info', 'diffs', 'debug_msg', 'applied_actions')
class Result:
    """
    A result is anything that has an origin and a message.

    Optionally it might affect a file.

    Result messages can also have arguments. The message is python
    style formatted with these arguments.

    >>> r = Result('origin','{arg1} and {arg2}',         message_arguments={'arg1': 'foo', 'arg2': 'bar'})
    >>> r.message
    'foo and bar'

    Message arguments may be changed later. The result message
    will also reflect these changes.

    >>> r.message_arguments = {'arg1': 'spam', 'arg2': 'eggs'}
    >>> r.message
    'spam and eggs'

    """

    @enforce_signature
    def __init__(self, origin, message: str, affected_code: (tuple, list)=(), severity: int=RESULT_SEVERITY.NORMAL, additional_info: str='', debug_msg='', diffs: (dict, None)=None, confidence: int=100, aspect: (aspectbase, None)=None, message_arguments: dict={}, applied_actions: dict={}, actions: list=[], alternate_diffs: (list, None)=None):
        if False:
            return 10
        '\n        :param origin:\n            Class name or creator object of this object.\n        :param message:\n            Base message to show with this result.\n        :param affected_code:\n            A tuple of ``SourceRange`` objects pointing to related positions\n            in the source code.\n        :param severity:\n            Severity of this result.\n        :param additional_info:\n            A long description holding additional information about the issue\n            and/or how to fix it. You can use this like a manual entry for a\n            category of issues.\n        :param debug_msg:\n            A message which may help the user find out why this result was\n            yielded.\n        :param diffs:\n            A dictionary with filename as key and ``Diff`` object\n            associated with it as value.\n        :param confidence:\n            A number between 0 and 100 describing the likelihood of this result\n            being a real issue.\n        :param aspect:\n            An aspectclass instance which this result is associated to.\n            Note that this should be a leaf of the aspect tree!\n            (If you have a node, spend some time figuring out which of\n            the leafs exactly your result belongs to.)\n        :param message_arguments:\n            Arguments to be provided to the base message.\n        :param applied_actions:\n            A dictionary that contains the result, file_dict, file_diff_dict and\n            the section for an action.\n        :param actions:\n            A list of action instances specific to the origin of the result.\n        :param alternate_diffs:\n            A list of dictionaries, where each element is an alternative diff.\n        :raises ValueError:\n            Raised when confidence is not between 0 and 100.\n        :raises KeyError:\n            Raised when message_base can not be formatted with\n            message_arguments.\n        '
        origin = origin or ''
        if not isinstance(origin, str):
            origin = origin.__class__.__name__
        if severity not in RESULT_SEVERITY.reverse:
            raise ValueError('severity is not a valid RESULT_SEVERITY')
        self.origin = origin
        self.message_base = message
        self.message_arguments = message_arguments
        self.applied_actions = applied_actions
        if message_arguments:
            self.message_base.format(**self.message_arguments)
        self.debug_msg = debug_msg
        self.additional_info = additional_info
        self.affected_code = tuple(sorted(affected_code))
        self.severity = severity
        if confidence < 0 or confidence > 100:
            raise ValueError('Value of confidence should be between 0 and 100.')
        self.confidence = confidence
        self.diffs = diffs
        self.id = uuid.uuid4().int
        self.aspect = aspect
        if self.aspect and (not self.additional_info):
            self.additional_info = f'{aspect.Docs.importance_reason} {aspect.Docs.fix_suggestions}'
        self.actions = actions
        self.alternate_diffs = alternate_diffs

    @property
    def message(self):
        if False:
            i = 10
            return i + 15
        if not self.message_arguments:
            return self.message_base
        return self.message_base.format(**self.message_arguments)

    @message.setter
    def message(self, value: str):
        if False:
            i = 10
            return i + 15
        self.message_base = value

    def set_applied_actions(self, applied_actions):
        if False:
            while True:
                i = 10
        self.applied_actions = applied_actions

    def get_applied_actions(self):
        if False:
            i = 10
            return i + 15
        return self.applied_actions

    @classmethod
    @enforce_signature
    def from_values(cls, origin, message: str, file: str, line: (int, None)=None, column: (int, None)=None, end_line: (int, None)=None, end_column: (int, None)=None, severity: int=RESULT_SEVERITY.NORMAL, additional_info: str='', debug_msg='', diffs: (dict, None)=None, confidence: int=100, aspect: (aspectbase, None)=None, message_arguments: dict={}, actions: list=[], alternate_diffs: (list, None)=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a result with only one SourceRange with the given start and end\n        locations.\n\n        :param origin:\n            Class name or creator object of this object.\n        :param message:\n            Base message to show with this result.\n        :param message_arguments:\n            Arguments to be provided to the base message\n        :param file:\n            The related file.\n        :param line:\n            The first related line in the file. (First line is 1)\n            line = None means an unknown line occurs anywhere in the file.\n        :param column:\n            The column indicating the first character. (First character is 1)\n            column = None means there is an error with the whole line.\n            All combinations of None values are allowed for line and column,\n            Except line = None and column = <some number>\n        :param end_line:\n            The last related line in the file.\n        :param end_column:\n            The column indicating the last character.\n        :param severity:\n            Severity of this result.\n        :param additional_info:\n            A long description holding additional information about the issue\n            and/or how to fix it. You can use this like a manual entry for a\n            category of issues.\n        :param debug_msg:\n            A message which may help the user find out why this result was\n            yielded.\n        :param diffs:\n            A dictionary with filename as key and ``Diff`` object\n            associated with it as value.\n        :param confidence:\n            A number between 0 and 100 describing the likelihood of this result\n            being a real issue.\n        :param aspect:\n            An aspect object which this result is associated to. Note that this\n            should be a leaf of the aspect tree! (If you have a node, spend\n            some time figuring out which of the leafs exactly your result\n            belongs to.)\n        :param actions:\n            A list of action instances specific to the origin of the result.\n        :param alternate_diffs:\n            A list of dictionaries, where each element is an alternative diff.\n        '
        source_range = SourceRange.from_values(file, line, column, end_line, end_column)
        return cls(origin=origin, message=message, affected_code=(source_range,), severity=severity, additional_info=additional_info, debug_msg=debug_msg, diffs=diffs, confidence=confidence, aspect=aspect, message_arguments=message_arguments, actions=actions, alternate_diffs=alternate_diffs)

    def to_string_dict(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Makes a dictionary which has all keys and values as strings and\n        contains all the data that the base Result has.\n\n        FIXME: diffs are not serialized ATM.\n        FIXME: Only the first SourceRange of affected_code is serialized. If\n        there are more, this data is currently missing.\n\n        :return: Dictionary with keys and values as string.\n        '
        retval = {}
        members = ['id', 'additional_info', 'debug_msg', 'message', 'message_base', 'message_arguments', 'origin', 'confidence']
        for member in members:
            value = getattr(self, member)
            retval[member] = '' if value is None else str(value)
        retval['severity'] = str(RESULT_SEVERITY.reverse.get(self.severity, ''))
        if len(self.affected_code) > 0:
            retval['file'] = self.affected_code[0].file
            line = self.affected_code[0].start.line
            retval['line_nr'] = '' if line is None else str(line)
        else:
            (retval['file'], retval['line_nr']) = ('', '')
        return retval

    @enforce_signature
    def apply(self, file_dict: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies all contained diffs to the given file_dict. This operation will\n        be done in-place.\n\n        :param file_dict: A dictionary containing all files with filename as\n                          key and all lines a value. Will be modified.\n        '
        for (filename, diff) in self.diffs.items():
            file_dict[filename] = diff.modified

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Joins those patches to one patch.\n\n        :param other: The other patch.\n        '
        assert isinstance(self.diffs, dict)
        assert isinstance(other.diffs, dict)
        for filename in other.diffs:
            if filename in self.diffs:
                self.diffs[filename] += other.diffs[filename]
            else:
                self.diffs[filename] = other.diffs[filename]
        return self

    def overlaps(self, ranges):
        if False:
            return 10
        '\n        Determines if the result overlaps with source ranges provided.\n\n        :param ranges: A list SourceRange objects to check for overlap.\n        :return:       True if the ranges overlap with the result.\n        '
        if isinstance(ranges, SourceRange):
            ranges = [ranges]
        for range in ranges:
            for self_range in self.affected_code:
                if range.overlaps(self_range):
                    return True
        return False

    def location_repr(self):
        if False:
            return 10
        '\n        Retrieves a string, that briefly represents\n        the affected code of the result.\n\n        :return: A string containing all of the affected files\n                 separated by a comma.\n        '
        if not self.affected_code:
            return 'the whole project'
        range_paths = set((sourcerange.file for sourcerange in self.affected_code))
        return ', '.join((repr(relpath(range_path)) for range_path in sorted(range_paths)))

    def __json__(self, use_relpath=False):
        if False:
            return 10
        _dict = get_public_members(self)
        if use_relpath and _dict['diffs']:
            _dict['diffs'] = {relpath(file): diff for (file, diff) in _dict['diffs'].items()}
        _dict['aspect'] = type(self.aspect).__qualname__
        return _dict