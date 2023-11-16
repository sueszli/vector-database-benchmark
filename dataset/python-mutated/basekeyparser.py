"""Base class for vim-like key sequence parser."""
import string
import types
import dataclasses
import traceback
from typing import Mapping, MutableMapping, Optional, Sequence
from qutebrowser.qt.core import QObject, pyqtSignal
from qutebrowser.qt.gui import QKeySequence, QKeyEvent
from qutebrowser.config import config
from qutebrowser.utils import log, usertypes, utils, message
from qutebrowser.keyinput import keyutils

@dataclasses.dataclass(frozen=True)
class MatchResult:
    """The result of matching a keybinding."""
    match_type: QKeySequence.SequenceMatch
    command: Optional[str]
    sequence: keyutils.KeySequence

    def __post_init__(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.match_type == QKeySequence.SequenceMatch.ExactMatch:
            assert self.command is not None
        else:
            assert self.command is None

class BindingTrie:
    """Helper class for key parser. Represents a set of bindings.

    Every BindingTree will either contain children or a command (for leaf
    nodes). The only exception is the root BindingNode, if there are no
    bindings at all.

    From the outside, this class works similar to a mapping of
    keyutils.KeySequence to str. Doing trie[sequence] = 'command' adds a
    binding, and so does calling .update() with a mapping. Additionally, a
    "matches" method can be used to do partial matching.

    However, some mapping methods are not (yet) implemented:
    - __getitem__ (use matches() instead)
    - __len__
    - __iter__
    - __delitem__

    Attributes:
        children: A mapping from KeyInfo to children BindingTries.
        command: Command associated with this trie node.
    """
    __slots__ = ('children', 'command')

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.children: MutableMapping[keyutils.KeyInfo, BindingTrie] = {}
        self.command: Optional[str] = None

    def __setitem__(self, sequence: keyutils.KeySequence, command: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        node = self
        for key in sequence:
            if key not in node.children:
                node.children[key] = BindingTrie()
            node = node.children[key]
        node.command = command

    def __contains__(self, sequence: keyutils.KeySequence) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.matches(sequence).match_type == QKeySequence.SequenceMatch.ExactMatch

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return utils.get_repr(self, children=self.children, command=self.command)

    def __str__(self) -> str:
        if False:
            return 10
        return '\n'.join(self.string_lines(blank=True))

    def string_lines(self, indent: int=0, blank: bool=False) -> Sequence[str]:
        if False:
            while True:
                i = 10
        'Get a list of strings for a pretty-printed version of this trie.'
        lines = []
        if self.command is not None:
            lines.append('{}=> {}'.format('  ' * indent, self.command))
        for (key, child) in sorted(self.children.items()):
            lines.append('{}{}:'.format('  ' * indent, key))
            lines.extend(child.string_lines(indent=indent + 1))
            if blank:
                lines.append('')
        return lines

    def update(self, mapping: Mapping[keyutils.KeySequence, str]) -> None:
        if False:
            while True:
                i = 10
        'Add data from the given mapping to the trie.'
        for key in mapping:
            self[key] = mapping[key]

    def matches(self, sequence: keyutils.KeySequence) -> MatchResult:
        if False:
            for i in range(10):
                print('nop')
        'Try to match a given keystring with any bound keychain.\n\n        Args:\n            sequence: The key sequence to match.\n\n        Return:\n            A MatchResult object.\n        '
        node = self
        for key in sequence:
            try:
                node = node.children[key]
            except KeyError:
                return MatchResult(match_type=QKeySequence.SequenceMatch.NoMatch, command=None, sequence=sequence)
        if node.command is not None:
            return MatchResult(match_type=QKeySequence.SequenceMatch.ExactMatch, command=node.command, sequence=sequence)
        elif node.children:
            return MatchResult(match_type=QKeySequence.SequenceMatch.PartialMatch, command=None, sequence=sequence)
        else:
            return MatchResult(match_type=QKeySequence.SequenceMatch.NoMatch, command=None, sequence=sequence)

class BaseKeyParser(QObject):
    """Parser for vim-like key sequences and shortcuts.

    Not intended to be instantiated directly. Subclasses have to override
    execute() to do whatever they want to.

    Attributes:
        mode_name: The name of the mode in the config.
        bindings: Bound key bindings
        _mode: The usertypes.KeyMode associated with this keyparser.
        _win_id: The window ID this keyparser is associated with.
        _sequence: The currently entered key sequence
        _do_log: Whether to log keypresses or not.
        passthrough: Whether unbound keys should be passed through with this
                     handler.
        _supports_count: Whether count is supported.

    Signals:
        keystring_updated: Emitted when the keystring is updated.
                           arg: New keystring.
        request_leave: Emitted to request leaving a mode.
                       arg 0: Mode to leave.
                       arg 1: Reason for leaving.
                       arg 2: Ignore the request if we're not in that mode
    """
    keystring_updated = pyqtSignal(str)
    request_leave = pyqtSignal(usertypes.KeyMode, str, bool)

    def __init__(self, *, mode: usertypes.KeyMode, win_id: int, parent: QObject=None, do_log: bool=True, passthrough: bool=False, supports_count: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._win_id = win_id
        self._sequence = keyutils.KeySequence()
        self._count = ''
        self._mode = mode
        self._do_log = do_log
        self.passthrough = passthrough
        self._supports_count = supports_count
        self.bindings = BindingTrie()
        self._read_config()
        config.instance.changed.connect(self._on_config_changed)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return utils.get_repr(self, mode=self._mode, win_id=self._win_id, do_log=self._do_log, passthrough=self.passthrough, supports_count=self._supports_count)

    def _debug_log(self, msg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Log a message to the debug log if logging is active.\n\n        Args:\n            message: The message to log.\n        '
        if self._do_log:
            prefix = '{} for mode {}: '.format(self.__class__.__name__, self._mode.name)
            log.keyboard.debug(prefix + msg)

    def _match_key(self, sequence: keyutils.KeySequence) -> MatchResult:
        if False:
            return 10
        'Try to match a given keystring with any bound keychain.\n\n        Args:\n            sequence: The command string to find.\n\n        Return:\n            A tuple (matchtype, binding).\n                matchtype: Match.definitive, Match.partial or Match.none.\n                binding: - None with Match.partial/Match.none.\n                         - The found binding with Match.definitive.\n        '
        assert sequence
        return self.bindings.matches(sequence)

    def _match_without_modifiers(self, sequence: keyutils.KeySequence) -> MatchResult:
        if False:
            print('Hello World!')
        'Try to match a key with optional modifiers stripped.'
        self._debug_log('Trying match without modifiers')
        sequence = sequence.strip_modifiers()
        return self._match_key(sequence)

    def _match_key_mapping(self, sequence: keyutils.KeySequence) -> MatchResult:
        if False:
            return 10
        'Try to match a key in bindings.key_mappings.'
        self._debug_log('Trying match with key_mappings')
        mapped = sequence.with_mappings(types.MappingProxyType(config.cache['bindings.key_mappings']))
        if sequence != mapped:
            self._debug_log('Mapped {} -> {}'.format(sequence, mapped))
            return self._match_key(mapped)
        return MatchResult(match_type=QKeySequence.SequenceMatch.NoMatch, command=None, sequence=sequence)

    def _match_count(self, sequence: keyutils.KeySequence, dry_run: bool) -> bool:
        if False:
            i = 10
            return i + 15
        'Try to match a key as count.'
        if not config.val.input.match_counts:
            return False
        txt = str(sequence[-1])
        if txt in string.digits and self._supports_count and (not (not self._count and txt == '0')):
            self._debug_log('Trying match as count')
            assert len(txt) == 1, txt
            if not dry_run:
                self._count += txt
                self.keystring_updated.emit(self._count + str(self._sequence))
            return True
        return False

    def handle(self, e: QKeyEvent, *, dry_run: bool=False) -> QKeySequence.SequenceMatch:
        if False:
            i = 10
            return i + 15
        "Handle a new keypress.\n\n        Separate the keypress into count/command, then check if it matches\n        any possible command, and either run the command, ignore it, or\n        display an error.\n\n        Args:\n            e: the KeyPressEvent from Qt.\n            dry_run: Don't actually execute anything, only check whether there\n                     would be a match.\n\n        Return:\n            A QKeySequence match.\n        "
        try:
            info = keyutils.KeyInfo.from_event(e)
        except keyutils.InvalidKeyError as ex:
            log.keyboard.debug(f'Got invalid key: {ex}')
            self.clear_keystring()
            return QKeySequence.SequenceMatch.NoMatch
        self._debug_log(f'Got key: {info!r} (dry_run {dry_run})')
        if info.is_modifier_key():
            self._debug_log('Ignoring, only modifier')
            return QKeySequence.SequenceMatch.NoMatch
        try:
            sequence = self._sequence.append_event(e)
        except keyutils.KeyParseError as ex:
            self._debug_log('{} Aborting keychain.'.format(ex))
            self.clear_keystring()
            return QKeySequence.SequenceMatch.NoMatch
        result = self._match_key(sequence)
        del sequence
        if result.match_type == QKeySequence.SequenceMatch.NoMatch:
            result = self._match_without_modifiers(result.sequence)
        if result.match_type == QKeySequence.SequenceMatch.NoMatch:
            result = self._match_key_mapping(result.sequence)
        if result.match_type == QKeySequence.SequenceMatch.NoMatch:
            was_count = self._match_count(result.sequence, dry_run)
            if was_count:
                return QKeySequence.SequenceMatch.ExactMatch
        if dry_run:
            return result.match_type
        self._sequence = result.sequence
        self._handle_result(info, result)
        return result.match_type

    def _handle_result(self, info: keyutils.KeyInfo, result: MatchResult) -> None:
        if False:
            i = 10
            return i + 15
        'Handle a final MatchResult from handle().'
        if result.match_type == QKeySequence.SequenceMatch.ExactMatch:
            assert result.command is not None
            self._debug_log("Definitive match for '{}'.".format(result.sequence))
            try:
                count = int(self._count) if self._count else None
            except ValueError as err:
                message.error(f'Failed to parse count: {err}', stack=traceback.format_exc())
                self.clear_keystring()
                return
            self.clear_keystring()
            self.execute(result.command, count)
        elif result.match_type == QKeySequence.SequenceMatch.PartialMatch:
            self._debug_log("No match for '{}' (added {})".format(result.sequence, info))
            self.keystring_updated.emit(self._count + str(result.sequence))
        elif result.match_type == QKeySequence.SequenceMatch.NoMatch:
            self._debug_log("Giving up with '{}', no matches".format(result.sequence))
            self.clear_keystring()
        else:
            raise utils.Unreachable('Invalid match value {!r}'.format(result.match_type))

    @config.change_filter('bindings')
    def _on_config_changed(self) -> None:
        if False:
            return 10
        self._read_config()

    def _read_config(self) -> None:
        if False:
            print('Hello World!')
        'Read the configuration.'
        self.bindings = BindingTrie()
        config_bindings = config.key_instance.get_bindings_for(self._mode.name)
        for (key, cmd) in config_bindings.items():
            assert cmd
            self.bindings[key] = cmd

    def execute(self, cmdstr: str, count: int=None) -> None:
        if False:
            i = 10
            return i + 15
        'Handle a completed keychain.\n\n        Args:\n            cmdstr: The command to execute as a string.\n            count: The count if given.\n        '
        raise NotImplementedError

    def clear_keystring(self) -> None:
        if False:
            return 10
        'Clear the currently entered key sequence.'
        if self._sequence:
            self._debug_log('Clearing keystring (was: {}).'.format(self._sequence))
            self._sequence = keyutils.KeySequence()
            self._count = ''
            self.keystring_updated.emit('')