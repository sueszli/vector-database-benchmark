"""Custom useful data types."""
import html
import operator
import enum
import dataclasses
from typing import Optional, Sequence, TypeVar, Union
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, QObject, QTimer
from qutebrowser.qt.core import QUrl
from qutebrowser.utils import log, qtutils, utils
_T = TypeVar('_T', bound=utils.Comparable)

class Unset:
    """Class for an unset object."""
    __slots__ = ()

    def __repr__(self) -> str:
        if False:
            return 10
        return '<UNSET>'
UNSET = Unset()

class NeighborList(Sequence[_T]):
    """A list of items which saves its current position.

    Class attributes:
        Modes: Different modes, see constructor documentation.

    Attributes:
        fuzzyval: The value which is currently set but not in the list.
        _idx: The current position in the list.
        _items: A list of all items, accessed through item property.
        _mode: The current mode.
    """

    class Modes(enum.Enum):
        """Behavior for the 'mode' argument."""
        edge = enum.auto()
        exception = enum.auto()

    def __init__(self, items: Sequence[_T]=None, default: Union[_T, Unset]=UNSET, mode: Modes=Modes.exception) -> None:
        if False:
            print('Hello World!')
        'Constructor.\n\n        Args:\n            items: The list of items to iterate in.\n            _default: The initially selected value.\n            _mode: Behavior when the first/last item is reached.\n                   Modes.edge: Go to the first/last item\n                   Modes.exception: Raise an IndexError.\n        '
        if not isinstance(mode, self.Modes):
            raise TypeError('Mode {} is not a Modes member!'.format(mode))
        if items is None:
            self._items: Sequence[_T] = []
        else:
            self._items = list(items)
        self._default = default
        if not isinstance(default, Unset):
            idx = self._items.index(default)
            self._idx: Optional[int] = idx
        else:
            self._idx = None
        self._mode = mode
        self.fuzzyval: Optional[int] = None

    def __getitem__(self, key: int) -> _T:
        if False:
            print('Hello World!')
        return self._items[key]

    def __len__(self) -> int:
        if False:
            print('Hello World!')
        return len(self._items)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return utils.get_repr(self, items=self._items, mode=self._mode, idx=self._idx, fuzzyval=self.fuzzyval)

    def _snap_in(self, offset: int) -> bool:
        if False:
            print('Hello World!')
        'Set the current item to the closest item to self.fuzzyval.\n\n        Args:\n            offset: negative to get the next smaller item, positive for the\n                    next bigger one.\n\n        Return:\n            True if the value snapped in (changed),\n            False when the value already was in the list.\n        '
        assert isinstance(self.fuzzyval, (int, float)), self.fuzzyval
        op = operator.le if offset < 0 else operator.ge
        items = [(idx, e) for (idx, e) in enumerate(self._items) if op(e, self.fuzzyval)]
        if items:
            item = min(items, key=lambda tpl: abs(self.fuzzyval - tpl[1]))
        else:
            sorted_items = sorted(enumerate(self.items), key=lambda e: e[1])
            idx = 0 if offset < 0 else -1
            item = sorted_items[idx]
        self._idx = item[0]
        return self.fuzzyval not in self._items

    def _get_new_item(self, offset: int) -> _T:
        if False:
            i = 10
            return i + 15
        'Logic for getitem to get the item at offset.\n\n        Args:\n            offset: The offset of the current item, relative to the last one.\n\n        Return:\n            The new item.\n        '
        assert self._idx is not None
        try:
            if self._idx + offset >= 0:
                new = self._items[self._idx + offset]
            else:
                raise IndexError
        except IndexError:
            if self._mode == self.Modes.edge:
                assert offset != 0
                if offset > 0:
                    new = self.lastitem()
                else:
                    new = self.firstitem()
            elif self._mode == self.Modes.exception:
                raise
        else:
            self._idx += offset
        return new

    @property
    def items(self) -> Sequence[_T]:
        if False:
            i = 10
            return i + 15
        'Getter for items, which should not be set.'
        return self._items

    def getitem(self, offset: int) -> _T:
        if False:
            for i in range(10):
                print('nop')
        'Get the item with a relative position.\n\n        Args:\n            offset: The offset of the current item, relative to the last one.\n\n        Return:\n            The new item.\n        '
        log.misc.debug('{} items, idx {}, offset {}'.format(len(self._items), self._idx, offset))
        if not self._items:
            raise IndexError('No items found!')
        if self.fuzzyval is not None:
            snapped = self._snap_in(offset)
            if snapped and offset > 0:
                offset -= 1
            elif snapped:
                offset += 1
            self.fuzzyval = None
        return self._get_new_item(offset)

    def curitem(self) -> _T:
        if False:
            for i in range(10):
                print('nop')
        'Get the current item in the list.'
        if self._idx is not None:
            return self._items[self._idx]
        else:
            raise IndexError('No current item!')

    def nextitem(self) -> _T:
        if False:
            print('Hello World!')
        'Get the next item in the list.'
        return self.getitem(1)

    def previtem(self) -> _T:
        if False:
            print('Hello World!')
        'Get the previous item in the list.'
        return self.getitem(-1)

    def firstitem(self) -> _T:
        if False:
            return 10
        'Get the first item in the list.'
        if not self._items:
            raise IndexError('No items found!')
        self._idx = 0
        return self.curitem()

    def lastitem(self) -> _T:
        if False:
            while True:
                i = 10
        'Get the last item in the list.'
        if not self._items:
            raise IndexError('No items found!')
        self._idx = len(self._items) - 1
        return self.curitem()

    def reset(self) -> _T:
        if False:
            i = 10
            return i + 15
        'Reset the position to the default.'
        if self._default is UNSET:
            raise ValueError('No default set!')
        self._idx = self._items.index(self._default)
        return self.curitem()

class PromptMode(enum.Enum):
    """The mode of a Question."""
    yesno = enum.auto()
    text = enum.auto()
    user_pwd = enum.auto()
    alert = enum.auto()
    download = enum.auto()

class ClickTarget(enum.Enum):
    """How to open a clicked link."""
    normal = enum.auto()
    tab = enum.auto()
    tab_bg = enum.auto()
    window = enum.auto()
    hover = enum.auto()

class KeyMode(enum.Enum):
    """Key input modes."""
    normal = enum.auto()
    hint = enum.auto()
    command = enum.auto()
    yesno = enum.auto()
    prompt = enum.auto()
    insert = enum.auto()
    passthrough = enum.auto()
    caret = enum.auto()
    set_mark = enum.auto()
    jump_mark = enum.auto()
    record_macro = enum.auto()
    run_macro = enum.auto()
    register = enum.auto()

class Exit(enum.IntEnum):
    """Exit statuses for errors. Needs to be an int for sys.exit."""
    ok = 0
    reserved = 1
    exception = 2
    err_ipc = 3
    err_init = 4

class LoadStatus(enum.Enum):
    """Load status of a tab."""
    none = enum.auto()
    success = enum.auto()
    success_https = enum.auto()
    error = enum.auto()
    warn = enum.auto()
    loading = enum.auto()

class Backend(enum.Enum):
    """The backend being used (usertypes.backend)."""
    QtWebKit = enum.auto()
    QtWebEngine = enum.auto()

class JsWorld(enum.Enum):
    """World/context to run JavaScript code in."""
    main = enum.auto()
    application = enum.auto()
    user = enum.auto()
    jseval = enum.auto()

class JsLogLevel(enum.Enum):
    """Log level of a JS message.

    This needs to match up with the keys allowed for the
    content.javascript.log setting.
    """
    unknown = enum.auto()
    info = enum.auto()
    warning = enum.auto()
    error = enum.auto()

class MessageLevel(enum.Enum):
    """The level of a message being shown."""
    error = enum.auto()
    warning = enum.auto()
    info = enum.auto()

class IgnoreCase(enum.Enum):
    """Possible values for the 'search.ignore_case' setting."""
    smart = enum.auto()
    never = enum.auto()
    always = enum.auto()

class CommandValue(enum.Enum):
    """Special values which are injected when running a command handler."""
    count = enum.auto()
    win_id = enum.auto()
    cur_tab = enum.auto()
    count_tab = enum.auto()

class Question(QObject):
    """A question asked to the user, e.g. via the status bar.

    Note the creator is responsible for cleaning up the question after it
    doesn't need it anymore, e.g. via connecting Question.completed to
    Question.deleteLater.

    Attributes:
        mode: A PromptMode enum member.
              yesno: A question which can be answered with yes/no.
              text: A question which requires a free text answer.
              user_pwd: A question for a username and password.
        default: The default value.
                 For yesno, None (no default), True or False.
                 For text, a default text as string.
                 For user_pwd, a default username as string.
        title: The question title to show.
        text: The prompt text to display to the user.
        url: Any URL referenced in prompts.
        option: Boolean option to be set when answering always/never.
        answer: The value the user entered (as password for user_pwd).
        is_aborted: Whether the question was aborted.
        interrupted: Whether the question was interrupted by another one.

    Signals:
        answered: Emitted when the question has been answered by the user.
                  arg: The answer to the question.
        cancelled: Emitted when the question has been cancelled by the user.
        aborted: Emitted when the question was aborted programmatically.
                 In this case, cancelled is not emitted.
        answered_yes: Convenience signal emitted when a yesno question was
                      answered with yes.
        answered_no: Convenience signal emitted when a yesno question was
                     answered with no.
        completed: Emitted when the question was completed in any way.
    """
    answered = pyqtSignal(object)
    cancelled = pyqtSignal()
    aborted = pyqtSignal()
    answered_yes = pyqtSignal()
    answered_no = pyqtSignal()
    completed = pyqtSignal()

    def __init__(self, parent: QObject=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.mode: Optional[PromptMode] = None
        self.default: Union[bool, str, None] = None
        self.title: Optional[str] = None
        self.text: Optional[str] = None
        self.url: Optional[str] = None
        self.option: Optional[bool] = None
        self.answer: Union[str, bool, None] = None
        self.is_aborted = False
        self.interrupted = False

    def __repr__(self) -> str:
        if False:
            return 10
        return utils.get_repr(self, title=self.title, text=self.text, mode=self.mode, default=self.default, option=self.option)

    @pyqtSlot()
    def done(self) -> None:
        if False:
            return 10
        'Must be called when the question was answered completely.'
        self.answered.emit(self.answer)
        if self.mode == PromptMode.yesno:
            if self.answer:
                self.answered_yes.emit()
            else:
                self.answered_no.emit()
        self.completed.emit()

    @pyqtSlot()
    def cancel(self) -> None:
        if False:
            i = 10
            return i + 15
        'Cancel the question (resulting from user-input).'
        self.cancelled.emit()
        self.completed.emit()

    @pyqtSlot()
    def abort(self) -> None:
        if False:
            return 10
        'Abort the question.'
        if self.is_aborted:
            log.misc.debug('Question was already aborted')
            return
        self.is_aborted = True
        self.aborted.emit()
        self.completed.emit()

class Timer(QTimer):
    """A timer which has a name to show in __repr__ and checks for overflows.

    Attributes:
        _name: The name of the timer.
    """

    def __init__(self, parent: QObject=None, name: str=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        if name is None:
            self._name = 'unnamed'
        else:
            self.setObjectName(name)
            self._name = name

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return utils.get_repr(self, name=self._name)

    def setInterval(self, msec: int) -> None:
        if False:
            while True:
                i = 10
        'Extend setInterval to check for overflows.'
        qtutils.check_overflow(msec, 'int')
        super().setInterval(msec)

    def start(self, msec: int=None) -> None:
        if False:
            print('Hello World!')
        'Extend start to check for overflows.'
        if msec is not None:
            qtutils.check_overflow(msec, 'int')
            super().start(msec)
        else:
            super().start()

class UndeferrableError(Exception):
    """An AbstractCertificateErrorWrapper isn't deferrable."""

class AbstractCertificateErrorWrapper:
    """A wrapper over an SSL/certificate error."""

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._certificate_accepted: Optional[bool] = None

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def is_overridable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def html(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<p>{html.escape(str(self))}</p>'

    def accept_certificate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._certificate_accepted = True

    def reject_certificate(self) -> None:
        if False:
            while True:
                i = 10
        self._certificate_accepted = False

    def defer(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def certificate_was_accepted(self) -> bool:
        if False:
            while True:
                i = 10
        'Check whether the certificate was accepted by the user.'
        if not self.is_overridable():
            return False
        if self._certificate_accepted is None:
            raise ValueError('No decision taken yet')
        return self._certificate_accepted

@dataclasses.dataclass
class NavigationRequest:
    """A request to navigate to the given URL."""

    class Type(enum.Enum):
        """The type of a request.

        Based on QWebEngineUrlRequestInfo::NavigationType and QWebPage::NavigationType.
        """
        link_clicked = 1
        typed = 2
        form_submitted = 3
        form_resubmitted = 4
        back_forward = 5
        reload = 6
        redirect = 7
        other = 8
    url: QUrl
    navigation_type: Type
    is_main_frame: bool
    accepted: bool = True