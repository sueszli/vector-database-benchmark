"""Loggers and utilities related to Qt logging."""
import argparse
import contextlib
import faulthandler
import logging
import sys
import traceback
from typing import Iterator, Optional
from qutebrowser.qt import core as qtcore
from qutebrowser.utils import log
_args = None

def init(args: argparse.Namespace) -> None:
    if False:
        while True:
            i = 10
    'Install Qt message handler based on the argparse namespace passed.'
    global _args
    _args = args
    qtcore.qInstallMessageHandler(qt_message_handler)

@qtcore.pyqtSlot()
def shutdown_log() -> None:
    if False:
        return 10
    qtcore.qInstallMessageHandler(None)

@contextlib.contextmanager
def disable_qt_msghandler() -> Iterator[None]:
    if False:
        print('Hello World!')
    'Contextmanager which temporarily disables the Qt message handler.'
    old_handler = qtcore.qInstallMessageHandler(None)
    try:
        yield
    finally:
        qtcore.qInstallMessageHandler(old_handler)

def qt_message_handler(msg_type: qtcore.QtMsgType, context: qtcore.QMessageLogContext, msg: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Qt message handler to redirect qWarning etc. to the logging system.\n\n    Args:\n        msg_type: The level of the message.\n        context: The source code location of the message.\n        msg: The message text.\n    '
    qt_to_logging = {qtcore.QtMsgType.QtDebugMsg: logging.DEBUG, qtcore.QtMsgType.QtWarningMsg: logging.WARNING, qtcore.QtMsgType.QtCriticalMsg: logging.ERROR, qtcore.QtMsgType.QtFatalMsg: logging.CRITICAL, qtcore.QtMsgType.QtInfoMsg: logging.INFO}
    suppressed_msgs = ['libpng warning: iCCP: Not recognizing known sRGB profile that has been edited', 'libpng warning: iCCP: known incorrect sRGB profile', 'OpenType support missing for script ', 'QNetworkReplyImplPrivate::error: Internal problem, this method must only be called once.', 'load glyph failed ', 'content-type missing in HTTP POST, defaulting to application/x-www-form-urlencoded. Use QNetworkRequest::setHeader() to fix this problem.', 'Using blocking call!', '"Method "GetAll" with signature "s" on interface "org.freedesktop.DBus.Properties" doesn\'t exist', '"Method \\"GetAll\\" with signature \\"s\\" on interface \\"org.freedesktop.DBus.Properties\\" doesn\'t exist\\n"', 'WOFF support requires QtWebKit to be built with zlib support.', 'QXcbWindow: Unhandled client message: "_E_', 'QXcbWindow: Unhandled client message: "_ECORE_', 'QXcbWindow: Unhandled client message: "_GTK_', 'SetProcessDpiAwareness failed:', 'QObject::connect: Cannot connect (null)::stateChanged(QNetworkSession::State) to QNetworkReplyHttpImpl::_q_networkSessionStateChanged(QNetworkSession::State)', "Image of format '' blocked because it is not considered safe. If you are sure it is safe to do so, you can white-list the format by setting the environment variable QTWEBKIT_IMAGEFORMAT_WHITELIST=", 'QSslSocket: cannot resolve ', 'QSslSocket: cannot call unresolved function ', 'Remote debugging server started successfully. Try pointing a Chromium-based browser to ', 'QXcbClipboard: SelectionRequest too old', 'QXcbWindow: Unhandled client message: ""', 'QObject::disconnect: Unexpected null parameter', 'Attribute Qt::AA_ShareOpenGLContexts must be set before QCoreApplication is created.', 'GL format 0 is not supported']
    if sys.platform == 'darwin':
        suppressed_msgs += ['virtual void QSslSocketBackendPrivate::transmit() SSLRead failed with: -9805']
    if not msg:
        msg = 'Logged empty message!'
    if any((msg.strip().startswith(pattern) for pattern in suppressed_msgs)):
        level = logging.DEBUG
    elif context.category == 'qt.webenginecontext' and (msg.strip().startswith('GL Type: ') or msg.strip().startswith('GLImplementation:')):
        level = logging.DEBUG
    else:
        level = qt_to_logging[msg_type]
    if context.line is None:
        lineno = -1
    else:
        lineno = context.line
    if context.function is None:
        func = 'none'
    elif ':' in context.function:
        func = '"{}"'.format(context.function)
    else:
        func = context.function
    if context.category is None or context.category == 'default':
        name = 'qt'
    else:
        name = 'qt-' + context.category
    if msg.splitlines()[0] == 'This application failed to start because it could not find or load the Qt platform plugin "xcb".':
        msg += '\n\nOn Archlinux, this should fix the problem:\n    pacman -S libxkbcommon-x11'
        faulthandler.disable()
    assert _args is not None
    if _args.debug:
        stack: Optional[str] = ''.join(traceback.format_stack())
    else:
        stack = None
    record = log.qt.makeRecord(name=name, level=level, fn=context.file, lno=lineno, msg=msg, args=(), exc_info=None, func=func, sinfo=stack)
    log.qt.handle(record)

class QtWarningFilter(logging.Filter):
    """Filter to filter Qt warnings.

    Attributes:
        _pattern: The start of the message.
    """

    def __init__(self, pattern: str) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._pattern = pattern

    def filter(self, record: logging.LogRecord) -> bool:
        if False:
            while True:
                i = 10
        'Determine if the specified record is to be logged.'
        do_log = not record.msg.strip().startswith(self._pattern)
        return do_log

@contextlib.contextmanager
def hide_qt_warning(pattern: str, logger: str='qt') -> Iterator[None]:
    if False:
        i = 10
        return i + 15
    'Hide Qt warnings matching the given regex.'
    log_filter = QtWarningFilter(pattern)
    logger_obj = logging.getLogger(logger)
    logger_obj.addFilter(log_filter)
    try:
        yield
    finally:
        logger_obj.removeFilter(log_filter)