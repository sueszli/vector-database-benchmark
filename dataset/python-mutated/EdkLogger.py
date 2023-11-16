from __future__ import absolute_import
import Common.LongFilePathOs as os, sys, logging
import traceback
from .BuildToolError import *
try:
    from logging.handlers import QueueHandler
except:

    class QueueHandler(logging.Handler):
        """
        This handler sends events to a queue. Typically, it would be used together
        with a multiprocessing Queue to centralise logging to file in one process
        (in a multi-process application), so as to avoid file write contention
        between processes.

        This code is new in Python 3.2, but this class can be copy pasted into
        user code for use with earlier Python versions.
        """

        def __init__(self, queue):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Initialise an instance, using the passed queue.\n            '
            logging.Handler.__init__(self)
            self.queue = queue

        def enqueue(self, record):
            if False:
                i = 10
                return i + 15
            '\n            Enqueue a record.\n\n            The base implementation uses put_nowait. You may want to override\n            this method if you want to use blocking, timeouts or custom queue\n            implementations.\n            '
            self.queue.put_nowait(record)

        def prepare(self, record):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Prepares a record for queuing. The object returned by this method is\n            enqueued.\n\n            The base implementation formats the record to merge the message\n            and arguments, and removes unpickleable items from the record\n            in-place.\n\n            You might want to override this method if you want to convert\n            the record to a dict or JSON string, or send a modified copy\n            of the record while leaving the original intact.\n            '
            msg = self.format(record)
            record.message = msg
            record.msg = msg
            record.args = None
            record.exc_info = None
            record.exc_text = None
            return record

        def emit(self, record):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Emit a record.\n\n            Writes the LogRecord to the queue, preparing it for pickling first.\n            '
            try:
                self.enqueue(self.prepare(record))
            except Exception:
                self.handleError(record)

class BlockQueueHandler(QueueHandler):

    def enqueue(self, record):
        if False:
            return 10
        self.queue.put(record, True)
DEBUG_0 = 1
DEBUG_1 = 2
DEBUG_2 = 3
DEBUG_3 = 4
DEBUG_4 = 5
DEBUG_5 = 6
DEBUG_6 = 7
DEBUG_7 = 8
DEBUG_8 = 9
DEBUG_9 = 10
VERBOSE = 15
INFO = 20
WARN = 30
QUIET = 40
ERROR = 50
SILENT = 99
IsRaiseError = True
_ToolName = os.path.basename(sys.argv[0])
_LogLevels = [DEBUG_0, DEBUG_1, DEBUG_2, DEBUG_3, DEBUG_4, DEBUG_5, DEBUG_6, DEBUG_7, DEBUG_8, DEBUG_9, VERBOSE, WARN, INFO, ERROR, QUIET, SILENT]
_DebugLogger = logging.getLogger('tool_debug')
_DebugFormatter = logging.Formatter('[%(asctime)s.%(msecs)d]: %(message)s', datefmt='%H:%M:%S')
_InfoLogger = logging.getLogger('tool_info')
_InfoFormatter = logging.Formatter('%(message)s')
_ErrorLogger = logging.getLogger('tool_error')
_ErrorFormatter = logging.Formatter('%(message)s')
_ErrorMessageTemplate = '\n\n%(tool)s...\n%(file)s(%(line)s): error %(errorcode)04X: %(msg)s\n\t%(extra)s'
_ErrorMessageTemplateWithoutFile = '\n\n%(tool)s...\n : error %(errorcode)04X: %(msg)s\n\t%(extra)s'
_WarningMessageTemplate = '%(tool)s...\n%(file)s(%(line)s): warning: %(msg)s'
_WarningMessageTemplateWithoutFile = '%(tool)s: : warning: %(msg)s'
_DebugMessageTemplate = '%(file)s(%(line)s): debug: \n    %(msg)s'
_WarningAsError = False

def debug(Level, Message, ExtraData=None):
    if False:
        while True:
            i = 10
    if _DebugLogger.level > Level:
        return
    if Level > DEBUG_9:
        return
    CallerStack = traceback.extract_stack()[-2]
    TemplateDict = {'file': CallerStack[0], 'line': CallerStack[1], 'msg': Message}
    if ExtraData is not None:
        LogText = _DebugMessageTemplate % TemplateDict + '\n    %s' % ExtraData
    else:
        LogText = _DebugMessageTemplate % TemplateDict
    _DebugLogger.log(Level, LogText)

def verbose(Message):
    if False:
        while True:
            i = 10
    return _InfoLogger.log(VERBOSE, Message)

def warn(ToolName, Message, File=None, Line=None, ExtraData=None):
    if False:
        for i in range(10):
            print('nop')
    if _InfoLogger.level > WARN:
        return
    if ToolName is None or ToolName == '':
        ToolName = os.path.basename(traceback.extract_stack()[-2][0])
    if Line is None:
        Line = '...'
    else:
        Line = '%d' % Line
    TemplateDict = {'tool': ToolName, 'file': File, 'line': Line, 'msg': Message}
    if File is not None:
        LogText = _WarningMessageTemplate % TemplateDict
    else:
        LogText = _WarningMessageTemplateWithoutFile % TemplateDict
    if ExtraData is not None:
        LogText += '\n    %s' % ExtraData
    _InfoLogger.log(WARN, LogText)
    if _WarningAsError == True:
        raise FatalError(WARNING_AS_ERROR)
info = _InfoLogger.info

def error(ToolName, ErrorCode, Message=None, File=None, Line=None, ExtraData=None, RaiseError=IsRaiseError):
    if False:
        return 10
    if Line is None:
        Line = '...'
    else:
        Line = '%d' % Line
    if Message is None:
        if ErrorCode in gErrorMessage:
            Message = gErrorMessage[ErrorCode]
        else:
            Message = gErrorMessage[UNKNOWN_ERROR]
    if ExtraData is None:
        ExtraData = ''
    TemplateDict = {'tool': _ToolName, 'file': File, 'line': Line, 'errorcode': ErrorCode, 'msg': Message, 'extra': ExtraData}
    if File is not None:
        LogText = _ErrorMessageTemplate % TemplateDict
    else:
        LogText = _ErrorMessageTemplateWithoutFile % TemplateDict
    _ErrorLogger.log(ERROR, LogText)
    if RaiseError and IsRaiseError:
        raise FatalError(ErrorCode)
quiet = _ErrorLogger.error

def LogClientInitialize(log_q):
    if False:
        for i in range(10):
            print('nop')
    _DebugLogger.setLevel(INFO)
    _DebugChannel = BlockQueueHandler(log_q)
    _DebugChannel.setFormatter(_DebugFormatter)
    _DebugLogger.addHandler(_DebugChannel)
    _InfoLogger.setLevel(INFO)
    _InfoChannel = BlockQueueHandler(log_q)
    _InfoChannel.setFormatter(_InfoFormatter)
    _InfoLogger.addHandler(_InfoChannel)
    _ErrorLogger.setLevel(INFO)
    _ErrorCh = BlockQueueHandler(log_q)
    _ErrorCh.setFormatter(_ErrorFormatter)
    _ErrorLogger.addHandler(_ErrorCh)

def SetLevel(Level):
    if False:
        i = 10
        return i + 15
    if Level not in _LogLevels:
        info('Not supported log level (%d). Use default level instead.' % Level)
        Level = INFO
    _DebugLogger.setLevel(Level)
    _InfoLogger.setLevel(Level)
    _ErrorLogger.setLevel(Level)

def Initialize():
    if False:
        return 10
    _DebugLogger.setLevel(INFO)
    _DebugChannel = logging.StreamHandler(sys.stdout)
    _DebugChannel.setFormatter(_DebugFormatter)
    _DebugLogger.addHandler(_DebugChannel)
    _InfoLogger.setLevel(INFO)
    _InfoChannel = logging.StreamHandler(sys.stdout)
    _InfoChannel.setFormatter(_InfoFormatter)
    _InfoLogger.addHandler(_InfoChannel)
    _ErrorLogger.setLevel(INFO)
    _ErrorCh = logging.StreamHandler(sys.stderr)
    _ErrorCh.setFormatter(_ErrorFormatter)
    _ErrorLogger.addHandler(_ErrorCh)

def InitializeForUnitTest():
    if False:
        for i in range(10):
            print('nop')
    Initialize()
    SetLevel(SILENT)

def GetLevel():
    if False:
        i = 10
        return i + 15
    return _InfoLogger.getEffectiveLevel()

def SetWarningAsError():
    if False:
        return 10
    global _WarningAsError
    _WarningAsError = True

def SetLogFile(LogFile):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists(LogFile):
        os.remove(LogFile)
    _Ch = logging.FileHandler(LogFile)
    _Ch.setFormatter(_DebugFormatter)
    _DebugLogger.addHandler(_Ch)
    _Ch = logging.FileHandler(LogFile)
    _Ch.setFormatter(_InfoFormatter)
    _InfoLogger.addHandler(_Ch)
    _Ch = logging.FileHandler(LogFile)
    _Ch.setFormatter(_ErrorFormatter)
    _ErrorLogger.addHandler(_Ch)
if __name__ == '__main__':
    pass