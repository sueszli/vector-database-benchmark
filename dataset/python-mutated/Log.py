"""
Logger
"""
from sys import argv
from sys import stdout
from sys import stderr
import os.path
from os import remove
from logging import getLogger
from logging import Formatter
from logging import StreamHandler
from logging import FileHandler
from traceback import extract_stack
from Logger.ToolError import FatalError
from Logger.ToolError import WARNING_AS_ERROR
from Logger.ToolError import gERROR_MESSAGE
from Logger.ToolError import UNKNOWN_ERROR
from Library import GlobalData
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
QUIET_1 = 41
ERROR = 50
SILENT = 60
IS_RAISE_ERROR = True
SUPRESS_ERROR = False
_TOOL_NAME = os.path.basename(argv[0])
_LOG_LEVELS = [DEBUG_0, DEBUG_1, DEBUG_2, DEBUG_3, DEBUG_4, DEBUG_5, DEBUG_6, DEBUG_7, DEBUG_8, DEBUG_9, VERBOSE, WARN, INFO, ERROR, QUIET, QUIET_1, SILENT]
_DEBUG_LOGGER = getLogger('tool_debug')
_DEBUG_FORMATTER = Formatter('[%(asctime)s.%(msecs)d]: %(message)s', datefmt='%H:%M:%S')
_INFO_LOGGER = getLogger('tool_info')
_INFO_FORMATTER = Formatter('%(message)s')
_ERROR_LOGGER = getLogger('tool_error')
_ERROR_FORMATTER = Formatter('%(message)s')
_ERROR_MESSAGE_TEMPLATE = '\n\n%(tool)s...\n%(file)s(%(line)s): error %(errorcode)04X: %(msg)s\n\t%(extra)s'
__ERROR_MESSAGE_TEMPLATE_WITHOUT_FILE = '\n\n%(tool)s...\n : error %(errorcode)04X: %(msg)s\n\t%(extra)s'
_WARNING_MESSAGE_TEMPLATE = '%(tool)s...\n%(file)s(%(line)s): warning: %(msg)s'
_WARNING_MESSAGE_TEMPLATE_WITHOUT_FILE = '%(tool)s: : warning: %(msg)s'
_DEBUG_MESSAGE_TEMPLATE = '%(file)s(%(line)s): debug: \n    %(msg)s'

def Info(msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    _INFO_LOGGER.info(msg, *args, **kwargs)

def Quiet(msg, *args, **kwargs):
    if False:
        print('Hello World!')
    _ERROR_LOGGER.error(msg, *args, **kwargs)

def Debug(Level, Message, ExtraData=None):
    if False:
        while True:
            i = 10
    if _DEBUG_LOGGER.level > Level:
        return
    if Level > DEBUG_9:
        return
    CallerStack = extract_stack()[-2]
    TemplateDict = {'file': CallerStack[0], 'line': CallerStack[1], 'msg': Message}
    if ExtraData is not None:
        LogText = _DEBUG_MESSAGE_TEMPLATE % TemplateDict + '\n    %s' % ExtraData
    else:
        LogText = _DEBUG_MESSAGE_TEMPLATE % TemplateDict
    _DEBUG_LOGGER.log(Level, LogText)

def Verbose(Message):
    if False:
        print('Hello World!')
    return _INFO_LOGGER.log(VERBOSE, Message)

def Warn(ToolName, Message, File=None, Line=None, ExtraData=None):
    if False:
        while True:
            i = 10
    if _INFO_LOGGER.level > WARN:
        return
    if ToolName is None or ToolName == '':
        ToolName = os.path.basename(extract_stack()[-2][0])
    if Line is None:
        Line = '...'
    else:
        Line = '%d' % Line
    TemplateDict = {'tool': ToolName, 'file': File, 'line': Line, 'msg': Message}
    if File is not None:
        LogText = _WARNING_MESSAGE_TEMPLATE % TemplateDict
    else:
        LogText = _WARNING_MESSAGE_TEMPLATE_WITHOUT_FILE % TemplateDict
    if ExtraData is not None:
        LogText += '\n    %s' % ExtraData
    _INFO_LOGGER.log(WARN, LogText)
    if GlobalData.gWARNING_AS_ERROR == True:
        raise FatalError(WARNING_AS_ERROR)

def Error(ToolName, ErrorCode, Message=None, File=None, Line=None, ExtraData=None, RaiseError=IS_RAISE_ERROR):
    if False:
        return 10
    if ToolName:
        pass
    if Line is None:
        Line = '...'
    else:
        Line = '%d' % Line
    if Message is None:
        if ErrorCode in gERROR_MESSAGE:
            Message = gERROR_MESSAGE[ErrorCode]
        else:
            Message = gERROR_MESSAGE[UNKNOWN_ERROR]
    if ExtraData is None:
        ExtraData = ''
    TemplateDict = {'tool': _TOOL_NAME, 'file': File, 'line': Line, 'errorcode': ErrorCode, 'msg': Message, 'extra': ExtraData}
    if File is not None:
        LogText = _ERROR_MESSAGE_TEMPLATE % TemplateDict
    else:
        LogText = __ERROR_MESSAGE_TEMPLATE_WITHOUT_FILE % TemplateDict
    if not SUPRESS_ERROR:
        _ERROR_LOGGER.log(ERROR, LogText)
    if RaiseError:
        raise FatalError(ErrorCode)

def Initialize():
    if False:
        while True:
            i = 10
    _DEBUG_LOGGER.setLevel(INFO)
    _DebugChannel = StreamHandler(stdout)
    _DebugChannel.setFormatter(_DEBUG_FORMATTER)
    _DEBUG_LOGGER.addHandler(_DebugChannel)
    _INFO_LOGGER.setLevel(INFO)
    _InfoChannel = StreamHandler(stdout)
    _InfoChannel.setFormatter(_INFO_FORMATTER)
    _INFO_LOGGER.addHandler(_InfoChannel)
    _ERROR_LOGGER.setLevel(INFO)
    _ErrorCh = StreamHandler(stderr)
    _ErrorCh.setFormatter(_ERROR_FORMATTER)
    _ERROR_LOGGER.addHandler(_ErrorCh)

def SetLevel(Level):
    if False:
        while True:
            i = 10
    if Level not in _LOG_LEVELS:
        Info('Not supported log level (%d). Use default level instead.' % Level)
        Level = INFO
    _DEBUG_LOGGER.setLevel(Level)
    _INFO_LOGGER.setLevel(Level)
    _ERROR_LOGGER.setLevel(Level)

def GetLevel():
    if False:
        while True:
            i = 10
    return _INFO_LOGGER.getEffectiveLevel()

def SetWarningAsError():
    if False:
        while True:
            i = 10
    GlobalData.gWARNING_AS_ERROR = True

def SetLogFile(LogFile):
    if False:
        print('Hello World!')
    if os.path.exists(LogFile):
        remove(LogFile)
    _Ch = FileHandler(LogFile)
    _Ch.setFormatter(_DEBUG_FORMATTER)
    _DEBUG_LOGGER.addHandler(_Ch)
    _Ch = FileHandler(LogFile)
    _Ch.setFormatter(_INFO_FORMATTER)
    _INFO_LOGGER.addHandler(_Ch)
    _Ch = FileHandler(LogFile)
    _Ch.setFormatter(_ERROR_FORMATTER)
    _ERROR_LOGGER.addHandler(_Ch)