"""Event Log Utilities - helper for win32evtlog.pyd
"""
import win32api
import win32con
import win32evtlog
import winerror
error = win32api.error
langid = win32api.MAKELANGID(win32con.LANG_NEUTRAL, win32con.SUBLANG_NEUTRAL)

def AddSourceToRegistry(appName, msgDLL=None, eventLogType='Application', eventLogFlags=None, categoryDLL=None, categoryCount=0):
    if False:
        return 10
    'Add a source of messages to the event log.\n\n    Allows Python program to register a custom source of messages in the\n    registry.  You must also provide the DLL name that has the message table, so the\n    full message text appears in the event log.\n\n    Note that the win32evtlog.pyd file has a number of string entries with just "%1"\n    built in, so many Python programs can simply use this DLL.  Disadvantages are that\n    you do not get language translation, and the full text is stored in the event log,\n    blowing the size of the log up.\n    '
    if msgDLL is None:
        msgDLL = win32evtlog.__file__
    hkey = win32api.RegCreateKey(win32con.HKEY_LOCAL_MACHINE, f'SYSTEM\\CurrentControlSet\\Services\\EventLog\\{eventLogType}\\{appName}')
    win32api.RegSetValueEx(hkey, 'EventMessageFile', 0, win32con.REG_EXPAND_SZ, msgDLL)
    if eventLogFlags is None:
        eventLogFlags = win32evtlog.EVENTLOG_ERROR_TYPE | win32evtlog.EVENTLOG_WARNING_TYPE | win32evtlog.EVENTLOG_INFORMATION_TYPE
    win32api.RegSetValueEx(hkey, 'TypesSupported', 0, win32con.REG_DWORD, eventLogFlags)
    if categoryCount > 0:
        if categoryDLL is None:
            categoryDLL = win32evtlog.__file__
        win32api.RegSetValueEx(hkey, 'CategoryMessageFile', 0, win32con.REG_EXPAND_SZ, categoryDLL)
        win32api.RegSetValueEx(hkey, 'CategoryCount', 0, win32con.REG_DWORD, categoryCount)
    win32api.RegCloseKey(hkey)

def RemoveSourceFromRegistry(appName, eventLogType='Application'):
    if False:
        while True:
            i = 10
    'Removes a source of messages from the event log.'
    try:
        win32api.RegDeleteKey(win32con.HKEY_LOCAL_MACHINE, f'SYSTEM\\CurrentControlSet\\Services\\EventLog\\{eventLogType}\\{appName}')
    except win32api.error as exc:
        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise

def ReportEvent(appName, eventID, eventCategory=0, eventType=win32evtlog.EVENTLOG_ERROR_TYPE, strings=None, data=None, sid=None):
    if False:
        while True:
            i = 10
    'Report an event for a previously added event source.'
    hAppLog = win32evtlog.RegisterEventSource(None, appName)
    win32evtlog.ReportEvent(hAppLog, eventType, eventCategory, eventID, sid, strings, data)
    win32evtlog.DeregisterEventSource(hAppLog)

def FormatMessage(eventLogRecord, logType='Application'):
    if False:
        return 10
    'Given a tuple from ReadEventLog, and optionally where the event\n    record came from, load the message, and process message inserts.\n\n    Note that this function may raise win32api.error.  See also the\n    function SafeFormatMessage which will return None if the message can\n    not be processed.\n    '
    keyName = 'SYSTEM\\CurrentControlSet\\Services\\EventLog\\{}\\{}'.format(logType, eventLogRecord.SourceName)
    handle = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, keyName)
    try:
        dllNames = win32api.RegQueryValueEx(handle, 'EventMessageFile')[0].split(';')
        data = None
        for dllName in dllNames:
            try:
                dllName = win32api.ExpandEnvironmentStrings(dllName)
                dllHandle = win32api.LoadLibraryEx(dllName, 0, win32con.LOAD_LIBRARY_AS_DATAFILE)
                try:
                    data = win32api.FormatMessageW(win32con.FORMAT_MESSAGE_FROM_HMODULE, dllHandle, eventLogRecord.EventID, langid, eventLogRecord.StringInserts)
                finally:
                    win32api.FreeLibrary(dllHandle)
            except win32api.error:
                pass
            if data is not None:
                break
    finally:
        win32api.RegCloseKey(handle)
    return data or ''

def SafeFormatMessage(eventLogRecord, logType=None):
    if False:
        while True:
            i = 10
    'As for FormatMessage, except returns an error message if\n    the message can not be processed.\n    '
    if logType is None:
        logType = 'Application'
    try:
        return FormatMessage(eventLogRecord, logType)
    except win32api.error:
        if eventLogRecord.StringInserts is None:
            desc = ''
        else:
            desc = ', '.join(eventLogRecord.StringInserts)
        return '<The description for Event ID ( %d ) in Source ( %r ) could not be found. It contains the following insertion string(s):%r.>' % (winerror.HRESULT_CODE(eventLogRecord.EventID), eventLogRecord.SourceName, desc)

def FeedEventLogRecords(feeder, machineName=None, logName='Application', readFlags=None):
    if False:
        for i in range(10):
            print('nop')
    if readFlags is None:
        readFlags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    h = win32evtlog.OpenEventLog(machineName, logName)
    try:
        while 1:
            objects = win32evtlog.ReadEventLog(h, readFlags, 0)
            if not objects:
                break
            map(lambda item, feeder=feeder: feeder(*(item,)), objects)
    finally:
        win32evtlog.CloseEventLog(h)