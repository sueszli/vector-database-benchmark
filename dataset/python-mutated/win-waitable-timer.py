import contextlib
from datetime import datetime, timedelta, timezone
import cffi
import trio
from trio._core._windows_cffi import ffi, kernel32, raise_winerror
with contextlib.suppress(cffi.CDefError):
    ffi.cdef('\ntypedef struct _PROCESS_LEAP_SECOND_INFO {\n  ULONG Flags;\n  ULONG Reserved;\n} PROCESS_LEAP_SECOND_INFO, *PPROCESS_LEAP_SECOND_INFO;\n\ntypedef struct _SYSTEMTIME {\n  WORD wYear;\n  WORD wMonth;\n  WORD wDayOfWeek;\n  WORD wDay;\n  WORD wHour;\n  WORD wMinute;\n  WORD wSecond;\n  WORD wMilliseconds;\n} SYSTEMTIME, *PSYSTEMTIME, *LPSYSTEMTIME;\n')
ffi.cdef('\ntypedef LARGE_INTEGER FILETIME;\ntypedef FILETIME* LPFILETIME;\n\nHANDLE CreateWaitableTimerW(\n  LPSECURITY_ATTRIBUTES lpTimerAttributes,\n  BOOL                  bManualReset,\n  LPCWSTR               lpTimerName\n);\n\nBOOL SetWaitableTimer(\n  HANDLE              hTimer,\n  const LPFILETIME    lpDueTime,\n  LONG                lPeriod,\n  void*               pfnCompletionRoutine,\n  LPVOID              lpArgToCompletionRoutine,\n  BOOL                fResume\n);\n\nBOOL SetProcessInformation(\n  HANDLE                    hProcess,\n  /* Really an enum, PROCESS_INFORMATION_CLASS */\n  int32_t                   ProcessInformationClass,\n  LPVOID                    ProcessInformation,\n  DWORD                     ProcessInformationSize\n);\n\nvoid GetSystemTimeAsFileTime(\n  LPFILETIME       lpSystemTimeAsFileTime\n);\n\nBOOL SystemTimeToFileTime(\n  const SYSTEMTIME *lpSystemTime,\n  LPFILETIME       lpFileTime\n);\n', override=True)
ProcessLeapSecondInfo = 8
PROCESS_LEAP_SECOND_INFO_FLAG_ENABLE_SIXTY_SECOND = 1

def set_leap_seconds_enabled(enabled):
    if False:
        print('Hello World!')
    plsi = ffi.new('PROCESS_LEAP_SECOND_INFO*')
    if enabled:
        plsi.Flags = PROCESS_LEAP_SECOND_INFO_FLAG_ENABLE_SIXTY_SECOND
    else:
        plsi.Flags = 0
    plsi.Reserved = 0
    if not kernel32.SetProcessInformation(ffi.cast('HANDLE', -1), ProcessLeapSecondInfo, plsi, ffi.sizeof('PROCESS_LEAP_SECOND_INFO')):
        raise_winerror()

def now_as_filetime():
    if False:
        return 10
    ft = ffi.new('LARGE_INTEGER*')
    kernel32.GetSystemTimeAsFileTime(ft)
    return ft[0]
FILETIME_TICKS_PER_SECOND = 10 ** 7
FILETIME_EPOCH = datetime.strptime('1601-01-01 00:00:00 Z', '%Y-%m-%d %H:%M:%S %z')

def py_datetime_to_win_filetime(dt):
    if False:
        print('Hello World!')
    assert dt.tzinfo is timezone.utc
    return round((dt - FILETIME_EPOCH).total_seconds() * FILETIME_TICKS_PER_SECOND)

async def main():
    h = kernel32.CreateWaitableTimerW(ffi.NULL, True, ffi.NULL)
    if not h:
        raise_winerror()
    print(h)
    SECONDS = 2
    wakeup = datetime.now(timezone.utc) + timedelta(seconds=SECONDS)
    wakeup_filetime = py_datetime_to_win_filetime(wakeup)
    wakeup_cffi = ffi.new('LARGE_INTEGER *')
    wakeup_cffi[0] = wakeup_filetime
    print(wakeup_filetime, wakeup_cffi)
    print(f'Sleeping for {SECONDS} seconds (until {wakeup})')
    if not kernel32.SetWaitableTimer(h, wakeup_cffi, 0, ffi.NULL, ffi.NULL, False):
        raise_winerror()
    await trio.hazmat.WaitForSingleObject(h)
    print(f'Current FILETIME: {now_as_filetime()}')
    set_leap_seconds_enabled(False)
    print(f'Current FILETIME: {now_as_filetime()}')
    set_leap_seconds_enabled(True)
    print(f'Current FILETIME: {now_as_filetime()}')
    set_leap_seconds_enabled(False)
    print(f'Current FILETIME: {now_as_filetime()}')
trio.run(main)