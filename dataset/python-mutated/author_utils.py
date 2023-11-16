import ctypes as ctypes
from ctypes import wintypes as wintypes
import os
import sys
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
advapi32 = ctypes.WinDLL('advapi32', use_last_error=True)
ERROR_INVALID_FUNCTION = 1
ERROR_FILE_NOT_FOUND = 2
ERROR_PATH_NOT_FOUND = 3
ERROR_ACCESS_DENIED = 5
ERROR_SHARING_VIOLATION = 32
SE_FILE_OBJECT = 1
OWNER_SECURITY_INFORMATION = 1
GROUP_SECURITY_INFORMATION = 2
DACL_SECURITY_INFORMATION = 4
SACL_SECURITY_INFORMATION = 8
LABEL_SECURITY_INFORMATION = 16
_DEFAULT_SECURITY_INFORMATION = OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION | LABEL_SECURITY_INFORMATION
LPDWORD = ctypes.POINTER(wintypes.DWORD)
SE_OBJECT_TYPE = wintypes.DWORD
SECURITY_INFORMATION = wintypes.DWORD

class SID_NAME_USE(wintypes.DWORD):
    _sid_types = dict(enumerate('\n        User Group Domain Alias WellKnownGroup DeletedAccount\n        Invalid Unknown Computer Label'.split(), 1))

    def __init__(self, value=None):
        if False:
            while True:
                i = 10
        if value is not None:
            if value not in self.sid_types:
                raise ValueError('invalid SID type')
            wintypes.DWORD.__init__(value)

    def __str__(self):
        if False:
            return 10
        if self.value not in self._sid_types:
            raise ValueError('invalid SID type')
        return self._sid_types[self.value]

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'SID_NAME_USE(%s)' % self.value
PSID_NAME_USE = ctypes.POINTER(SID_NAME_USE)

class PLOCAL(wintypes.LPVOID):
    _needs_free = False

    def __init__(self, value=None, needs_free=False):
        if False:
            print('Hello World!')
        super(PLOCAL, self).__init__(value)
        self._needs_free = needs_free

    def __del__(self):
        if False:
            while True:
                i = 10
        if self and self._needs_free:
            kernel32.LocalFree(self)
            self._needs_free = False
PACL = PLOCAL

class PSID(PLOCAL):

    def __init__(self, value=None, needs_free=False):
        if False:
            print('Hello World!')
        super(PSID, self).__init__(value, needs_free)

    def __str__(self):
        if False:
            while True:
                i = 10
        if not self:
            raise ValueError('NULL pointer access')
        sid = wintypes.LPWSTR()
        advapi32.ConvertSidToStringSidW(self, ctypes.byref(sid))
        try:
            return sid.value
        finally:
            if sid:
                kernel32.LocalFree(sid)

class PSECURITY_DESCRIPTOR(PLOCAL):

    def __init__(self, value=None, needs_free=False):
        if False:
            while True:
                i = 10
        super(PSECURITY_DESCRIPTOR, self).__init__(value, needs_free)
        self.pOwner = PSID()
        self.pGroup = PSID()
        self.pDacl = PACL()
        self.pSacl = PACL()
        self.pOwner._SD = self
        self.pGroup._SD = self
        self.pDacl._SD = self
        self.pSacl._SD = self

    def get_owner(self, system_name=None):
        if False:
            print('Hello World!')
        if not self or not self.pOwner:
            raise ValueError('NULL pointer access')
        return look_up_account_sid(self.pOwner, system_name)

    def get_group(self, system_name=None):
        if False:
            return 10
        if not self or not self.pGroup:
            raise ValueError('NULL pointer access')
        return look_up_account_sid(self.pGroup, system_name)

def _check_bool(result, func, args):
    if False:
        print('Hello World!')
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return args
advapi32.ConvertSidToStringSidW.errcheck = _check_bool
advapi32.ConvertSidToStringSidW.argtypes = (PSID, ctypes.POINTER(wintypes.LPWSTR))
advapi32.LookupAccountSidW.errcheck = _check_bool
advapi32.LookupAccountSidW.argtypes = (wintypes.LPCWSTR, PSID, wintypes.LPCWSTR, LPDWORD, wintypes.LPCWSTR, LPDWORD, PSID_NAME_USE)
advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
advapi32.GetNamedSecurityInfoW.argtypes = (wintypes.LPWSTR, SE_OBJECT_TYPE, SECURITY_INFORMATION, ctypes.POINTER(PSID), ctypes.POINTER(PSID), ctypes.POINTER(PACL), ctypes.POINTER(PACL), ctypes.POINTER(PSECURITY_DESCRIPTOR))

def look_up_account_sid(sid, system_name=None):
    if False:
        print('Hello World!')
    SIZE = 256
    name = ctypes.create_unicode_buffer(SIZE)
    domain = ctypes.create_unicode_buffer(SIZE)
    cch_name = wintypes.DWORD(SIZE)
    cch_domain = wintypes.DWORD(SIZE)
    sid_type = SID_NAME_USE()
    advapi32.LookupAccountSidW(system_name, sid, name, ctypes.byref(cch_name), domain, ctypes.byref(cch_domain), ctypes.byref(sid_type))
    return (name.value, domain.value, sid_type)

def get_file_security(filename, request=_DEFAULT_SECURITY_INFORMATION):
    if False:
        return 10
    pSD = PSECURITY_DESCRIPTOR(needs_free=True)
    error = advapi32.GetNamedSecurityInfoW(filename, SE_FILE_OBJECT, request, ctypes.byref(pSD.pOwner), ctypes.byref(pSD.pGroup), ctypes.byref(pSD.pDacl), ctypes.byref(pSD.pSacl), ctypes.byref(pSD))
    if error != 0:
        raise ctypes.WinError(error)
    return pSD

def get_author(filename):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(filename, bytes):
        if hasattr(os, 'fsdecode'):
            filename = os.fsdecode(filename)
        else:
            filename = filename.decode(sys.getfilesystemencoding())
    pSD = get_file_security(filename)
    (owner_name, owner_domain, owner_sid_type) = pSD.get_owner()
    if owner_domain:
        owner_name = '{}\\{}'.format(owner_domain, owner_name)
    return owner_name