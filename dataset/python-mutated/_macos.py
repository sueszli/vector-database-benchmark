import contextlib
import ctypes
import platform
import ssl
import typing
from ctypes import CDLL, POINTER, c_bool, c_char_p, c_int32, c_long, c_uint32, c_ulong, c_void_p
from ctypes.util import find_library
from ._ssl_constants import _set_ssl_context_verify_mode
_mac_version = platform.mac_ver()[0]
_mac_version_info = tuple(map(int, _mac_version.split('.')))
if _mac_version_info < (10, 8):
    raise ImportError(f'Only OS X 10.8 and newer are supported, not {_mac_version_info[0]}.{_mac_version_info[1]}')

def _load_cdll(name: str, macos10_16_path: str) -> CDLL:
    if False:
        while True:
            i = 10
    'Loads a CDLL by name, falling back to known path on 10.16+'
    try:
        path: str | None
        if _mac_version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError
        return CDLL(path, use_errno=True)
    except OSError:
        raise ImportError(f'The library {name} failed to load') from None
Security = _load_cdll('Security', '/System/Library/Frameworks/Security.framework/Security')
CoreFoundation = _load_cdll('CoreFoundation', '/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
Boolean = c_bool
CFIndex = c_long
CFStringEncoding = c_uint32
CFData = c_void_p
CFString = c_void_p
CFArray = c_void_p
CFMutableArray = c_void_p
CFError = c_void_p
CFType = c_void_p
CFTypeID = c_ulong
CFTypeRef = POINTER(CFType)
CFAllocatorRef = c_void_p
OSStatus = c_int32
CFErrorRef = POINTER(CFError)
CFDataRef = POINTER(CFData)
CFStringRef = POINTER(CFString)
CFArrayRef = POINTER(CFArray)
CFMutableArrayRef = POINTER(CFMutableArray)
CFArrayCallBacks = c_void_p
CFOptionFlags = c_uint32
SecCertificateRef = POINTER(c_void_p)
SecPolicyRef = POINTER(c_void_p)
SecTrustRef = POINTER(c_void_p)
SecTrustResultType = c_uint32
SecTrustOptionFlags = c_uint32
try:
    Security.SecCertificateCreateWithData.argtypes = [CFAllocatorRef, CFDataRef]
    Security.SecCertificateCreateWithData.restype = SecCertificateRef
    Security.SecCertificateCopyData.argtypes = [SecCertificateRef]
    Security.SecCertificateCopyData.restype = CFDataRef
    Security.SecCopyErrorMessageString.argtypes = [OSStatus, c_void_p]
    Security.SecCopyErrorMessageString.restype = CFStringRef
    Security.SecTrustSetAnchorCertificates.argtypes = [SecTrustRef, CFArrayRef]
    Security.SecTrustSetAnchorCertificates.restype = OSStatus
    Security.SecTrustSetAnchorCertificatesOnly.argtypes = [SecTrustRef, Boolean]
    Security.SecTrustSetAnchorCertificatesOnly.restype = OSStatus
    Security.SecTrustEvaluate.argtypes = [SecTrustRef, POINTER(SecTrustResultType)]
    Security.SecTrustEvaluate.restype = OSStatus
    Security.SecPolicyCreateRevocation.argtypes = [CFOptionFlags]
    Security.SecPolicyCreateRevocation.restype = SecPolicyRef
    Security.SecPolicyCreateSSL.argtypes = [Boolean, CFStringRef]
    Security.SecPolicyCreateSSL.restype = SecPolicyRef
    Security.SecTrustCreateWithCertificates.argtypes = [CFTypeRef, CFTypeRef, POINTER(SecTrustRef)]
    Security.SecTrustCreateWithCertificates.restype = OSStatus
    Security.SecTrustGetTrustResult.argtypes = [SecTrustRef, POINTER(SecTrustResultType)]
    Security.SecTrustGetTrustResult.restype = OSStatus
    Security.SecTrustRef = SecTrustRef
    Security.SecTrustResultType = SecTrustResultType
    Security.OSStatus = OSStatus
    kSecRevocationUseAnyAvailableMethod = 3
    kSecRevocationRequirePositiveResponse = 8
    CoreFoundation.CFRelease.argtypes = [CFTypeRef]
    CoreFoundation.CFRelease.restype = None
    CoreFoundation.CFGetTypeID.argtypes = [CFTypeRef]
    CoreFoundation.CFGetTypeID.restype = CFTypeID
    CoreFoundation.CFStringCreateWithCString.argtypes = [CFAllocatorRef, c_char_p, CFStringEncoding]
    CoreFoundation.CFStringCreateWithCString.restype = CFStringRef
    CoreFoundation.CFStringGetCStringPtr.argtypes = [CFStringRef, CFStringEncoding]
    CoreFoundation.CFStringGetCStringPtr.restype = c_char_p
    CoreFoundation.CFStringGetCString.argtypes = [CFStringRef, c_char_p, CFIndex, CFStringEncoding]
    CoreFoundation.CFStringGetCString.restype = c_bool
    CoreFoundation.CFDataCreate.argtypes = [CFAllocatorRef, c_char_p, CFIndex]
    CoreFoundation.CFDataCreate.restype = CFDataRef
    CoreFoundation.CFDataGetLength.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetLength.restype = CFIndex
    CoreFoundation.CFDataGetBytePtr.argtypes = [CFDataRef]
    CoreFoundation.CFDataGetBytePtr.restype = c_void_p
    CoreFoundation.CFArrayCreate.argtypes = [CFAllocatorRef, POINTER(CFTypeRef), CFIndex, CFArrayCallBacks]
    CoreFoundation.CFArrayCreate.restype = CFArrayRef
    CoreFoundation.CFArrayCreateMutable.argtypes = [CFAllocatorRef, CFIndex, CFArrayCallBacks]
    CoreFoundation.CFArrayCreateMutable.restype = CFMutableArrayRef
    CoreFoundation.CFArrayAppendValue.argtypes = [CFMutableArrayRef, c_void_p]
    CoreFoundation.CFArrayAppendValue.restype = None
    CoreFoundation.CFArrayGetCount.argtypes = [CFArrayRef]
    CoreFoundation.CFArrayGetCount.restype = CFIndex
    CoreFoundation.CFArrayGetValueAtIndex.argtypes = [CFArrayRef, CFIndex]
    CoreFoundation.CFArrayGetValueAtIndex.restype = c_void_p
    CoreFoundation.CFErrorGetCode.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorGetCode.restype = CFIndex
    CoreFoundation.CFErrorCopyDescription.argtypes = [CFErrorRef]
    CoreFoundation.CFErrorCopyDescription.restype = CFStringRef
    CoreFoundation.kCFAllocatorDefault = CFAllocatorRef.in_dll(CoreFoundation, 'kCFAllocatorDefault')
    CoreFoundation.kCFTypeArrayCallBacks = c_void_p.in_dll(CoreFoundation, 'kCFTypeArrayCallBacks')
    CoreFoundation.CFTypeRef = CFTypeRef
    CoreFoundation.CFArrayRef = CFArrayRef
    CoreFoundation.CFStringRef = CFStringRef
    CoreFoundation.CFErrorRef = CFErrorRef
except AttributeError:
    raise ImportError('Error initializing ctypes') from None

def _handle_osstatus(result: OSStatus, _: typing.Any, args: typing.Any) -> typing.Any:
    if False:
        while True:
            i = 10
    '\n    Raises an error if the OSStatus value is non-zero.\n    '
    if int(result) == 0:
        return args
    error_message_cfstring = None
    try:
        error_message_cfstring = Security.SecCopyErrorMessageString(result, None)
        error_message_cfstring_c_void_p = ctypes.cast(error_message_cfstring, ctypes.POINTER(ctypes.c_void_p))
        message = CoreFoundation.CFStringGetCStringPtr(error_message_cfstring_c_void_p, CFConst.kCFStringEncodingUTF8)
        if message is None:
            buffer = ctypes.create_string_buffer(1024)
            result = CoreFoundation.CFStringGetCString(error_message_cfstring_c_void_p, buffer, 1024, CFConst.kCFStringEncodingUTF8)
            if not result:
                raise OSError('Error copying C string from CFStringRef')
            message = buffer.value
    finally:
        if error_message_cfstring is not None:
            CoreFoundation.CFRelease(error_message_cfstring)
    if message is None or message == '':
        message = f'SecureTransport operation returned a non-zero OSStatus: {result}'
    raise ssl.SSLError(message)
Security.SecTrustCreateWithCertificates.errcheck = _handle_osstatus
Security.SecTrustSetAnchorCertificates.errcheck = _handle_osstatus
Security.SecTrustGetTrustResult.errcheck = _handle_osstatus

class CFConst:
    """CoreFoundation constants"""
    kCFStringEncodingUTF8 = CFStringEncoding(134217984)
    errSecIncompleteCertRevocationCheck = -67635
    errSecHostNameMismatch = -67602
    errSecCertificateExpired = -67818
    errSecNotTrusted = -67843

def _bytes_to_cf_data_ref(value: bytes) -> CFDataRef:
    if False:
        while True:
            i = 10
    return CoreFoundation.CFDataCreate(CoreFoundation.kCFAllocatorDefault, value, len(value))

def _bytes_to_cf_string(value: bytes) -> CFString:
    if False:
        return 10
    '\n    Given a Python binary data, create a CFString.\n    The string must be CFReleased by the caller.\n    '
    c_str = ctypes.c_char_p(value)
    cf_str = CoreFoundation.CFStringCreateWithCString(CoreFoundation.kCFAllocatorDefault, c_str, CFConst.kCFStringEncodingUTF8)
    return cf_str

def _cf_string_ref_to_str(cf_string_ref: CFStringRef) -> str | None:
    if False:
        print('Hello World!')
    '\n    Creates a Unicode string from a CFString object. Used entirely for error\n    reporting.\n    Yes, it annoys me quite a lot that this function is this complex.\n    '
    string = CoreFoundation.CFStringGetCStringPtr(cf_string_ref, CFConst.kCFStringEncodingUTF8)
    if string is None:
        buffer = ctypes.create_string_buffer(1024)
        result = CoreFoundation.CFStringGetCString(cf_string_ref, buffer, 1024, CFConst.kCFStringEncodingUTF8)
        if not result:
            raise OSError('Error copying C string from CFStringRef')
        string = buffer.value
    if string is not None:
        string = string.decode('utf-8')
    return string

def _der_certs_to_cf_cert_array(certs: list[bytes]) -> CFMutableArrayRef:
    if False:
        return 10
    'Builds a CFArray of SecCertificateRefs from a list of DER-encoded certificates.\n    Responsibility of the caller to call CoreFoundation.CFRelease on the CFArray.\n    '
    cf_array = CoreFoundation.CFArrayCreateMutable(CoreFoundation.kCFAllocatorDefault, 0, ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks))
    if not cf_array:
        raise MemoryError('Unable to allocate memory!')
    for cert_data in certs:
        cf_data = None
        sec_cert_ref = None
        try:
            cf_data = _bytes_to_cf_data_ref(cert_data)
            sec_cert_ref = Security.SecCertificateCreateWithData(CoreFoundation.kCFAllocatorDefault, cf_data)
            CoreFoundation.CFArrayAppendValue(cf_array, sec_cert_ref)
        finally:
            if cf_data:
                CoreFoundation.CFRelease(cf_data)
            if sec_cert_ref:
                CoreFoundation.CFRelease(sec_cert_ref)
    return cf_array

@contextlib.contextmanager
def _configure_context(ctx: ssl.SSLContext) -> typing.Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    check_hostname = ctx.check_hostname
    verify_mode = ctx.verify_mode
    ctx.check_hostname = False
    _set_ssl_context_verify_mode(ctx, ssl.CERT_NONE)
    try:
        yield
    finally:
        ctx.check_hostname = check_hostname
        _set_ssl_context_verify_mode(ctx, verify_mode)

def _verify_peercerts_impl(ssl_context: ssl.SSLContext, cert_chain: list[bytes], server_hostname: str | None=None) -> None:
    if False:
        return 10
    certs = None
    policies = None
    trust = None
    cf_error = None
    try:
        if server_hostname is not None:
            cf_str_hostname = None
            try:
                cf_str_hostname = _bytes_to_cf_string(server_hostname.encode('ascii'))
                ssl_policy = Security.SecPolicyCreateSSL(True, cf_str_hostname)
            finally:
                if cf_str_hostname:
                    CoreFoundation.CFRelease(cf_str_hostname)
        else:
            ssl_policy = Security.SecPolicyCreateSSL(True, None)
        policies = ssl_policy
        if ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_CHAIN:
            policies = CoreFoundation.CFArrayCreateMutable(CoreFoundation.kCFAllocatorDefault, 0, ctypes.byref(CoreFoundation.kCFTypeArrayCallBacks))
            CoreFoundation.CFArrayAppendValue(policies, ssl_policy)
            CoreFoundation.CFRelease(ssl_policy)
            revocation_policy = Security.SecPolicyCreateRevocation(kSecRevocationUseAnyAvailableMethod | kSecRevocationRequirePositiveResponse)
            CoreFoundation.CFArrayAppendValue(policies, revocation_policy)
            CoreFoundation.CFRelease(revocation_policy)
        elif ssl_context.verify_flags & ssl.VERIFY_CRL_CHECK_LEAF:
            raise NotImplementedError('VERIFY_CRL_CHECK_LEAF not implemented for macOS')
        certs = None
        try:
            certs = _der_certs_to_cf_cert_array(cert_chain)
            trust = Security.SecTrustRef()
            Security.SecTrustCreateWithCertificates(certs, policies, ctypes.byref(trust))
        finally:
            if certs:
                CoreFoundation.CFRelease(certs)
        ctx_ca_certs_der: list[bytes] | None = ssl_context.get_ca_certs(binary_form=True)
        if ctx_ca_certs_der:
            ctx_ca_certs = None
            try:
                ctx_ca_certs = _der_certs_to_cf_cert_array(cert_chain)
                Security.SecTrustSetAnchorCertificates(trust, ctx_ca_certs)
            finally:
                if ctx_ca_certs:
                    CoreFoundation.CFRelease(ctx_ca_certs)
        else:
            Security.SecTrustSetAnchorCertificates(trust, None)
        cf_error = CoreFoundation.CFErrorRef()
        sec_trust_eval_result = Security.SecTrustEvaluateWithError(trust, ctypes.byref(cf_error))
        if sec_trust_eval_result == 1:
            is_trusted = True
        elif sec_trust_eval_result == 0:
            is_trusted = False
        else:
            raise ssl.SSLError(f'Unknown result from Security.SecTrustEvaluateWithError: {sec_trust_eval_result!r}')
        cf_error_code = 0
        if not is_trusted:
            cf_error_code = CoreFoundation.CFErrorGetCode(cf_error)
            if ssl_context.verify_mode != ssl.CERT_REQUIRED and (cf_error_code == CFConst.errSecNotTrusted or cf_error_code == CFConst.errSecCertificateExpired):
                is_trusted = True
            elif not ssl_context.check_hostname and cf_error_code == CFConst.errSecHostNameMismatch:
                is_trusted = True
        if not is_trusted:
            cf_error_string_ref = None
            try:
                cf_error_string_ref = CoreFoundation.CFErrorCopyDescription(cf_error)
                cf_error_message = _cf_string_ref_to_str(cf_error_string_ref) or 'Certificate verification failed'
                sec_trust_result_type = Security.SecTrustResultType()
                Security.SecTrustGetTrustResult(trust, ctypes.byref(sec_trust_result_type))
                err = ssl.SSLCertVerificationError(cf_error_message)
                err.verify_message = cf_error_message
                err.verify_code = cf_error_code
                raise err
            finally:
                if cf_error_string_ref:
                    CoreFoundation.CFRelease(cf_error_string_ref)
    finally:
        if policies:
            CoreFoundation.CFRelease(policies)
        if trust:
            CoreFoundation.CFRelease(trust)