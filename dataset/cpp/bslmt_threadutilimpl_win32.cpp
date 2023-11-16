// bslmt_threadutilimpl_win32.cpp                                     -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <bslmt_threadutilimpl_win32.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bslmt_threadutilimpl_win32_cpp,"$Id$ $CSID$")

#ifdef BSLMT_PLATFORM_WIN32_THREADS

#include <windows.h>

#include <bslmt_configuration.h>
#include <bslmt_threadattributes.h>

#include <bsls_systemclocktype.h>
#include <bsls_systemtime.h>

#include <bsl_cstring.h>  // 'memcpy'

#include <bsls_assert.h>
#include <bsls_bslonce.h>
#include <bsls_types.h>

#include <process.h>      // '_begintthreadex', '_endthreadex'
#include <windows.h>

#if defined(BSLS_PLATFORM_OS_WINDOWS) && defined(BSLS_PLATFORM_CPU_64_BIT)
    // On 64-bit Windows, we have to deal with the fact that Windows ThreadProc
    // thread procedures only return a 32-bit DWORD value.  We use an
    // intermediate map to store the actual 'void *' return or exit values, so
    // they can be retrieved by join.

    #define BSLMT_USE_RETURN_VALUE_MAP
    #include <bsl_unordered_map.h>

    #include <bslma_default.h>
#endif

namespace BloombergLP {

// CLASS DATA
const bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::Handle
bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::INVALID_HANDLE =
                                                   { INVALID_HANDLE_VALUE, 0 };

namespace {
namespace u {

class HandleGuard {
    // This guard mechanism closes the windows handle (using 'CloseHandle')
    // when this guard goes out of scope and is destroyed.

    // DATA
    HANDLE d_handle;

  private:
    // NOT IMPLEMENTED
    HandleGuard(const HandleGuard&);
    HandleGuard operator=(const HandleGuard&);

  public:

    explicit HandleGuard(HANDLE handle);
        // Create a guard for the specified 'handle', that upon going out of
        // scope and being destroyed, will call 'CloseHandle' on 'handle'.

    ~HandleGuard();
        // Call 'CloseHandle' on the windows handle supplied at construction.
};

HandleGuard::HandleGuard(HANDLE handle)
: d_handle(handle)
{
}

HandleGuard::~HandleGuard()
{
    if (!CloseHandle(d_handle)) {
        BSLS_ASSERT_OPT(false);
    }
}

struct ThreadStartupInfo {
    // Control structure used to pass startup information to the thread entry
    // function.

    bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::Handle  d_handle;
    bslmt_ThreadFunction                                          d_function;
    void                                                         *d_threadArg;
    ThreadStartupInfo                                            *d_next;
};

struct ThreadSpecificDestructor {
    // This structure implements a linked list of destructors associated with
    // thread-specific key.

    bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::Key  d_key;
    bslmt_KeyDestructorFunction                                d_destructor;
    ThreadSpecificDestructor                                  *d_next;
};

                                // =============
                                // ThreadNameAPI
                                // =============

class ThreadNameAPI {
    // This 'class' manages dynamic loading of the 'Kernel32.dll' and access to
    // the functions for setting and getting thread names in that dll.  Note
    // that prior to Windows 10 version 1067, these functions are not
    // available, in which case the thread name will just appear to be the
    // empty string.

    // PRIVATE TYPES
    typedef HRESULT (WINAPI *GetThreadDescriptionType)(HANDLE, PWSTR *);
    typedef HRESULT (WINAPI* SetThreadDescriptionType)(HANDLE, PCWSTR);

    // DATA
    GetThreadDescriptionType d_gtdFuncPtr;
    SetThreadDescriptionType d_stdFuncPtr;

    // PRIVATE CREATOR
    ThreadNameAPI();
        // Dynamically load the dll and set the two function pointers in this

  public:
    // CLASS METHOD
    static ThreadNameAPI& singleton();
        // Return a reference to the thread name API singleton, creating it if
        // necessary.

    // ACCESSORS
    void getThreadName(bsl::string *threadName) const;
        // Set the specified 'threadName' to the thread name.

    void setThreadName(const bslstl::StringRef& threadName) const;
        // Set the thread name to the specified 'threadName'.
};

                                // -------------
                                // ThreadNameAPI
                                // -------------

ThreadNameAPI::ThreadNameAPI()
{
    HMODULE handle = ::GetModuleHandleA("Kernel32.dll");
    if (handle) {
        d_gtdFuncPtr = reinterpret_cast<GetThreadDescriptionType>(
                             ::GetProcAddress(handle, "GetThreadDescription"));
        d_stdFuncPtr = reinterpret_cast<SetThreadDescriptionType>(
                             ::GetProcAddress(handle, "SetThreadDescription"));
    }
}

ThreadNameAPI& ThreadNameAPI::singleton()
{
    static ThreadNameAPI *object_p = 0;
    static bsls::BslOnce  once = BSLS_BSLONCE_INITIALIZER;
    bsls::BslOnceGuard    onceGuard;
    if (onceGuard.enter(&once)) {
        static ThreadNameAPI object;
        object_p = &object;
    }
    return *object_p;
}

void ThreadNameAPI::getThreadName(bsl::string *threadName) const
{
    BSLS_ASSERT(threadName);

    threadName->clear();

    if (d_gtdFuncPtr) {
        wchar_t *utf16Result = 0;
        HRESULT rc = (*d_gtdFuncPtr)(::GetCurrentThread(), &utf16Result);

        // 'GetThreadDescription' returned us a UTF-16 string, while our caller
        // wants a 'bsl::string'.  The BDE UTF-16 -> UTF-8 translation
        // compoonent is in 'bdlde', so we use Windows provided APIs for
        // character conversion

        if (SUCCEEDED(rc) && utf16Result) {
            // The 0th pass through this loop will just set 'rc' to the number
            // of bytes needed to store the result without outputting any bytes
            // of UTF-8.  The 1th pass will actually output to the buffer.

            // Because 'utf16Result' is null-terminated, the length calculated
            // will include room for a terminating '\0' which will be output
            // and which we will have to trim afterward.

            int lenPlusOne = 0;
            for (int pass = 0; pass < 2; ++pass) {
                int rc = ::WideCharToMultiByte(CP_UTF8,
                                               0,
                                               utf16Result,
                                               -1,
                                               pass ? &(*threadName)[0] : NULL,
                                               pass ? lenPlusOne : 0,
                                               NULL,
                                               NULL);
                if (0 == rc) {
                    // Translation failed.

                    return;                                           // RETURN
                }

                if (0 == pass) {
                    lenPlusOne = rc;
                    BSLS_ASSERT(0 < lenPlusOne);
                    threadName->resize(lenPlusOne);
                }
                else {
                    BSLS_ASSERT(lenPlusOne == rc);
                }
            }
            BSLS_ASSERT('\0' == (*threadName)[lenPlusOne - 1]);
            threadName->resize(lenPlusOne - 1);

            LocalFree(utf16Result);
        }
    }
}

void ThreadNameAPI::setThreadName(const bslstl::StringRef& threadName) const
{
    if (d_stdFuncPtr) {
        bsl::wstring utf16String;

        const int threadNameLen = static_cast<int>(threadName.length());
        int len = 0;
        for (int pass = 0; pass < 2; ++pass) {
            // The 0th pass through this loop will just set 'rc' to the number
            // of bytes needed to store the result without outputting any bytes
            // of UTF-8.  The 1th pass will actually output to the buffer.

            int rc = ::MultiByteToWideChar(CP_UTF8,
                                           0,
                                           threadName.data(),
                                           threadNameLen,
                                           pass ? &utf16String[0] : NULL,
                                           pass ? len : 0);
            if (0 == rc) {
                // Translation failed.

                return;                                               // RETURN
            }
            if (0 == pass) {
                len = rc;
                BSLS_ASSERT(0 < len);
                utf16String.resize(len);
            }
            else {
                BSLS_ASSERT(len == rc);
            }
        }

        (*d_stdFuncPtr)(::GetCurrentThread(), utf16String.c_str());
    }
}

struct Win32Initializer {
    // This structure is used to initialize and de-initialize the BCE threading
    // environment.  At creation, 'bslmt_threadutil_win32_Initialize' is called
    // to initialize the environment.  When the object is destroyed, it calls
    // 'bslmt_threadutil_win32_Deinitialize' to cleanup the environment.

    // CREATORS
    Win32Initializer();
        // Initialize the BCE threading environment.

    ~Win32Initializer();
        // De-initialize the BCE threading environment.
};

enum InitializationState {
    e_INITIALIZED   =  0  // threading environment has been initialized
  , e_UNINITIALIZED =  1  // threading environment has not been initialized
  , e_INITIALIZING  =  2  // threading environment is currently initializing
  , e_DEINITIALIZED = -1  // threading environment has been de-initialized
};

static void *volatile            startupInfoCache = 0;
static DWORD                     threadInfoTLSIndex = 0;
static ThreadSpecificDestructor *destructors = 0;
static CRITICAL_SECTION          threadSpecificDestructorsListLock;
static volatile long             initializationState = e_UNINITIALIZED;

#ifdef BSLMT_USE_RETURN_VALUE_MAP
static CRITICAL_SECTION          returnValueMapLock;

// Access to this map will be serialized with 'returnValueMapLock'.  It must be
// declared before 'initializer' - the 'initializer' destructor will empty the
// map.  'returnValueMapValid' will be false if this module's static
// initialization has not run, or static cleanup has occurred.

static volatile bool                      returnValueMapValid = false;
typedef bsl::unordered_map<DWORD, void *> TReturnValueMap;
static TReturnValueMap                    returnValueMap(
                                            bslma::Default::globalAllocator());
#endif

static Win32Initializer   initializer;

static inline
int bslmt_threadutil_win32_Initialize()
    // This function is used to initialize the BCE threading environment If the
    // environment has already been initialized, it returns immediately with a
    // 0 result.  Otherwise if the environment is currently being initialized
    // from some other thread, then it waits until the environment is
    // initialized and returns.
{
    if (e_INITIALIZED == initializationState) {
        return 0;                                                     // RETURN
    }
    else {
        long result;
        do {
            result = InterlockedCompareExchange(&initializationState,
                                                e_INITIALIZING,
                                                e_UNINITIALIZED);
            if (e_INITIALIZING == result) {
                ::Sleep(0);
            }
            else {
                break;
            }
        } while (1);

        if (e_UNINITIALIZED == result) {
            threadInfoTLSIndex = TlsAlloc();
            InitializeCriticalSection(&threadSpecificDestructorsListLock);
#ifdef BSLMT_USE_RETURN_VALUE_MAP
            InitializeCriticalSection(&returnValueMapLock);
            returnValueMapValid = true;
#endif
            InterlockedExchange(&u::initializationState, e_INITIALIZED);
        }
        return e_INITIALIZED == u::initializationState ? 0 : 1;       // RETURN
    }
}

static void bslmt_threadutil_win32_Deinitialize()
    // This function de-initializes the BCE threading environment and releases
    // all resources allocated by the environment.  Note that once the
    // environment has been de-initialized, it cannot be re-initialized.  This
    // prevents static objects from inadvertently re-initializing
    // re-initializing the environment when they are destroyed.
{
    if (InterlockedExchange(&u::initializationState, e_DEINITIALIZED)
                                                            != e_INITIALIZED) {
        return;                                                       // RETURN
    }

    TlsFree(threadInfoTLSIndex);

    EnterCriticalSection(&threadSpecificDestructorsListLock);
    ThreadSpecificDestructor *d = destructors;
    destructors = 0;
    LeaveCriticalSection(&threadSpecificDestructorsListLock);
    DeleteCriticalSection(&threadSpecificDestructorsListLock);

#ifdef BSLMT_USE_RETURN_VALUE_MAP
    EnterCriticalSection(&returnValueMapLock);
    returnValueMap.erase(returnValueMap.begin(), returnValueMap.end());
    returnValueMapValid = false;
    LeaveCriticalSection(&returnValueMapLock);
    // NOT deleted, so static objects can spawn threads.
    // DeleteCriticalSection(&returnValueMapLock);
#endif

    while (d) {
        ThreadSpecificDestructor *t = d;
        d = d->d_next;
        delete t;
    }

    ThreadStartupInfo *head;
    head = (ThreadStartupInfo *)InterlockedExchangePointer(
                                                      &u::startupInfoCache, 0);
    while (head) {
        ThreadStartupInfo *t = head;
        head = head->d_next;
        delete t;
    }
}

Win32Initializer::Win32Initializer()
{
    bslmt_threadutil_win32_Initialize();
}

Win32Initializer::~Win32Initializer()
{
    bslmt_threadutil_win32_Deinitialize();
}

static ThreadStartupInfo *allocStartupInfo()
    // This function provides an efficient allocator for 'ThreadStartupInfo'
    // objects.  The implementation uses atomic operations to manage a pool
    // objects without any locking overhead.
{
    ThreadStartupInfo *head;
    head = (ThreadStartupInfo *)InterlockedCompareExchangePointer(
                                                   &u::startupInfoCache, 0, 0);
    while (head) {
        void *t;
        t = InterlockedCompareExchangePointer(&u::startupInfoCache,
                                              head->d_next, head);
        if (t == head) {
            break;
        }
        else {
            head = (ThreadStartupInfo *)t;
        }
    }
    if (!head) {
        head = new ThreadStartupInfo;
    }
    return head;
}

static void freeStartupInfo(ThreadStartupInfo *item)
    // This function provides an efficient deallocator for 'ThreadStartupInfo'
    // objects.  The implementation uses atomic operations to manage a pool
    // objects without any locking overhead.
{
    item->d_next = 0;

    while (1) {
        ThreadStartupInfo *t;
        t = (ThreadStartupInfo*)InterlockedCompareExchangePointer(
                                                          &u::startupInfoCache,
                                                          item,
                                                          item->d_next);
        if (t == item->d_next) {
            break;
        }
        item->d_next = t;
    }
}

static void invokeDestructors()
    // This function is called when a BCE thread terminates to clean up any
    // thread specific data that have associated destructor functions.  It
    // iterates through all registered destructor functions and invokes each
    // destructor that has a non-zero key value.
{
    if (initializationState != e_INITIALIZED) {
        return;                                                       // RETURN
    }

    EnterCriticalSection(&threadSpecificDestructorsListLock);
    ThreadSpecificDestructor *d = destructors;
    while (d) {
        void *value = TlsGetValue(d->d_key);
        if (value) {
            d->d_destructor(value);
        }
        d = d->d_next;
    }
    LeaveCriticalSection(&threadSpecificDestructorsListLock);
}

static unsigned _stdcall ThreadEntry(void *arg)
    // This function is the entry point for all BCE thread functions.
{
    ThreadStartupInfo startInfo;

    bsl::memcpy(&startInfo, arg, sizeof(startInfo));

    freeStartupInfo((ThreadStartupInfo *)arg);
    TlsSetValue(threadInfoTLSIndex, &startInfo.d_handle);
    void *ret = startInfo.d_function(startInfo.d_threadArg);
    invokeDestructors();
#ifdef BSLMT_USE_RETURN_VALUE_MAP
    EnterCriticalSection(&returnValueMapLock);
    if (returnValueMapValid) {
        returnValueMap[startInfo.d_handle.d_id] = ret;
    }
    LeaveCriticalSection(&returnValueMapLock);
#endif
    return (unsigned)(bsls::Types::IntPtr)ret;
}

}  // close namespace u
}  // close unnamed namespace

               // --------------------------------------------
               // class ThreadUtilImpl<Platform::Win32Threads>
               // --------------------------------------------

// CLASS METHODS
int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::create(
                                                Handle               *thread,
                                                bslmt_ThreadFunction  function,
                                                void                 *userData)
{
    ThreadAttributes attribute;
    return create(thread, attribute, function, userData);
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::create(
                                            Handle                  *handle,
                                            const ThreadAttributes&  attribute,
                                            bslmt_ThreadFunction     function,
                                            void                    *userData)
{
    if (u::bslmt_threadutil_win32_Initialize()) {
        return 1;                                                     // RETURN
    }

    u::ThreadStartupInfo *startInfo = u::allocStartupInfo();

    int stackSize = attribute.stackSize();
    if (ThreadAttributes::e_UNSET_STACK_SIZE == stackSize) {
        stackSize = Configuration::defaultThreadStackSize();
        if (ThreadAttributes::e_UNSET_STACK_SIZE == stackSize) {
            stackSize = Configuration::nativeDefaultThreadStackSize();
        }
    }

    BSLS_ASSERT_OPT(stackSize >= 0);    // 0 is a valid stack size to pass to
                                        // _beginthreadex -- it means
                                        // 'the same size as the main thread'.
                                        // This is not documented bde behavior,
                                        // but allow it just in case anyone was
                                        // depending on it.

    startInfo->d_threadArg = userData;
    startInfo->d_function  = function;
    handle->d_handle = (HANDLE)_beginthreadex(
                                             0,
                                             stackSize,
                                             u::ThreadEntry,
                                             startInfo,
                                             STACK_SIZE_PARAM_IS_A_RESERVATION,
                                             (unsigned int *)&handle->d_id);
    if ((HANDLE)0 == handle->d_handle) {
        u::freeStartupInfo(startInfo);
        return 1;                                                     // RETURN
    }
    if (ThreadAttributes::e_CREATE_DETACHED ==
                                                   attribute.detachedState()) {
        HANDLE tmpHandle = handle->d_handle;
        handle->d_handle  = 0;
        startInfo->d_handle = *handle;
        ResumeThread(tmpHandle);
        CloseHandle(tmpHandle);
    }
    else {
        startInfo->d_handle = *handle;
        ResumeThread(handle->d_handle);
    }
    return 0;
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::join(
                                                              Handle&   handle,
                                                              void    **status)
{
    // Cannot self - join

    if (!handle.d_handle || handle.d_id == GetCurrentThreadId()) {
        return 1;                                                     // RETURN
    }
    if (WaitForSingleObject(handle.d_handle, INFINITE) != WAIT_OBJECT_0) {
        return 2;                                                     // RETURN
    }
    DWORD exitStatus;
    DWORD result = GetExitCodeThread(handle.d_handle,&exitStatus);

#ifdef BSLMT_USE_RETURN_VALUE_MAP
    // In this case, we ignore 'exitStatus', but we're still fetching it in to
    // get the 'result' value

    if (status) {
        EnterCriticalSection(&u::returnValueMapLock);
        if (u::returnValueMapValid) {
            *status = u::returnValueMap[handle.d_id];
        }
        LeaveCriticalSection(&u::returnValueMapLock);
    }
#else
    if (status) {
        *status = (void *)exitStatus;
    }
#endif // def BSLMT_USE_RETURN_VALUE_MAP
    CloseHandle(handle.d_handle);
    handle.d_handle = 0;

    return FALSE == result ? 3 : 0;
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::detach(
                                                                Handle& handle)
{
    if (handle.d_handle == GetCurrentThread()
     && handle.d_id     == GetCurrentThreadId()) {
        Handle *realHandle = (Handle *)TlsGetValue(u::threadInfoTLSIndex);
        if (!realHandle || !realHandle->d_handle) {
            return 1;                                                 // RETURN
        }
        if (!CloseHandle(realHandle->d_handle)) {
            return 1;                                                 // RETURN
        }
        handle.d_handle = 0;
        return 0;                                                     // RETURN
    }
    else if (handle.d_handle) {
        if (!CloseHandle(handle.d_handle)) {
            return 1;                                                 // RETURN
        }
        handle.d_handle = 0;
        return 0;                                                     // RETURN
    }
    return 2;
}

void bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::exit(void *status)
{
    u::invokeDestructors();
#ifdef BSLMT_USE_RETURN_VALUE_MAP
    EnterCriticalSection(&u::returnValueMapLock);
    if (u::returnValueMapValid) {
        u::returnValueMap[GetCurrentThreadId()] = status;
    }
    LeaveCriticalSection(&u::returnValueMapLock);
#endif
    _endthreadex((unsigned)(bsls::Types::IntPtr)status);
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::createKey(
                                       Key                         *key,
                                       bslmt_KeyDestructorFunction  destructor)
{
    // It is not uncommon for applications to have global objects that make
    // calls to create thread-specific keys.  It is possible that those objects
    // are initialized before the BCE threading environment, so make sure to do
    // so first.

    if (u::bslmt_threadutil_win32_Initialize()) {
        return 1;                                                     // RETURN
    }

    *key = TlsAlloc();
    if (TLS_OUT_OF_INDEXES == *key) {
        return 1;                                                     // RETURN
    }
    if (destructor) {
        u::ThreadSpecificDestructor *d = new u::ThreadSpecificDestructor;
        d->d_key = *key;
        d->d_destructor = destructor;
        EnterCriticalSection(&u::threadSpecificDestructorsListLock);
        d->d_next = u::destructors;
        u::destructors = d;
        LeaveCriticalSection(&u::threadSpecificDestructorsListLock);
    }
    return 0;
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::deleteKey(Key& key)
{
    u::ThreadSpecificDestructor *prev = 0;
    if (!TlsFree(key)) {
        return 1;                                                     // RETURN
    }
    EnterCriticalSection(&u::threadSpecificDestructorsListLock);
    u::ThreadSpecificDestructor *d = u::destructors;
    while (d) {
        if (d->d_key == key) {
            if (prev) {
                prev->d_next = d->d_next;
            }
            else {
                u::destructors = d->d_next;
            }
            delete d;
            break;
        }
        prev = d;
        d = d->d_next;
    }
    LeaveCriticalSection(&u::threadSpecificDestructorsListLock);
    return 0;
}

bool bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::areEqual(
                                                               const Handle& a,
                                                               const Handle& b)
{
    return a.d_id == b.d_id;
}

void bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::getThreadName(
                                                       bsl::string *threadName)
{
    u::ThreadNameAPI::singleton().getThreadName(threadName);
}

void bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::setThreadName(
                                           const bslstl::StringRef& threadName)
{
    u::ThreadNameAPI::singleton().setThreadName(threadName);
}

int bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::sleepUntil(
                                      const bsls::TimeInterval&   absoluteTime,
                                      bsls::SystemClockType::Enum clockType)
{
    // ASSERT that the interval is between January 1, 1970 00:00.000 and the
    // end of December 31, 9999 (i.e., less than January 1, 10000).

    BSLS_ASSERT(absoluteTime >= bsls::TimeInterval(0, 0));
    BSLS_ASSERT(absoluteTime <  bsls::TimeInterval(253402300800LL, 0));

    // This implementation is very sensitive to the 'clockType'.  For safety,
    // we will assert the value is one of the two currently expected values.

    BSLS_ASSERT(bsls::SystemClockType::e_REALTIME ==  clockType ||
                bsls::SystemClockType::e_MONOTONIC == clockType);

    if (clockType == bsls::SystemClockType::e_REALTIME) {

        HANDLE timer = CreateWaitableTimer(0, false, 0);
        if (0 == timer) {
            return GetLastError();                                    // RETURN
        }
        u::HandleGuard guard(timer);

        LARGE_INTEGER clockTime;

        // As indicated in the documentation for 'SetWaitableTimer':
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms686289 A
        // positive value represents an *absolute* time in increments of 100
        // nanoseconds.  Critically though, Microsoft's epoch is different for
        // epoch used by the C run-time (and BDE).  BDE uses January 1, 1970,
        // Microsoft uses January 1, 1601, see:
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms724186 The
        // following page on converting a 'time_t' to a 'FILETIME' shows the
        // constant, 116444736000000000 in 100ns (or 11643609600 seconds)
        // needed to convert between the two epochs:
        // http://msdn.microsoft.com/en-us/library/windows/desktop/ms724228

        enum { k_HUNDRED_NANOSECS_PER_SEC = 10 * 1000 * 1000 };  // 10 million
        clockTime.QuadPart = absoluteTime.seconds()
                                                   * k_HUNDRED_NANOSECS_PER_SEC
                           + absoluteTime.nanoseconds() / 100
                           + 116444736000000000LL;

        if (!SetWaitableTimer(timer, &clockTime , 0, 0, 0, 0)) {
            return GetLastError();                                    // RETURN
        }

        if (WAIT_OBJECT_0 != WaitForSingleObject(timer, INFINITE)) {
            return GetLastError();                                    // RETURN
        }
    }
    else { // monotonic clock
        // The windows system function 'WaitForSingleObject' (which is used
        // here to implement 'sleepUntil') is based on a real-time clock.  If a
        // client supplies a monotonic clock time, rather than convert that
        // monotonic time to a time relative to the real-time clock (which
        // would introduce errors), we instead determine a sleep interval
        // relative to the monotonic clock and 'sleep' for that period (this
        // may also introduce errors, but potentially fewer because the period
        // over which a change in the system clock will impact the result will
        // be smaller). This may wake up prior to the specified 'absoluteTime',
        // so we will loop over a decreasing 'relativeTime' interval to ensure
        // that we don't return until after 'absoluteTime' has passed.

        bsls::TimeInterval relativeTime =
                          absoluteTime - bsls::SystemTime::nowMonotonicClock();
        while (relativeTime > bsls::TimeInterval(0, 0)) {
            sleep(relativeTime);
            relativeTime = absoluteTime - bsls::SystemTime::nowMonotonicClock();
        }
    }

    return 0;
}

unsigned int
bslmt::ThreadUtilImpl<bslmt::Platform::Win32Threads>::hardwareConcurrency()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

}  // close enterprise namespace

#endif  // BSLMT_PLATFORM_WIN32_THREADS

// ----------------------------------------------------------------------------
// Copyright 2019 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------- END-OF-FILE ----------------------------------
