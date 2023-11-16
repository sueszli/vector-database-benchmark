// bdls_processutil.cpp                                               -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <bdls_processutil.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdls_processutil_cpp,"$Id$ $CSID$")

#include <bdls_filesystemutil.h>
#include <bdls_pathutil.h>

#include <bdlsb_fixedmemoutstreambuf.h>

#include <bslma_allocator.h>
#include <bslmf_assert.h>
#include <bsls_assert.h>
#include <bsls_atomicoperations.h>
#include <bsls_log.h>
#include <bsls_platform.h>

#include <bsl_algorithm.h>
#include <bsl_cstdio.h>
#include <bsl_cstdlib.h>
#include <bsl_iostream.h>

#include <sys/types.h>        // must precede 'stat.h'
#include <sys/stat.h>         // 'stat'

#if defined BSLS_PLATFORM_OS_UNIX
# include <limits.h>           // 'PATH_MAX'
# include <unistd.h>           // 'getpid'

# if defined BSLS_PLATFORM_OS_AIX
#   include <bslmf_assert.h>
#   include <sys/procfs.h>
#   include <procinfo.h>
# elif defined BSLS_PLATFORM_OS_DARWIN
#   include <libproc.h>
# elif defined BSLS_PLATFORM_OS_LINUX
#   include <bslma_deallocatorguard.h>
#   include <bsls_types.h>
#   include <errno.h>            // ::program_invocation_name
# elif defined BSLS_PLATFORM_OS_SOLARIS
#   include <bsl_fstream.h>
# else
#   error Unrecognized Platform
# endif
#else    // if defined BSLS_PLATFORM_OS_WINDOWS
# include <bdlde_charconvertutf16.h>
# include <windows.h>
#endif

namespace {
namespace u {

using namespace BloombergLP;

// This macro is equivalent to calling 'BSLS_LOG_ERROR(...)', except that it
// uses a static atomic int to only emit one trace per process.

#define U_LOG_ERROR_ONCE(...)                                                 \
    do {                                                                      \
        static bsls::AtomicOperations::AtomicTypes::Int done = { 0 };         \
        if (0 == bsls::AtomicOperations::testAndSwapInt(&done, 0, 1)) {       \
            if (bsls::LogSeverity::e_ERROR <=                                 \
                                            bsls::Log::severityThreshold()) { \
                bsls::Log::logFormattedMessage(bsls::LogSeverity::e_ERROR,    \
                                               __FILE__,                      \
                                               __LINE__,                      \
                                               __VA_ARGS__);                  \
            }                                                                 \
        }                                                                     \
    } while (false)

static
const char *doesProcExist()
    // Return a ptr to a string describing whether "/proc" exists or not.
{
#ifdef BSLS_PLATFORM_OS_WINDOWS
    return "";
#else
    return bdls::FilesystemUtil::isDirectory("/proc", true)
                                                 ? "\"/proc\" exists."
                                                 : "\"/proc\" does not exist.";
#endif
}

static inline
int getPid()
    // Return the process id.  Having this be separate from 'Obj::getProcessId'
    // allows us to call it inline within the component.
{
#ifdef BSLS_PLATFORM_OS_WINDOWS
    return static_cast<int>(GetCurrentProcessId());
#else
    return static_cast<int>(::getpid());
#endif
}

static inline
bool isExecutable(const char *path)
    // On Unix, return 'true' if the file at the specified 'path' exists, is
    // not a directory, and is executable (or is a symbolic link to such a
    // file) and 'false' otherwise.  On Windows, return 'true' if the file
    // exists and is not a directory.
{
#if defined BSLS_PLATFORM_OS_UNIX
    const int executableBits = S_IXUSR | S_IXGRP | S_IXOTH;

# if defined(BSLS_PLATFORM_OS_CYGWIN) || \
    (defined(BSLS_PLATFORM_OS_DARWIN) && defined(_DARWIN_FEATURE_64_BIT_INODE))
    struct stat s;
    int rc = ::stat(path, &s);
# else
    struct stat64 s;
    int rc = ::stat64(path, &s);
# endif
#else    // Windows
    const int executableBits = S_IEXEC;

    struct _stati64 s;
    int rc = ::_stati64(path, &s);
#endif

    return 0 == rc && !(s.st_mode & S_IFDIR) && (s.st_mode & executableBits);
}

static inline
bool isExecutable(const bsl::string& path)
{
    return isExecutable(path.c_str());
}


#if defined(BSLS_PLATFORM_OS_LINUX)   || \
    defined(BSLS_PLATFORM_OS_AIX)     || \
    defined(BSLS_PLATFORM_OS_SOLARIS)
static
void resolveSymLinksIfAny(bsl::string *fileName)
    // If the specified 'fileName' is a symlink, resolve it and overwrite
    // 'fileName' with the new value, but only if the new value refers to an
    // executable file.  The behavior is undefined unless '*fileName' is an
    // executable file or a symlink to one.  Note that this will only write
    // over 'fileName' if 'fileName' is a link to an executable file.
{
    BSLS_ASSERT_SAFE(isExecutable(*fileName));

    struct stat s;
    int rc = ::lstat(fileName->c_str(), &s);
    if (0 == rc && S_ISLNK(s.st_mode)) {
        // It's a symbolic link.  Resolve it.

#if defined PATH_MAX
        bsl::string linkString(PATH_MAX, '\0');
#else
        bsl::string linkString(4 * 1024, '\0');
#endif

        const char *pLinkString = ::realpath(fileName->c_str(),
                                             &linkString[0]);
        const bsl::size_t len = linkString.find('\0');
        if (0 != pLinkString && 0 < len && bsl::string::npos != len) {
            linkString.resize(len);

            if (isExecutable(linkString)) {
                fileName->assign(linkString);
            }
        }
    }
}
#endif

}  // close namespace u
}  // close unnamed namespace

namespace BloombergLP {
namespace bdls {

                             // ------------------
                             // struct ProcessUtil
                             // ------------------

// CLASS METHODS
int ProcessUtil::getProcessId()
{
    return u::getPid();
}

int ProcessUtil::getProcessName(bsl::string *result)
{
    BSLS_ASSERT(result);

    // The goal of 'getProcessName' is to return something like 'argv[0]', it
    // would be nice if it were a path to the executable, but failing that,
    // prefer to return something as similar to 'argv[0]' as possible.  If
    // 'argv[0]' is a symlink, prefer to return it unresolved.

#if defined BSLS_PLATFORM_OS_AIX

    enum { k_BUF_LEN = 8 * 1024 };

    bsl::string argsBuf(k_BUF_LEN, '\0');
    char *buf = &argsBuf[0];

    struct procentry64  procBuf;
    procBuf.pi_pid = u::getPid();

    // '::getargs' should fill the beginning of 'argsBuf' with null-terminated
    // 'argv[0]', which might be a relative path.

    if (0 != ::getargs(&procBuf, sizeof(procBuf), buf, k_BUF_LEN - 1)) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: ::getargs failed.  %s",
                                                           u::doesProcExist());

        return -1;                                                    // RETURN
    }

    // 'assign' here, not 'swap' since 'argsBuf' is irreversibly large and
    // probably uses a different allocator than '*result'.

    result->assign(argsBuf.c_str());
    return 0;

#elif defined BSLS_PLATFORM_OS_DARWIN

    bsl::string pidPathBuf(PROC_PIDPATHINFO_MAXSIZE, '\0');

    int numChars = ::proc_pidpath(u::getPid(),
                                  &pidPathBuf[0],
                                  PROC_PIDPATHINFO_MAXSIZE);
    if (numChars <= 0) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: ::proc_pidpath failed.  %s",
                                                           u::doesProcExist());

        return -1;                                                    // RETURN
    }
    BSLS_ASSERT(numChars <= PROC_PIDPATHINFO_MAXSIZE);

    result->assign(pidPathBuf.c_str(), numChars);
    return 0;

#elif defined BSLS_PLATFORM_OS_LINUX

    // We read '::program_invocation_name', which will just yield 'argv[0]',
    // which may or may not be a relative path.  Note that this will still
    // correctly handle the case where the path of the executable contains
    // spaces.

    const char *argv0 = ::program_invocation_name;
    if (!argv0 || !*argv0) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: ::program_invocation_name"
                                                                " failed.  %s",
                                                           u::doesProcExist());
        return -1;                                                    // RETURN
    }
    result->assign(argv0);
    return 0;

#elif defined BSLS_PLATFORM_OS_SOLARIS

    // "/proc/<pid>/cmdline' contains all the 'argv[*]' elements, separated by
    // '\0's.

    //: o If 'argv0' is obtained and is executable, use that.
    //: o Else, if 'execName' is non-empty and executable, use 'execName'.
    //: o Else, If 'execName' is non-empty and 'argv0' is empty, use
    //:   'execName'.
    //: o Else, if 'argv0' is not empty, use that.

    char cmdLineFileName[100];
    bdlsb::FixedMemOutStreamBuf sb(cmdLineFileName, sizeof(cmdLineFileName));
    bsl::ostream os(&sb);
    os << "/proc/" << u::getPid() << "/cmdline" << bsl::ends;
    bsl::ifstream cmdLineFile;
    cmdLineFile.open(cmdLineFileName,
                     bsl::ios_base::in | bsl::ios_base::binary);

    bsl::string argv0;
    if (!cmdLineFile.is_open()) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: Couldn't open %s.  %s",
                                          cmdLineFileName, u::doesProcExist());
    }
    else {
        bsl::getline(cmdLineFile, argv0, '\0');

        if (!argv0.empty() && u::isExecutable(argv0)) {
            result->assign(argv0);
            return 0;                                                 // RETURN
        }
    }

    // Either '/proc' is unavailable, or 'argv0' was a relative path and we've
    // changed the current directory since task startup.

    // '::getexecname' will return 'argv[0]' with symlinks resolved (which
    // might be a full path even though 'argv[0]' was relative), or 0 if it
    // fails.  No known limit on length.  Note this will still not result in
    // the path to the executable if 'argv[0]' was relative and the working
    // directory has changed since task startup.

    const char *execName = ::getexecname();
    if (!execName || !*execName) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: ::getexecname failed.  %s",
                                                           u::doesProcExist());
    }
    else if (argv0.empty() || u::isExecutable(execName)) {
        result->assign(execName);
        return 0;                                                     // RETURN
    }

    // Probably 'argv[0]' was relative and we've changed directories.  If we
    // have anything in 'argv0', return it as it may be useful as a process
    // name.

    if (!argv0.empty()) {
        result->assign(argv0);
        return 0;                                                     // RETURN
    }

    return -1;

#elif defined BSLS_PLATFORM_OS_WINDOWS

    // This code seems to be returning 'argv[0]' except that, on Windows,
    // 'argv[0]' is always an absolute path, even if it was specified on the
    // command line as a relative path.  The following code yields the full
    // path.  Also, on some Unix shells within windows, 'argv[0]' contains
    // '/'s instead of '\\'s and hence won't work with the file system, whereas
    // 'GetModuleFileName' below will return a usable path.

    bsl::wstring wResult(64 * 1024, '\0');
    DWORD length = GetModuleFileNameW(0, &wResult[0], wResult.length());
    if (!length) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: GetModuleFileName failed.");
    }
    else if (wResult.length() <= length) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil: GetModuleFileName buffer"
                                                                 " overflow.");
    }
    else {
        wResult.resize(length);

        // Use '*result's allocator for 'utf8Path', since 'utf8Path' will be
        // perfectly sized and we can then swap it into '*result'.

        bsl::string utf8Path(result->allocator().mechanism());
        int rc = bdlde::CharConvertUtf16::utf16ToUtf8(&utf8Path, wResult);
        if (0 != rc) {
            U_LOG_ERROR_ONCE("bdls::ProcessUtil: utf16ToUtf8 failed.");

            return -1;                                                // RETURN
        }

        result->swap(utf8Path);
        return 0;                                                     // RETURN
    }

    // Error

    return -1;
#else
# error    Unrecognized Platform
#endif
}

int ProcessUtil::getPathToExecutable(bsl::string *result)
{
    BSLS_ASSERT(result);

    int rc;

    // Try 'getProcessName' and see if it gives a path to a file.  This method
    // is prone to yielding relative paths, but if there's an executable file
    // at the yielded path, that is good enough.

    bsl::string processName;
    rc = getProcessName(&processName);
    if (0 != rc) {
        U_LOG_ERROR_ONCE("bdls::ProcessUtil::getPathToExecutable:"
                            " getProcessName failed.  %s", u::doesProcExist());
    }
    else if (u::isExecutable(processName)) {
        result->assign(processName);

        return 0;                                                     // RETURN
    }

    // Probably 'argv[0]' was a relative path and we've changed the current
    // directory since task startup.  Get the orignal directory from '$PWD' and
    // see if we can build the correct path to the executable with that.

    if (0 == rc && PathUtil::isRelative(processName)) {
        const char *taskStartUpDir = ::getenv("PWD");
        if (taskStartUpDir &&
                           FilesystemUtil::isDirectory(taskStartUpDir, true)) {
            bsl::string splicePath(taskStartUpDir);
            rc = PathUtil::appendIfValid(&splicePath, processName);
            if (0 == rc && u::isExecutable(splicePath)) {
                result->assign(splicePath);

                return 0;                                             // RETURN
            }
        }
    }

    // At this point we abandon trying to fix 'processName' and try a radically
    // different solution based on "/proc", which might or might not be
    // available.

#if defined(BSLS_PLATFORM_OS_LINUX)   || \
    defined(BSLS_PLATFORM_OS_AIX)     || \
    defined(BSLS_PLATFORM_OS_SOLARIS)

    char linkName[100];
    bdlsb::FixedMemOutStreamBuf sb(linkName, sizeof(linkName));
    bsl::ostream os(&sb);

#if defined BSLS_PLATFORM_OS_LINUX
    os << "/proc/" << u::getPid() << "/exe"          << bsl::ends;
#else
    os << "/proc/" << u::getPid() << "/object/a.out" << bsl::ends;
#endif

    if (!u::isExecutable(linkName)) {
        // Both 'processName' and 'linkName' are not executable.  Include
        // whether they even exist in the trace.

        const char *linkNameExists = FilesystemUtil::exists(linkName)
                                   ? "exists" : "doesn't exist";
        const char *procNameExists = FilesystemUtil::exists(processName)
                                   ? "exists" : "doesn't exist";

        U_LOG_ERROR_ONCE("bdls::ProcessUtil: '%s' not executable, %s."
                                                  " ProcessName: '%s' %s.  %s",
                                           linkName, linkNameExists,
                                           processName.c_str(), procNameExists,
                                                           u::doesProcExist());
    }
    else {
        // "/proc" was available (not always the case).
        //
        // 'linkName' may be a symlink or hardlink.  If it's a symlink, resolve
        // it, otherwise return 'linkName', which won't look anything like
        // 'argv[0]' but will at least allow the caller to access the
        // executable.

        *result = linkName;
        u::resolveSymLinksIfAny(result);

        return 0;                                                     // RETURN
    }

#endif

    return -1;
}

}  // close package namespace
}  // close enterprise namespace

// ----------------------------------------------------------------------------
// Copyright 2015 Bloomberg Finance L.P.
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
