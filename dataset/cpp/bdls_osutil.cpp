// bdls_osutil.cpp                                                    -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <bdls_osutil.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(bdls_osutil_cpp, "$Id$ $CSID$")

#include <bdls_processutil.h>

#include <bslmf_assert.h>
#include <bsls_platform.h>

#include <bsl_cstring.h>

#ifdef BSLS_PLATFORM_OS_WINDOWS
# include <bdlsb_fixedmemoutstreambuf.h>

# include <bsl_ostream.h>

# include <windows.h>

# define   U_VISTA_OR_LATER 0
# if 6 <= BSLS_PLATFORM_OS_VER_MAJOR
#   undef  U_VISTA_OR_LATER
#   define U_VISTA_OR_LATER 1
# endif
# if 0 != U_VISTA_OR_LATER
#   include <bsl_limits.h>
#   include <VersionHelpers.h>
# else
#   include <process.h>
# endif
#else
# include <unistd.h>
# include <sys/utsname.h>
#endif

namespace BloombergLP {

namespace {

template <class STRING_TYPE>
int u_OsUtil_getOsInfo(STRING_TYPE *osName,
                       STRING_TYPE *osVersion,
                       STRING_TYPE *osPatch);
    // Load the operating system name, version name and patch name into the
    // specified 'osName', 'osVersion' and 'osPatch' respectively.  Return
    // 0 on success and a non-zero value otherwise.  The loaded values may
    // represent an emulation provided for the current process (see
    // "manifest-based behavior" in Windows programming documentation for an
    // example) and therefore are not suitable for determining supported
    // features or the real environment/version.  If you need to determine the
    // presence of certain features please consult the documentation of the
    // operating systems you need to support.

#ifdef BSLS_PLATFORM_OS_WINDOWS
template <class STRING_TYPE>
int u_OsUtil_getOsInfo(STRING_TYPE *osName,
                       STRING_TYPE *osVersion,
                       STRING_TYPE *osPatch)
{
    BSLS_ASSERT(osName);
    BSLS_ASSERT(osVersion);
    BSLS_ASSERT(osPatch);

    *osName = "Windows";

#if 0 != U_VISTA_OR_LATER
    // On Windows, 'WORD' means a 16-bit unsigned int.

    WORD major = 0;
    WORD minor = 0;
    WORD servicePackMajor = 0;

    const WORD maxWord = bsl::numeric_limits<WORD>::max();

    while (IsWindowsVersionOrGreater(major, minor, servicePackMajor)) {
        if (major >= maxWord) {
            return -1;                                                // RETURN
        }
        ++major;
    }
    --major;
    while (IsWindowsVersionOrGreater(major, minor, servicePackMajor)) {
        if (minor >= maxWord) {
            return -1;                                                // RETURN
        }
        ++minor;
    }
    --minor;
    while (IsWindowsVersionOrGreater(major, minor, servicePackMajor)) {
        if (servicePackMajor >= maxWord) {
            return -1;                                                // RETURN
        }
        ++servicePackMajor;
    }
    --servicePackMajor;

    // Os version

    // We want to do this with a minimum of allocations.  Both an
    // 'ostringstream' and 'sprintf' would allocate memory, so we us a
    // 'bdlsb::FixedMemOutStreamBuf"

    char buf[256];
    bdlsb::FixedMemOutStreamBuf sb(buf, sizeof(buf));
    bsl::ostream ostr(&sb);

    ostr << major << '.' << minor << bsl::ends;
    *osVersion = buf;

    // Service pack number

    sb.pubsetbuf(buf, sizeof(buf));
    buf[0] = 0;

    if (servicePackMajor) {
        // Note that we are incapable of detecting any 'servicePackMinor'
        // version other than 0.  But it seems rational that if Microsoft had
        // any plans for non-zero 'servicePackMinor' version at or after
        // Vista, they would have made 'IsWindowsVersionOrGreater' take 4 args
        // instead of 3.

        ostr << "Service Pack " << servicePackMajor << ".0" << bsl::ends;
    }

    *osPatch = buf;

#else // i.e., 0 == U_VISTA_OR_LATER

    OSVERSIONINFOEX osvi;

    bsl::memset(&osvi, 0, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    if (!GetVersionEx((OSVERSIONINFO *)&osvi)) {
        return -1;                                                    // RETURN
    }

    // Os version

    // We want to do this with a minimum of allocations.  Both an
    // 'ostringstream' and 'sprintf' would allocate memory, so we us a
    // 'bdlsb::FixedMemOutStreamBuf"

    char buf[256];
    bdlsb::FixedMemOutStreamBuf sb(buf, sizeof(buf));
    bsl::ostream ostr(&sb);

    ostr << osvi.dwMajorVersion << '.' << osvi.dwMinorVersion << bsl::ends;
    *osVersion = buf;

    sb.pubsetbuf(buf, sizeof(buf));
    buf[0] = 0;

    // Service pack number

    if (osvi.wServicePackMajor) {
        ostr << "Service Pack " << osvi.wServicePackMajor << '.'
             << osvi.wServicePackMinor << bsl::ends;
    }
    *osPatch = buf;

#endif // 0 != U_VISTA_OR_LATER

    return 0;
}

#elif defined(BSLS_PLATFORM_OS_UNIX)

template <class STRING_TYPE>
int u_OsUtil_getOsInfo(STRING_TYPE *osName,
                       STRING_TYPE *osVersion,
                       STRING_TYPE *osPatch)
{
    BSLS_ASSERT(osName);
    BSLS_ASSERT(osVersion);
    BSLS_ASSERT(osPatch);

    struct utsname unameInfo;
    if (-1 == uname(&unameInfo)) {
        return -1;                                                    // RETURN
    }
    *osName = unameInfo.sysname;
    *osVersion = unameInfo.release;
    *osPatch = unameInfo.version;

    return 0;
}

#else
#error "Unsupported operating system"
#endif // OS CHECK


} // close unnamed namespace

namespace bdls {

                            // --------------------
                            // struct bdls::OsUtil
                            // --------------------

// CLASS METHODS

int OsUtil::getOsInfo(bsl::string *osName,
                      bsl::string *osVersion,
                      bsl::string *osPatch)
{
    return u_OsUtil_getOsInfo(osName, osVersion, osPatch);
}

int OsUtil::getOsInfo(std::string *osName,
                      std::string *osVersion,
                      std::string *osPatch)
{
    return u_OsUtil_getOsInfo(osName, osVersion, osPatch);
}

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
int OsUtil::getOsInfo(std::pmr::string *osName,
                      std::pmr::string *osVersion,
                      std::pmr::string *osPatch)
{
    return u_OsUtil_getOsInfo(osName, osVersion, osPatch);
}
#endif

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
