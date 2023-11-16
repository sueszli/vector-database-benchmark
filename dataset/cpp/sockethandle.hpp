#ifndef EE_NETWORKSOCKETHANDLE_HPP
#define EE_NETWORKSOCKETHANDLE_HPP

#include <eepp/config.hpp>

#if EE_PLATFORM == EE_PLATFORM_WIN
#include <basetsd.h>
#endif

namespace EE { namespace Network {

/** Define the low-level socket handle type, specific to each platform */

#if EE_PLATFORM == EE_PLATFORM_WIN
typedef UINT_PTR SocketHandle;
#else
typedef int SocketHandle;
#endif

}} // namespace EE::Network

#endif // EE_NETWORKSOCKETHANDLE_HPP
