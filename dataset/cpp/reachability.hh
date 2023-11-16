#pragma once

#if !defined ELLE_MACOS && !defined ELLE_IOS
# error "Unsupported platform"
#endif

#include <string>

#include <boost/signals2.hpp>

#include <elle/attribute.hh>

namespace elle
{
  namespace reactor
  {
    namespace network
    {
      /// macOS and iOS reachability callback.
      class Reachability
      {
      public:
        enum class NetworkStatus
        {
          Unreachable   = 0,
          Reachable     = 1,
# ifdef ELLE_IOS
          ReachableWWAN = 2, // Reachable using mobile data connection.
# endif
        };

        using StatusCallback = auto (NetworkStatus) -> void;

      public:
        /// Leaving the host empty tests zero address.
        Reachability(boost::optional<std::string> host = {},
                     std::function<StatusCallback> const& callback = {},
                     bool start = false);
        ~Reachability() = default;

        void
        start();
        void
        stop();

      public:
        bool
        reachable() const;
        NetworkStatus
        status() const;
        std::string
        status_string() const;

        ELLE_ATTRIBUTE_X(boost::signals2::signal<StatusCallback>, status_changed);

        class Impl;

      private:
        ELLE_ATTRIBUTE(std::unique_ptr<Impl>, impl);
      };
    }
  }
}
