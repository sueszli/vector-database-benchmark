#pragma once

#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/optional.hpp>

#include <elle/Printable.hh>
#include <elle/attribute.hh>
#include <elle/serialization/fwd.hh>
#include <elle/service/aws/Keys.hh>

namespace elle
{
  namespace service
  {
    namespace aws
    {
      class Credentials
        : public elle::Printable
      {
          /*-------------.
          | Construction |
          `-------------*/
        public:
          Credentials() = default;
          /// Constructor for federated user.
          Credentials(std::string access_key_id,
                      std::string secret_access_key,
                      std::string session_token,
                      std::string region,
                      std::string bucket,
                      std::string folder,
                      Time expiration,
                      Time server_time,
                      boost::optional<std::string> endpoint = {});
          /// Constructor for normal user (i.e.: No session_token).
          Credentials(std::string access_key_id,
                      std::string secret_access_key,
                      std::string region,
                      std::string bucket,
                      std::string folder,
                      boost::optional<std::string> endpoint = {});
          std::string
          credential_string(RequestTime const& request_time,
                            Service const& aws_service) const;

          bool
          valid();

          ELLE_ATTRIBUTE_R(std::string, access_key_id);
          ELLE_ATTRIBUTE_R(std::string, secret_access_key);
          ELLE_ATTRIBUTE_R(boost::optional<std::string>, session_token);
          ELLE_ATTRIBUTE_R(std::string, region);
          ELLE_ATTRIBUTE_R(std::string, bucket);
          ELLE_ATTRIBUTE_R(std::string, folder);
          // Amazon current time from server
          ELLE_ATTRIBUTE_R(Time, server_time);
          ELLE_ATTRIBUTE_R(Time, expiry);
          // Estimated skew between trusted server time and local universal time.
          ELLE_ATTRIBUTE_RW(Duration, skew);
          ELLE_ATTRIBUTE_R(bool, federated_user);
          ELLE_ATTRIBUTE_R(boost::optional<std::string>, endpoint);

          /*--------------.
          | Serialization |
          `--------------*/
        public:
          Credentials(elle::serialization::SerializerIn& s);
          void
          serialize(elle::serialization::Serializer& s);

          /*----------.
          | Printable |
          `----------*/
        public:
          void
          print(std::ostream& stream) const;

        private:
          void
          _initialize(); // compute skew and expiry from input data
      };
    }
  }
}
