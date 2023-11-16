#include <boost/algorithm/string.hpp>

#include "ssf/layer/proxy/auth_strategy.h"

namespace ssf {
namespace layer {
namespace proxy {

std::string AuthStrategy::ExtractAuthToken(const HttpResponse& response) const {
  std::string challenge_str;
  auto auth_name = AuthName();

  auto hdr_values = response.GetHeaderValues(
      proxy_authentication() ? "Proxy-Authenticate" : "WWW-Authenticate");
  for (auto& hdr_value : hdr_values) {
    // find auth header
    if (hdr_value.find(auth_name) == 0) {
      challenge_str = hdr_value.substr(auth_name.length());
      break;
    }
  }

  if (challenge_str.empty()) {
    return "";
  }

  boost::trim(challenge_str);

  return challenge_str;
}

}  // proxy
}  // layer
}  // ssf