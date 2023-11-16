#include "ssf/layer/proxy/base64.h"
#include "ssf/layer/proxy/basic_auth_strategy.h"

namespace ssf {
namespace layer {
namespace proxy {

BasicAuthStrategy::BasicAuthStrategy(const HttpProxy& proxy_ctx)
    : AuthStrategy(proxy_ctx, Status::kAuthenticating),
      request_populated_(false) {}

std::string BasicAuthStrategy::AuthName() const { return "Basic"; }

bool BasicAuthStrategy::Support(const HttpResponse& response) const {
  return !request_populated_ && response.IsAuthenticationAllowed(AuthName());
}

void BasicAuthStrategy::ProcessResponse(const HttpResponse& response) {
  if (response.Success()) {
    status_ = Status::kAuthenticated;
    return;
  }

  if (!Support(response)) {
    status_ = Status::kAuthenticationFailure;
    return;
  }

  set_proxy_authentication(response);
}

void BasicAuthStrategy::PopulateRequest(HttpRequest* p_request) {
  std::stringstream ss_credentials, header_value;
  ss_credentials << proxy_ctx_.username << ":" << proxy_ctx_.password;

  header_value << AuthName() << " " << Base64::Encode(ss_credentials.str());

  p_request->AddHeader(
      proxy_authentication() ? "Proxy-Authorization" : "Authorization",
      header_value.str());

  request_populated_ = true;
}

}  // proxy
}  // layer
}  // ssf