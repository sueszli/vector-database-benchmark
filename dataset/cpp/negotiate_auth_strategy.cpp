#include "ssf/layer/proxy/base64.h"
#include "ssf/layer/proxy/negotiate_auth_strategy.h"
#include "ssf/log/log.h"

#if defined(WIN32)
#include "ssf/layer/proxy/windows/sspi_auth_impl.h"
#else
#include "ssf/layer/proxy/unix/gssapi_auth_impl.h"
#endif

namespace ssf {
namespace layer {
namespace proxy {

NegotiateAuthStrategy::NegotiateAuthStrategy(const HttpProxy& proxy_ctx)
    : AuthStrategy(proxy_ctx, Status::kAuthenticating) {
#if defined(WIN32)
  p_impl_.reset(new SSPIAuthImpl(SSPIAuthImpl::kNegotiate, proxy_ctx));
#else
  p_impl_.reset(new GSSAPIAuthImpl(proxy_ctx));
#endif
  if (p_impl_.get() != nullptr) {
    if (!p_impl_->Init()) {
      SSF_LOG("network_proxy", debug,
              "negotiate: could not initialize platform impl");
      status_ = Status::kAuthenticationFailure;
    }
  }
}

std::string NegotiateAuthStrategy::AuthName() const { return "Negotiate"; }

bool NegotiateAuthStrategy::Support(const HttpResponse& response) const {
  return p_impl_.get() != nullptr &&
         status_ != Status::kAuthenticationFailure &&
         response.IsAuthenticationAllowed(AuthName());
}

void NegotiateAuthStrategy::ProcessResponse(const HttpResponse& response) {
  if (response.Success()) {
    status_ = Status::kAuthenticated;
    return;
  }

  if (!Support(response) || p_impl_.get() == nullptr) {
    status_ = Status::kAuthenticationFailure;
    return;
  }

  set_proxy_authentication(response);

  auto server_token = Base64::Decode(ExtractAuthToken(response));

  if (!p_impl_->ProcessServerToken(server_token)) {
    SSF_LOG("network_proxy", debug,
            "negotiate: could not process server token");
    status_ = Status::kAuthenticationFailure;
    return;
  }
}

void NegotiateAuthStrategy::PopulateRequest(HttpRequest* p_request) {
  if (p_impl_.get() == nullptr) {
    status_ = Status::kAuthenticationFailure;
    return;
  }

  auto auth_token = p_impl_->GetAuthToken();
  if (auth_token.empty()) {
    SSF_LOG("network_proxy", debug, "negotiate: response token empty");
    status_ = Status::kAuthenticationFailure;
    return;
  }

  std::string negotiate_value = AuthName() + " " + Base64::Encode(auth_token);
  p_request->AddHeader(
      proxy_authentication() ? "Proxy-Authorization" : "Authorization",
      negotiate_value);
}

}  // proxy
}  // layer
}  // ssf