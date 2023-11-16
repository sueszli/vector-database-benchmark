#include <sstream>

#include "ssf/layer/proxy/windows/sspi_auth_impl.h"
#include "ssf/log/log.h"

namespace ssf {
namespace layer {
namespace proxy {

SSPIAuthImpl::SecPackageNames SSPIAuthImpl::sec_package_names_ = {"NTLM",
                                                                  "Negotiate"};

SSPIAuthImpl::SSPIAuthImpl(SecurityPackage sec_package,
                           const HttpProxy& proxy_ctx)
    : PlatformAuthImpl(proxy_ctx),
      sec_package_(sec_package),
      h_cred_(),
      h_sec_ctx_(),
      output_token_(),
      output_token_length_(0),
      service_name_() {
  memset(&h_sec_ctx_, 0, sizeof(CtxtHandle));
  memset(&h_cred_, 0, sizeof(CredHandle));
}

SSPIAuthImpl::~SSPIAuthImpl() { Clear(); }

void SSPIAuthImpl::Clear() {
  if (IsSecurityContextSet()) {
    ::DeleteSecurityContext(&h_sec_ctx_);
    memset(&h_sec_ctx_, 0, sizeof(CtxtHandle));
  }
  if (h_cred_.dwLower != 0 || h_cred_.dwUpper != 0) {
    ::FreeCredentialsHandle(&h_cred_);
    memset(&h_cred_, 0, sizeof(CredHandle));
  }
}

bool SSPIAuthImpl::Init() {
  // determine max output token buffer size
  PSecPkgInfoA p_win_sec_package;
  TimeStamp expiry;

  auto status = ::QuerySecurityPackageInfoA(
      const_cast<char*>(GenerateSecurityPackageName(sec_package_).c_str()),
      &p_win_sec_package);
  if (status != SEC_E_OK) {
    SSF_LOG("network_proxy", debug, "sspi[{}] could not query security package",
            sec_package_names_[sec_package_]);
    state_ = State::kFailure;
    return false;
  }

  output_token_.resize(p_win_sec_package->cbMaxToken);
  service_name_ = GenerateServiceName(sec_package_);

  SEC_WINNT_AUTH_IDENTITY_A identity;
  memset(&identity, 0, sizeof(SEC_WINNT_AUTH_IDENTITY_A));
  bool use_identity = false;
  if ((SecurityPackage::kNTLM == sec_package_ && !proxy_ctx_.reuse_ntlm) ||
      (SecurityPackage::kNegotiate == sec_package_ && !proxy_ctx_.reuse_kerb)) {
    use_identity = true;
    identity.Domain = reinterpret_cast<unsigned char*>(
        const_cast<char*>(proxy_ctx_.domain.c_str()));
    identity.DomainLength =
        static_cast<unsigned long>(proxy_ctx_.domain.size());
    identity.User = reinterpret_cast<unsigned char*>(
        const_cast<char*>(proxy_ctx_.username.c_str()));
    identity.UserLength =
        static_cast<unsigned long>(proxy_ctx_.username.size());
    identity.Password = reinterpret_cast<unsigned char*>(
        const_cast<char*>(proxy_ctx_.password.c_str()));
    identity.PasswordLength =
        static_cast<unsigned long>(proxy_ctx_.password.size());
    identity.Flags = SEC_WINNT_AUTH_IDENTITY_ANSI;
  }

  auto cred_status = ::AcquireCredentialsHandleA(
      NULL, p_win_sec_package->Name, SECPKG_CRED_OUTBOUND, NULL,
      (use_identity ? &identity : NULL), NULL, NULL, &h_cred_, &expiry);

  ::FreeContextBuffer(p_win_sec_package);

  if (cred_status != SEC_E_OK) {
    SSF_LOG("network_proxy", debug, "sspi[{}] could not acquire credentials",
            sec_package_names_[sec_package_]);
    state_ = State::kFailure;
    Clear();
    return false;
  }

  return true;
}

bool SSPIAuthImpl::ProcessServerToken(const Token& server_token) {
  TimeStamp expiry;
  SecBufferDesc in_sec_buf_desc;
  SecBuffer in_sec_buff;
  SecBufferDesc out_sec_buff_desc;
  SecBuffer out_sec_buff;

  out_sec_buff_desc.ulVersion = SECBUFFER_VERSION;
  out_sec_buff_desc.cBuffers = 1;
  out_sec_buff_desc.pBuffers = &out_sec_buff;
  out_sec_buff.BufferType = SECBUFFER_TOKEN;
  out_sec_buff.pvBuffer = output_token_.data();
  out_sec_buff.cbBuffer = static_cast<unsigned long>(output_token_.size());

  // input token
  in_sec_buf_desc.ulVersion = SECBUFFER_VERSION;
  in_sec_buf_desc.cBuffers = 1;
  in_sec_buf_desc.pBuffers = &in_sec_buff;
  in_sec_buff.BufferType = SECBUFFER_TOKEN;
  in_sec_buff.pvBuffer = const_cast<uint8_t*>(server_token.data());
  in_sec_buff.cbBuffer = static_cast<unsigned long>(server_token.size());

  // update security context
  unsigned long attrs;

  auto status = ::InitializeSecurityContextA(
      &h_cred_, (IsSecurityContextSet() ? &h_sec_ctx_ : NULL),
      const_cast<SEC_CHAR*>(service_name_.c_str()), 0, 0, SECURITY_NATIVE_DREP,
      server_token.empty() ? NULL : &in_sec_buf_desc, 0, &h_sec_ctx_,
      &out_sec_buff_desc, &attrs, &expiry);

  switch (status) {
    case SEC_E_OK:
      state_ = State::kSuccess;
      break;
    case SEC_I_COMPLETE_AND_CONTINUE:
    case SEC_I_COMPLETE_NEEDED: {
      auto complete_status =
          ::CompleteAuthToken(&h_sec_ctx_, &out_sec_buff_desc);
      state_ = complete_status == SEC_E_OK ? State::kContinue : State::kFailure;
      break;
    }
    case SEC_I_CONTINUE_NEEDED:
      state_ = State::kContinue;
      break;
    default:
      SSF_LOG("network_proxy", debug,
              "sspi[{}] error initializing security context",
              sec_package_names_[sec_package_]);
      state_ = State::kFailure;
  }

  output_token_length_ = out_sec_buff.cbBuffer;

  return state_ != kFailure;
}

SSPIAuthImpl::Token SSPIAuthImpl::GetAuthToken() {
  if (state_ == kFailure) {
    return {};
  }

  auto begin_it = output_token_.begin();
  auto end_it = begin_it + output_token_length_;

  return Token(begin_it, end_it);
}

std::string SSPIAuthImpl::GenerateSecurityPackageName(
    SecurityPackage sec_package) {
  switch (sec_package) {
    case kNTLM:
    case kNegotiate:
      return sec_package_names_[sec_package];
    default:
      return "";
  }
}

bool SSPIAuthImpl::IsSecurityContextSet() {
  return h_sec_ctx_.dwLower != 0 || h_sec_ctx_.dwUpper != 0;
}

std::string SSPIAuthImpl::GenerateServiceName(SecurityPackage sec_package) {
  std::stringstream ss_name;
  ss_name << "HTTP/";

  switch (sec_package) {
    case kNTLM:
      ss_name << proxy_ctx_.host << ":" << proxy_ctx_.port;
      break;
    case kNegotiate:
      ss_name << proxy_ctx_.host;
      break;
    default:
      break;
  }

  return ss_name.str();
}

}  // proxy
}  // layer
}  // ssf
