// Copyright 2020 Google LLC
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
//
///////////////////////////////////////////////////////////////////////////////

#include "pybind11/pybind11.h"
#include "tink/aead.h"
#include "tink/util/statusor.h"
#include "tink/cc/pybind/aead.h"
#include "tink/cc/pybind/cc_hpke_config.h"
#include "tink/cc/pybind/cc_jwt_config.h"
#include "tink/cc/pybind/cc_key_manager.h"
#include "tink/cc/pybind/cc_streaming_aead_wrappers.h"
#include "tink/cc/pybind/cc_tink_config.h"
#include "tink/cc/pybind/deterministic_aead.h"
#include "tink/cc/pybind/hybrid_decrypt.h"
#include "tink/cc/pybind/hybrid_encrypt.h"
#include "tink/cc/pybind/input_stream_adapter.h"
#include "tink/cc/pybind/mac.h"
#include "tink/cc/pybind/output_stream_adapter.h"
#include "tink/cc/pybind/prf.h"
#include "tink/cc/pybind/public_key_sign.h"
#include "tink/cc/pybind/public_key_verify.h"
#include "tink/cc/pybind/python_file_object_adapter.h"
#include "tink/cc/pybind/streaming_aead.h"
#include "tink/cc/pybind/tink_exception.h"


namespace crypto {
namespace tink {

PYBIND11_MODULE(tink_bindings, m) {
  namespace py = pybind11;

  py::register_exception<pybind11::google_tink::TinkException>(
      m, "PythonTinkException");
  PybindRegisterCcStreamingAeadWrappers(&m);
  PybindRegisterAead(&m);
  PybindRegisterHybridEncrypt(&m);
  PybindRegisterCcTinkConfig(&m);
  PybindRegisterCcHpkeConfig(&m);
  PybindRegisterCcJwtConfig(&m);
  PybindRegisterStreamingAead(&m);
  PybindRegisterDeterministicAead(&m);
  PybindRegisterPublicKeySign(&m);
  PybindRegisterMac(&m);
  PybindRegisterPrf(&m);
  PybindRegisterHybridDecrypt(&m);
  PybindRegisterOutputStreamAdapter(&m);
  PybindRegisterCcKeyManager(&m);
  PybindRegisterPythonFileObjectAdapter(&m);
  PybindRegisterInputStreamAdapter(&m);
  PybindRegisterPublicKeyVerify(&m);
}

}  // namespace tink
}  // namespace crypto
