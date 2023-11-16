#include <elle/cryptography/rsa/low.hh>
#include <elle/cryptography/rsa/der.hh>
#include <elle/cryptography/finally.hh>

#include <elle/log.hh>
#include <elle/Buffer.hh>

#include <openssl/rsa.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/engine.h>
#include <openssl/crypto.h>
#include <openssl/err.h>

namespace elle
{
  namespace cryptography
  {
    namespace rsa
    {
      namespace low
      {
        /*----------.
        | Functions |
        `----------*/

        ::RSA*
        RSA_priv2pub(::RSA* private_key)
        {
          ELLE_ASSERT(private_key);

          return (::RSAPublicKey_dup(private_key));
        }

        ::RSA*
        RSA_dup(::RSA* key)
        {
          ELLE_ASSERT(key);

          // Increase the reference counter on this object rather
          // than duplicating the structure.
          ::RSA_up_ref(key);

          return (key);
        }
      }
    }
  }
}
