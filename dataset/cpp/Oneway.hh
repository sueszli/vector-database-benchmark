#pragma once

#include <iosfwd>

#include <elle/cryptography/fwd.hh>

#include <elle/serialization/Serializer.hh>

namespace elle
{
  namespace cryptography
  {
    /*-------------.
    | Enumerations |
    `-------------*/

    /// Define the oneway algorithm.
    enum class Oneway
    {
      md5,
      sha,
      sha1,
      sha224,
      sha256,
      sha384,
      sha512
    };

    /*----------.
    | Operators |
    `----------*/

    std::ostream&
    operator <<(std::ostream& stream,
                Oneway const oneway);

    namespace oneway
    {
      /*----------.
      | Functions |
      `----------*/

      /// Resolve a algorithm name into an EVP function pointer.
      ::EVP_MD const*
      resolve(Oneway const name);
      /// Return the name of the oneway function.
      Oneway
      resolve(::EVP_MD const* function);
    }
  }
}

/*--------------.
| Serialization |
`--------------*/

namespace elle
{
  namespace serialization
  {
    template <>
    struct Serialize<cryptography::Oneway>
    {
      using Type = uint8_t;
      static
      uint8_t
      convert(elle::cryptography::Oneway const& value);
      static
      elle::cryptography::Oneway
      convert(uint8_t const& representation);
    };
  }
}
