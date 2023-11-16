#ifndef ELLE_CRYPTOGRAPHY_RSA_PEM_HH
# define ELLE_CRYPTOGRAPHY_RSA_PEM_HH

# include <filesystem>

# include <elle/cryptography/rsa/Padding.hh>
# include <elle/cryptography/rsa/KeyPair.hh>
# include <elle/cryptography/pem.hh>

namespace elle
{
  namespace cryptography
  {
    namespace rsa
    {
      namespace pem
      {
        /*----------.
        | Functions |
        `----------*/

        /// Import an RSA public key from a path.
        PublicKey
        import_K(std::filesystem::path const& path);
        /// Import an RSA private key from a path.
        PrivateKey
        import_k(std::filesystem::path const& path,
                 std::string const& passphrase =
                   cryptography::pem::defaults::passphrase);
        /// Import an RSA key pair from a path.
        KeyPair
        import_keypair(std::filesystem::path const& path,
                       std::string const& passphrase =
                         cryptography::pem::defaults::passphrase);
        /// Export an RSA public key.
        void
        export_K(PublicKey const& K,
                 std::filesystem::path const& path);
        /// Export an RSA private key, providing the passphrase, cipher and
        /// mode to encrypt it with.
        void
        export_k(PrivateKey const& k,
                 std::filesystem::path const& path,
                 std::string const& passphrase =
                   cryptography::pem::defaults::passphrase,
                 Cipher const& cipher =
                   cryptography::pem::defaults::cipher,
                 Mode const& mode =
                   cryptography::pem::defaults::mode);
        /// Export an RSA key pair.
        void
        export_keypair(KeyPair const& keypair,
                       std::filesystem::path const& path,
                       std::string const& passphrase =
                         cryptography::pem::defaults::passphrase,
                       Cipher const& cipher =
                         cryptography::pem::defaults::cipher,
                       Mode const& mode =
                         cryptography::pem::defaults::mode);
      }
    }
  }
}

#endif
