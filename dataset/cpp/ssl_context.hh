#pragma once

#include <li/http_server/tcp_server.hh>
#include <openssl/err.h>
#include <openssl/ssl.h>

namespace li {

// void cleanup_openssl() { EVP_cleanup(); }

// SSL context.
// Initialize the ssl context that will instantiate new ssl connection.
static bool openssl_initialized = false;
struct ssl_context {
  SSL_CTX* ctx = nullptr;

  ~ssl_context() {
    if (ctx)
      SSL_CTX_free(ctx);
  }

  ssl_context(const std::string& key_path, const std::string& cert_path, 
              const std::string& ciphers) {
    if (!openssl_initialized) {
      SSL_load_error_strings();
      OpenSSL_add_ssl_algorithms();
      openssl_initialized = true;
    }

    const SSL_METHOD* method;

    method = SSLv23_server_method();

    ctx = SSL_CTX_new(method);
    if (!ctx) {
      perror("Unable to create SSL context");
      ERR_print_errors_fp(stderr);
      exit(EXIT_FAILURE);
    }

    SSL_CTX_set_ecdh_auto(ctx, 1);

    /* Set the ciphersuite if provided */
    if (ciphers.size() && SSL_CTX_set_cipher_list(ctx, ciphers.c_str()) <= 0) {
      ERR_print_errors_fp(stderr);
      exit(EXIT_FAILURE);
    }

    /* Set the key and cert */
    if (SSL_CTX_use_certificate_file(ctx, cert_path.c_str(), SSL_FILETYPE_PEM) <= 0) {
      ERR_print_errors_fp(stderr);
      exit(EXIT_FAILURE);
    }

    if (SSL_CTX_use_PrivateKey_file(ctx, key_path.c_str(), SSL_FILETYPE_PEM) <= 0) {
      ERR_print_errors_fp(stderr);
      exit(EXIT_FAILURE);
    }

  }
};

} // namespace li
