/*
 * Copyright (c) 2002-2016 Balabit
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * As an additional exemption you are allowed to compile & link against the
 * OpenSSL libraries as published by the OpenSSL project. See the file
 * COPYING for details.
 *
 */

#include "compat/openssl_support.h"
#include "syslog-ng.h"
#include "thread-utils.h"
#include <openssl/ssl.h>
#include <openssl/bn.h>

#if !SYSLOG_NG_HAVE_DECL_SSL_CTX_GET0_PARAM
X509_VERIFY_PARAM *SSL_CTX_get0_param(SSL_CTX *ctx)
{
  return ctx->param;
}
#endif

#if !SYSLOG_NG_HAVE_DECL_X509_STORE_CTX_GET0_CERT
X509 *X509_STORE_CTX_get0_cert(X509_STORE_CTX *ctx)
{
  return ctx->cert;
}
#endif

#if !SYSLOG_NG_HAVE_DECL_X509_GET_EXTENSION_FLAGS
uint32_t X509_get_extension_flags(X509 *x)
{
  return x->ex_flags;
}
#endif


/* ThreadID callbacks for various OpenSSL versions */
#if OPENSSL_VERSION_NUMBER < 0x10000000

static unsigned long
_ssl_thread_id(void)
{
  return (unsigned long) get_thread_id();
}

static void
_init_threadid_callback(void)
{
  CRYPTO_set_id_callback(_ssl_thread_id);
}

#elif OPENSSL_VERSION_NUMBER < 0x10100000L

static void
_ssl_thread_id2(CRYPTO_THREADID *id)
{
  CRYPTO_THREADID_set_numeric(id, (unsigned long) get_thread_id());
}

static void
_init_threadid_callback(void)
{
  CRYPTO_THREADID_set_callback(_ssl_thread_id2);
}

#endif

/* locking callbacks for OpenSSL prior to 1.1.0 */
#if OPENSSL_VERSION_NUMBER < 0x10100000L

static gint ssl_lock_count;
static GMutex *ssl_locks;

static void
_ssl_locking_callback(int mode, int type, const char *file, int line)
{
  if (mode & CRYPTO_LOCK)
    {
      g_mutex_lock(&ssl_locks[type]);
    }
  else
    {
      g_mutex_unlock(&ssl_locks[type]);
    }
}

static void
_init_locks(void)
{
  gint i;

  ssl_lock_count = CRYPTO_num_locks();
  ssl_locks = g_new(GMutex, ssl_lock_count);
  for (i = 0; i < ssl_lock_count; i++)
    {
      g_mutex_init(&ssl_locks[i]);
    }
  CRYPTO_set_locking_callback(_ssl_locking_callback);
}

static void
_deinit_locks(void)
{
  gint i;

  for (i = 0; i < ssl_lock_count; i++)
    {
      g_mutex_clear(&ssl_locks[i]);
    }
  g_free(ssl_locks);
}

void
openssl_crypto_init_threading(void)
{
  _init_locks();
  _init_threadid_callback();
}

void
openssl_crypto_deinit_threading(void)
{
  _deinit_locks();
}

#else

void
openssl_crypto_init_threading(void)
{
}

void
openssl_crypto_deinit_threading(void)
{
}

#endif

void
openssl_init(void)
{
#if OPENSSL_VERSION_NUMBER < 0x10100000L
  SSL_library_init();
  SSL_load_error_strings();
  OpenSSL_add_all_algorithms();
#endif
}

void
openssl_ctx_setup_ecdh(SSL_CTX *ctx)
{
#if OPENSSL_VERSION_NUMBER >= 0x10100000L

  /* No need to setup as ECDH auto is the default */

#elif OPENSSL_VERSION_NUMBER >= 0x10002000L

  SSL_CTX_set_ecdh_auto(ctx, 1);

#elif OPENSSL_VERSION_NUMBER >= 0x10001000L

  EC_KEY *ecdh = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
  if (!ecdh)
    return;

  SSL_CTX_set_tmp_ecdh(ctx, ecdh);
  EC_KEY_free(ecdh);

#endif
}

gboolean
openssl_ctx_setup_dh(SSL_CTX *ctx)
{
#if OPENSSL_VERSION_NUMBER >= 0x30000000L
  return SSL_CTX_set_dh_auto(ctx, 1);
#else
  DH *dh = DH_new();
  if (!dh)
    return 0;

  /*
   * "2048-bit MODP Group" from RFC3526, Section 3.
   *
   * The prime is: 2^2048 - 2^1984 - 1 + 2^64 * { [2^1918 pi] + 124476 }
   *
   * RFC3526 specifies a generator of 2.
   */

  BIGNUM *g = NULL;
  BN_dec2bn(&g, "2");

  if (!DH_set0_pqg(dh, BN_get_rfc3526_prime_2048(NULL), NULL, g))
    {
      BN_free(g);
      DH_free(dh);
      return 0;
    }

  long ctx_dh_success = SSL_CTX_set_tmp_dh(ctx, dh);

  DH_free(dh);
  return ctx_dh_success;
#endif
}

#if OPENSSL_VERSION_NUMBER < 0x30000000L
static long
_is_dh_valid(DH *dh)
{
  if (!dh)
    return 0;

  int check_flags;
  if (!DH_check(dh, &check_flags))
    return 0;

  long error_flag_is_set = check_flags &
                           (DH_CHECK_P_NOT_PRIME
                            | DH_UNABLE_TO_CHECK_GENERATOR
                            | DH_CHECK_P_NOT_SAFE_PRIME
                            | DH_NOT_SUITABLE_GENERATOR);

  return !error_flag_is_set;
}
#endif

gboolean
openssl_ctx_load_dh_from_file(SSL_CTX *ctx, const gchar *dhparam_file)
{
  BIO *bio = BIO_new_file(dhparam_file, "r");
  if (!bio)
    return 0;

#if OPENSSL_VERSION_NUMBER >= 0x30000000L
  EVP_PKEY *dh_params = PEM_read_bio_Parameters(bio, NULL);
  BIO_free(bio);

  if (!dh_params)
    return 0;

  int ctx_dh_success = SSL_CTX_set0_tmp_dh_pkey(ctx, dh_params);

  if (!ctx_dh_success)
    EVP_PKEY_free(dh_params);

  return ctx_dh_success;

#else
  DH *dh = PEM_read_bio_DHparams(bio, NULL, NULL, NULL);
  BIO_free(bio);

  if (!_is_dh_valid(dh))
    {
      DH_free(dh);
      return 0;
    }

  long ctx_dh_success = SSL_CTX_set_tmp_dh(ctx, dh);

  DH_free(dh);
  return ctx_dh_success;
#endif
}

#if !SYSLOG_NG_HAVE_DECL_DH_SET0_PQG && OPENSSL_VERSION_NUMBER < 0x30000000L
int DH_set0_pqg(DH *dh, BIGNUM *p, BIGNUM *q, BIGNUM *g)
{
  if ((dh->p == NULL && p == NULL)
      || (dh->g == NULL && g == NULL))
    return 0;

  if (p != NULL)
    {
      BN_free(dh->p);
      dh->p = p;
    }
  if (q != NULL)
    {
      BN_free(dh->q);
      dh->q = q;
    }
  if (g != NULL)
    {
      BN_free(dh->g);
      dh->g = g;
    }

  if (q != NULL)
    dh->length = BN_num_bits(q);

  return 1;
}
#endif

#if !SYSLOG_NG_HAVE_DECL_BN_GET_RFC3526_PRIME_2048
BIGNUM *
BN_get_rfc3526_prime_2048(BIGNUM *bn)
{
  return get_rfc3526_prime_2048(bn);
}
#endif

void
openssl_ctx_setup_session_tickets(SSL_CTX *ctx)
{
  /* This is a workaround for an OpenSSL TLS 1.3 bug that results in data loss
   * when one-way protocols are used and a connection is closed by the client
   * right after sending data.
   *
   * Remove this call, or make it version-dependent after the bug has been fixed:
   * - https://github.com/openssl/openssl/issues/10880
   * - https://github.com/openssl/openssl/issues/7948
   */

#if SYSLOG_NG_HAVE_DECL_SSL_CTX_SET_NUM_TICKETS
  SSL_CTX_set_num_tickets(ctx, 0);
#endif
}
