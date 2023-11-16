/*
 * Soft:        Keepalived is a failover program for the LVS project
 *              <www.linuxvirtualserver.org>. It monitor & manipulate
 *              a loadbalanced server pool using multi-layer checks.
 *
 * Part:        SSL GET CHECK. Perform an ssl get query to a specified
 *              url, compute a MD5 over this result and match it to the
 *              expected value.
 *
 * Authors:     Alexandre Cassen, <acassen@linux-vs.org>
 *              Jan Holmberg, <jan@artech.net>
 *
 *              This program is distributed in the hope that it will be useful,
 *              but WITHOUT ANY WARRANTY; without even the implied warranty of
 *              MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *              See the GNU General Public License for more details.
 *
 *              This program is free software; you can redistribute it and/or
 *              modify it under the terms of the GNU General Public License
 *              as published by the Free Software Foundation; either version
 *              2 of the License, or (at your option) any later version.
 *
 * Copyright (C) 2001-2017 Alexandre Cassen, <acassen@gmail.com>
 */

#include "config.h"

#include <fcntl.h>
#include <openssl/md5.h>
#include <openssl/err.h>

#include "check_ssl.h"
#include "check_api.h"
#include "check_http.h"
#include "logger.h"
#ifdef THREAD_DUMP
#include "scheduler.h"
#endif

/* SSL primitives */
/* Free an SSL context */
void
clear_ssl(ssl_data_t *ssl)
{
	if (ssl && ssl->ctx) {
		SSL_CTX_free(ssl->ctx);
		ssl->ctx = NULL;
	}
}

/* PEM password callback function */
static int
password_cb(char *buf, int num, __attribute__((unused)) int rwflag, void *userdata)
{
	ssl_data_t *ssl = PTR_CAST(ssl_data_t, userdata);
	size_t plen = strlen(ssl->password);

	if ((unsigned)num < plen + 1)
		return (0);

	strcpy(buf, ssl->password);
	return (int)plen;
}

/* Inititalize global SSL context */
static bool
build_ssl_ctx(void)
{
	ssl_data_t *ssl;

	/* Library initialization */
#ifdef HAVE_OPENSSL_INIT_CRYPTO
#ifndef HAVE_OPENSSL_INIT_NO_LOAD_CONFIG_BUG
	/* In OpenSSL v1.1.1 if the following is called, SSL_CTX_new() below fails.
	 * It works in v1.1.0h and v1.1.1b.
	 * It transpires that it works without setting NO_LOAD_CONFIG, but it is
	 * presumably more efficient not to load it. */
	if (!OPENSSL_init_crypto(OPENSSL_INIT_NO_LOAD_CONFIG, NULL))
		log_message(LOG_INFO, "OPENSSL_init_crypto failed");
#endif
#else
	SSL_library_init();
	SSL_load_error_strings();
#endif

	if (!check_data->ssl) {
		PMALLOC(ssl);
	} else
		ssl = check_data->ssl;

	/* Initialize SSL context */
#ifdef HAVE_TLS_METHOD
	ssl->meth = TLS_method();
#else
	ssl->meth = SSLv23_method();
#endif
	if (!(ssl->ctx = SSL_CTX_new(ssl->meth))) {
		log_message(LOG_INFO, "SSL error: cannot create new SSL context");

		if (!check_data->ssl)
			FREE(ssl);

		return false;
	}

	/* return for autogen context */
	if (!check_data->ssl) {
		check_data->ssl = ssl;
		goto end;
	}

	/* Load our keys and certificates */
	if (check_data->ssl->certfile)
		if (!(SSL_CTX_use_certificate_chain_file(ssl->ctx,
							 check_data->ssl->certfile))) {
			log_message(LOG_INFO, "SSL error : Cant load certificate file...");
			return false;
		}

	/* Handle password callback using userdata ssl */
	if (check_data->ssl->password) {
		SSL_CTX_set_default_passwd_cb_userdata(ssl->ctx,
						       check_data->ssl);
		SSL_CTX_set_default_passwd_cb(ssl->ctx, password_cb);
	}

	if (check_data->ssl->keyfile)
		if (!(SSL_CTX_use_PrivateKey_file(ssl->ctx,
						  check_data->ssl->keyfile,
						  SSL_FILETYPE_PEM))) {
			log_message(LOG_INFO, "SSL error : Cant load key file...");
			return false;
		}

	/* Load the CAs we trust */
	if (check_data->ssl->cafile)
		if (!(SSL_CTX_load_verify_locations(ssl->ctx,
						    check_data->ssl->cafile, 0))) {
			log_message(LOG_INFO, "SSL error : Cant load CA file...");
			return false;
		}

      end:
#if HAVE_SSL_CTX_SET_VERIFY_DEPTH
	SSL_CTX_set_verify_depth(ssl->ctx, 1);
#endif

	return true;
}

/*
 * Initialize the SSL context, with or without specific
 * configuration files.
 */
bool
init_ssl_ctx(void)
{
	ssl_data_t *ssl = check_data->ssl;

	if (!build_ssl_ctx()) {
		log_message(LOG_INFO, "Error Initialize SSL, ctx Instance");
		log_message(LOG_INFO, "  SSL  keyfile:%s", ssl->keyfile);
		log_message(LOG_INFO, "  SSL password:%s", ssl->password);
		log_message(LOG_INFO, "  SSL   cafile:%s", ssl->cafile);
		log_message(LOG_INFO, "Terminate...");
		clear_ssl(ssl);
		return false;
	}
	return true;
}

/* Display SSL error to readable string */
int
ssl_printerr(int err)
{
	switch (err) {
	case SSL_ERROR_ZERO_RETURN:
		log_message(LOG_INFO, "  SSL error: (zero return)");
		break;
	case SSL_ERROR_WANT_READ:
		log_message(LOG_INFO, "  SSL error: (read error)");
		break;
	case SSL_ERROR_WANT_WRITE:
		log_message(LOG_INFO, "  SSL error: (write error)");
		break;
	case SSL_ERROR_WANT_CONNECT:
		log_message(LOG_INFO, "  SSL error: (connect error)");
		break;
	case SSL_ERROR_WANT_X509_LOOKUP:
		log_message(LOG_INFO, "  SSL error: (X509 lookup error)");
		break;
	case SSL_ERROR_SYSCALL:
		log_message(LOG_INFO, "  SSL error: (syscall error)");
		break;
	case SSL_ERROR_SSL:
		/* Note: the following is not thread safe. Use MALLOC(256) and ERR_error_string_n if need thread safety */
		log_message(LOG_INFO, "  SSL error: (%s)", ERR_error_string(ERR_get_error(), NULL));
		break;
	}
	return 0;
}

int
ssl_connect(thread_ref_t thread, int new_req)
{
	checker_t *checker = THREAD_ARG(thread);
	http_checker_t *http_get_check = CHECKER_ARG(checker);
	request_t *req = http_get_check->req;
#ifdef _HAVE_SSL_SET_TLSEXT_HOST_NAME_
	url_t *url = http_get_check->url_it;
	/* The man page for SSL_set_tlsext_host_name states name is const char *,
	 * but it is cast to a void * */
	union {
		const char *name_const;
		char *name;
	} vhost;
#endif
	int ret;

	/* First round, create SSL context */
	if (new_req) {
		int bio_fd;

		if (!(req->ssl = SSL_new(check_data->ssl->ctx))) {
			log_message(LOG_INFO, "Unable to establish ssl connection - SSL_new() failed");
			return 0;
		}

		if (!(req->bio = BIO_new_socket(thread->u.f.fd, BIO_NOCLOSE))) {
			log_message(LOG_INFO, "Unable to establish ssl connection - BIO_new_socket() failed");
			return 0;
		}

		BIO_get_fd(req->bio, &bio_fd);
		if (bio_fd != thread->u.f.fd) {
			if (fcntl(bio_fd, F_SETFD, fcntl(bio_fd, F_GETFD) | FD_CLOEXEC) == -1)
				log_message(LOG_INFO, "Setting CLOEXEC failed on ssl socket - errno %d", errno);
		}

		/* There is a memory leak in openSSL at least in version 3.0.1, which is fixed
		 * by version 3.0.5. It was not present in version 1.1.1n. Since I haven't been
		 * able to identify the OpenSSL patch that resolved the leak, we play safe and
		 * assume it is in versions 3.0.0 up to 3.0.4.
		 * The leak is memory allocated by
		 *   p = OPENSSL_malloc(len);
		 * in ssl3_setup_write_buffer() in ssl/record/ssl_buffer.c
		 *
		 * It appears that setting SSL_MODE_RELEASE_BUFFERS causes the memory leak not
		 * to occur.
		 */
#ifdef OPENSSL_VERSION_MAJOR
#if OPENSSL_VERSION_MAJOR == 3 && OPENSSL_VERSION_MINOR == 0 && OPENSSL_VERSION_PATCH <= 4
		SSL_set_mode(req->ssl, SSL_MODE_RELEASE_BUFFERS);
#endif
#endif

		if (!url->tls_compliant && !http_get_check->tls_compliant)
			SSL_set_quiet_shutdown(req->ssl, 1);

#if defined HAVE_SSL_SET0_RBIO && defined HAVE_SSL_SET0_WBIO
		BIO_up_ref(req->bio);
		SSL_set0_rbio(req->ssl, req->bio);
		SSL_set0_wbio(req->ssl, req->bio);
#else
		SSL_set_bio(req->ssl, req->bio, req->bio);
#endif
#ifdef _HAVE_SSL_SET_TLSEXT_HOST_NAME_
		if (http_get_check->enable_sni) {
			if (url && url->virtualhost)
				vhost.name_const = url->virtualhost;
			else if (http_get_check->virtualhost)
				vhost.name_const = http_get_check->virtualhost;
			else if (checker->vs->virtualhost)
				vhost.name_const = checker->vs->virtualhost;
			else
				vhost.name_const = NULL;
			if (vhost.name)
				SSL_set_tlsext_host_name(req->ssl, vhost.name);
		}
#endif
	}

	ret = SSL_connect(req->ssl);

	return ret;
}

bool
ssl_send_request(SSL * ssl, const char *str_request, int request_len)
{
	int r;

	r = SSL_write(ssl, str_request, request_len);

	return (r == request_len);
}

/* Asynchronous SSL stream reader */
void
ssl_read_thread(thread_ref_t thread)
{
	checker_t *checker = THREAD_ARG(thread);
	http_checker_t *http_get_check = CHECKER_ARG(checker);
	request_t *req = http_get_check->req;
	url_t *url = http_get_check->url_it;
	unsigned long timeout;
	unsigned char digest[MD5_DIGEST_LENGTH];
	int r = 0;

	/* Handle read timeout */
	if (thread->type == THREAD_READ_TIMEOUT) {
		SSL_shutdown(req->ssl);

		timeout_epilog(thread, "Timeout SSL read");
		return;
	}

	timeout = timer_long(thread->sands) - timer_long(time_now);

	/* read the SSL stream - allow for terminating the data with '\0 */
	r = SSL_read(req->ssl, req->buffer + req->len, (int)(MAX_BUFFER_LENGTH - 1 - req->len));

	if (r > 0) {
		/* Handle response stream */
		http_process_response(thread, req, (size_t)r, url);

		/*
		 * Register next ssl stream reader.
		 * Register itself to not perturbe global I/O multiplexer.
		 */

		thread_add_read(thread->master, ssl_read_thread, checker,
				thread->u.f.fd, timeout, THREAD_DESTROY_CLOSE_FD);
		return;
	}

	req->error = SSL_get_error(req->ssl, r);

	if (req->error == SSL_ERROR_WANT_READ) {
		 /* async read unfinished */
		thread_add_read(thread->master, ssl_read_thread, checker,
				thread->u.f.fd, timeout, THREAD_DESTROY_CLOSE_FD);
		return;
	}

	if (req->error == SSL_ERROR_SSL) {
		const char *file;
		int line;
#ifdef HAVE_ERR_GET_ERROR_ALL
		const char *func;
#endif
		const char *data;
		int flags;
		unsigned long err;

#ifdef HAVE_ERR_GET_ERROR_ALL
		while ((err = ERR_get_error_all(&file, &line, &func, &data, &flags))) {
#else
		while ((err = ERR_get_error_line_data(&file, &line, &data, &flags))) {

#endif
			/* Don't output the same error message repeatedly */
			if (err == url->last_ssl_error)
				continue;

			url->last_ssl_error = err;

			log_message(LOG_INFO, "SSL error '%s' at %s:%d "
#ifdef HAVE_ERR_GET_ERROR_ALL
					"(%s) "
#endif
					"data '%s' flags 0x%x",
					ERR_error_string(err, NULL),
					file, line,
#ifdef HAVE_ERR_GET_ERROR_ALL
					func,
#endif
					data, (unsigned)flags);
		}
	}

	/* All the SSL stream has been parsed */
	if (url->digest) {
		EVP_DigestFinal_ex(req->context, digest, NULL);
		EVP_MD_CTX_free(req->context);
		req->context = NULL;
		if (req->error == SSL_ERROR_ZERO_RETURN &&
		    http_get_check->genhash_flags & GENHASH_VERBOSE)
			dump_digest(digest, MD5_DIGEST_LENGTH);
	} else
		digest[0] = 0;

	if (req->error != SSL_ERROR_SSL && req->error != SSL_ERROR_SYSCALL)
		r = SSL_shutdown(req->ssl);
	else
		r = 0;

	if (r && !req->extracted) {
		timeout_epilog(thread, "SSL read error from");
		return;
	}

	/* Handle response stream */
	http_handle_response(thread, digest, !req->extracted);
}

#ifdef THREAD_DUMP
void
register_check_ssl_addresses(void)
{
	register_thread_address("ssl_read_thread", ssl_read_thread);
}
#endif
