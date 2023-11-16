#include <eepp/network/ssl/backend/openssl/opensslsocket.hpp>

/** This implementation is based on the Godot Game Engine implementation (
 * https://github.com/okamstudio/godot ), MIT licensed. */

#ifdef EE_OPENSSL

#include <eepp/network/packet.hpp>
#include <eepp/network/ssl/backend/openssl/curl_hostcheck.h>
#include <eepp/system/filesystem.hpp>
#include <eepp/system/log.hpp>
#include <eepp/system/packmanager.hpp>

namespace EE { namespace Network { namespace SSL {

static std::vector<X509*> sCerts;

bool OpenSSLSocket::matchHostname( const char* name, const char* hostname ) {
	return Tool_Curl_cert_hostcheck( name, hostname ) == CURL_HOST_MATCH;
}

bool OpenSSLSocket::matchCommonName( const char* hostname, const X509* server_cert ) {
	int common_name_loc = -1;
	X509_NAME_ENTRY* common_name_entry = NULL;
	ASN1_STRING* common_name_asn1 = NULL;

	// Find the position of the CN field in the Subject field of the certificate
	common_name_loc = X509_NAME_get_index_by_NID( X509_get_subject_name( (X509*)server_cert ),
												  NID_commonName, -1 );

	// Extract the CN field
	common_name_entry =
		X509_NAME_get_entry( X509_get_subject_name( (X509*)server_cert ), common_name_loc );

	// Convert the CN field to a C string
	common_name_asn1 = X509_NAME_ENTRY_get_data( common_name_entry );

#if OPENSSL_VERSION_NUMBER < 0x10100000L
	char* common_name_str = (char*)ASN1_STRING_data( common_name_asn1 );
#else
	const char* common_name_str = (const char*)ASN1_STRING_get0_data( common_name_asn1 );
#endif

	// Make sure there isn't an embedded NUL character in the CN
	bool malformed_certificate =
		(size_t)ASN1_STRING_length( common_name_asn1 ) != strlen( common_name_str );

	if ( malformed_certificate ) {
		return false;
	}

	// Compare expected hostname with the CN
	return matchHostname( common_name_str, hostname );
}

/** Tries to find a match for hostname in the certificate's Subject Alternative Name extension. */
bool OpenSSLSocket::matchSubjectAlternativeName( const char* hostname, const X509* server_cert ) {
	bool result = false;
	int i;
	int san_names_nb = -1;
	STACK_OF( GENERAL_NAME )* san_names = NULL;

	// Try to extract the names within the SAN extension from the certificate
	san_names = (STACK_OF( GENERAL_NAME )*)X509_get_ext_d2i( (X509*)server_cert,
															 NID_subject_alt_name, NULL, NULL );

	if ( san_names == NULL ) {
		return false;
	}

	san_names_nb = sk_GENERAL_NAME_num( san_names );

	// Check each name within the extension
	for ( i = 0; i < san_names_nb; i++ ) {
		const GENERAL_NAME* current_name = sk_GENERAL_NAME_value( san_names, i );

		if ( current_name->type == GEN_DNS ) {
// Current name is a DNS name, let's check it
#if OPENSSL_VERSION_NUMBER < 0x10100000L
			char* dns_name = (char*)ASN1_STRING_data( current_name->d.dNSName );
#else
			const char* dns_name = (const char*)ASN1_STRING_get0_data( current_name->d.dNSName );
#endif

			// Make sure there isn't an embedded NUL character in the DNS name
			if ( (size_t)ASN1_STRING_length( current_name->d.dNSName ) != strlen( dns_name ) ) {
				result = false;
				break;
			} else { // Compare expected hostname with the DNS name
				if ( matchHostname( dns_name, hostname ) ) {
					result = true;
					break;
				}
			}
		}
	}

	sk_GENERAL_NAME_pop_free( san_names, GENERAL_NAME_free );

	return result;
}

int OpenSSLSocket::certVerifyCb( X509_STORE_CTX* x509_ctx, void* arg ) {
	/* This is the function that OpenSSL would call if we hadn't called
	 * SSL_CTX_set_cert_verify_callback().  Therefore, we are "wrapping"
	 * the default functionality, rather than replacing it. */
	bool base_cert_valid = 0 != X509_verify_cert( x509_ctx );

	if ( !base_cert_valid ) {
		Log::error( "Cause: %s",
					X509_verify_cert_error_string( X509_STORE_CTX_get_error( x509_ctx ) ) );
		ERR_print_errors_fp( stdout );
	}

	X509* server_cert = X509_STORE_CTX_get_current_cert( x509_ctx );

	char cert_str[256];
	X509_NAME_oneline( X509_get_subject_name( server_cert ), cert_str, sizeof( cert_str ) );

	Log::debug( "CERT STR: %s", cert_str );
	Log::debug( "VALID: %d", (int)base_cert_valid );

	if ( !base_cert_valid ) {
		return 0;
	}

	OpenSSLSocket* ssl = (OpenSSLSocket*)arg;

	if ( ssl->mSSLSocket->mValidateHostname ) {
		bool err = !matchSubjectAlternativeName( ssl->mSSLSocket->mHostName.c_str(), server_cert );

		if ( err ) {
			err = !matchCommonName( ssl->mSSLSocket->mHostName.c_str(), server_cert );
		}

		if ( err ) {
			ssl->mStatus = Socket::Error;
			return 0;
		}
	}

	return 1;
}

bool OpenSSLSocket::init() {
#if OPENSSL_VERSION_NUMBER < 0x10100000L
	CRYPTO_malloc_init(); // Initialize malloc, free, etc for OpenSSL's use

	SSL_library_init();
#else
	OPENSSL_init_ssl( 0, NULL );
#endif

	SSL_load_error_strings(); // Load SSL error strings

	ERR_load_BIO_strings(); // Load BIO error strings

	OpenSSL_add_all_algorithms(); // Load all available encryption algorithms

	ScopedBuffer data;

	if ( FileSystem::fileExists( SSLSocket::CertificatesPath ) ) {
		FileSystem::fileGet( SSLSocket::CertificatesPath, data );
	} else if ( PackManager::instance()->isFallbackToPacksActive() ) {
		std::string tPath( SSLSocket::CertificatesPath );

		Pack* tPack = PackManager::instance()->exists( tPath );

		if ( NULL != tPack ) {
			tPack->extractFileToMemory( tPath, data );
		}
	}

	if ( data.length() > 0 ) {
		BIO* mem = BIO_new( BIO_s_mem() );

		BIO_puts( mem, (const char*)data.get() );

		while ( true ) {
			X509* cert = PEM_read_bio_X509( mem, NULL, 0, NULL );

			if ( !cert )
				break;

			sCerts.push_back( cert );
		}

		BIO_free( mem );
	}

	return true;
}

bool OpenSSLSocket::end() {
	if ( !sCerts.empty() ) {
		for ( size_t i = 0; i < sCerts.size(); i++ ) {
			X509_free( sCerts[i] );
		}

		sCerts.clear();
	}

	return true;
}

OpenSSLSocket::OpenSSLSocket( SSLSocket* socket ) :
	SSLSocketImpl( socket ),
	mCTX( NULL ),
	mSSL( NULL ),
	mBIO( NULL ),
	mConnected( false ),
	mStatus( Socket::Disconnected ),
	mMaxCertChainDepth( 9 ) {
	mSSLSocket = socket;
}

OpenSSLSocket::~OpenSSLSocket() {
	disconnect();
}

Socket::Status OpenSSLSocket::connect( const IpAddress& /*remoteAddress*/,
									   unsigned short /*remotePort*/, Time /*timeout*/ ) {
	if ( mConnected ) {
		disconnect();
	}

	// Set up a SSL_CTX object, which will tell our BIO object how to do its work
	mCTX = SSL_CTX_new( SSLv23_client_method() );

	SSL_CTX_set_options( mCTX, SSL_OP_NO_TICKET );

	if ( mSSLSocket->mValidateCertificate ) {
		if ( !sCerts.empty() ) {
			// yay for undocumented OpenSSL functions
			X509_STORE* store = SSL_CTX_get_cert_store( mCTX );

			for ( size_t i = 0; i < sCerts.size(); i++ ) {
				X509_STORE_add_cert( store, sCerts[i] );
			}
		}

		/* Ask OpenSSL to verify the server certificate.  Note that this
		 * does NOT include verifying that the hostname is correct.
		 * So, by itself, this means anyone with any legitimate
		 * CA-issued certificate for any website, can impersonate any
		 * other website in the world.  This is not good.  See "The
		 * Most Dangerous Code in the World" article at
		 * https://crypto.stanford.edu/~dabo/pubs/abstracts/ssl-client-bugs.html
		 */
		SSL_CTX_set_verify( mCTX, SSL_VERIFY_PEER, NULL );

		/* This is how we solve the problem mentioned in the previous
		 * comment.  We "wrap" OpenSSL's validation routine in our
		 * own routine, which also validates the hostname by calling
		 * the code provided by iSECPartners.  Note that even though
		 * the "Everything You've Always Wanted to Know About
		 * Certificate Validation With OpenSSL (But Were Afraid to
		 * Ask)" paper from iSECPartners says very explicitly not to
		 * call SSL_CTX_set_cert_verify_callback (at the bottom of
		 * page 2), what we're doing here is safe because our
		 * cert_verify_callback() calls X509_verify_cert(), which is
		 * OpenSSL's built-in routine which would have been called if
		 * we hadn't set the callback.  Therefore, we're just
		 * "wrapping" OpenSSL's routine, not replacing it. */
		SSL_CTX_set_cert_verify_callback( mCTX, certVerifyCb, this );

		// Let the verify_callback catch the verify_depth error so that we get an appropriate error
		// in the logfile. (??)
		SSL_CTX_set_verify_depth( mCTX, mMaxCertChainDepth + 1 );
	}

	mSSL = SSL_new( mCTX );

	SSL_set_fd( mSSL, (int)mSSLSocket->mSocket );

	// Set the SSL to automatically retry on failure.
	SSL_set_mode( mSSL, SSL_MODE_AUTO_RETRY );

	mStatus = Socket::Done;

	int result = SSL_set_tlsext_host_name( mSSL, mSSLSocket->mHostName.c_str() );

	if ( 1 != result ) {
		ERR_print_errors_fp( stdout );

		printError( result );

		mStatus = Socket::Error;

		return mStatus;
	}

	// Same as before, try to connect.
	result = SSL_connect( mSSL );

	if ( result < 1 ) {
		ERR_print_errors_fp( stdout );

		printError( result );

		mStatus = Socket::Error;

		return mStatus;
	}

	X509* peer = SSL_get_peer_certificate( mSSL );

	if ( peer ) {
		// Log::debug( "cert_ok: %d", (int)( SSL_get_verify_result(mSSL) == X509_V_OK ) );
		mStatus = Socket::Done;
	} else if ( mSSLSocket->mValidateCertificate ) {
		mStatus = Socket::Error;
	}

	if ( mStatus == Socket::Done ) {
		mConnected = true;
	}

	return mStatus;
}

void OpenSSLSocket::disconnect() {
	if ( !mConnected )
		return;

	SSL_shutdown( mSSL );
	SSL_free( mSSL );
	SSL_CTX_free( mCTX );

	mSSL = NULL;
	mCTX = NULL;
	mConnected = false;
	mStatus = Socket::Disconnected;
}

void OpenSSLSocket::printError( int err ) {
	err = SSL_get_error( mSSL, err );

	switch ( err ) {
		case SSL_ERROR_NONE:
			Log::error( "NO ERROR: The TLS/SSL I/O operation completed" );
			break;
		case SSL_ERROR_ZERO_RETURN:
			Log::error( "The TLS/SSL connection has been closed." );
		case SSL_ERROR_WANT_READ:
		case SSL_ERROR_WANT_WRITE:
			Log::error( "The operation did not complete." );
			break;
		case SSL_ERROR_WANT_CONNECT:
		case SSL_ERROR_WANT_ACCEPT:
			Log::error( "The connect/accept operation did not complete" );
			break;
		case SSL_ERROR_WANT_X509_LOOKUP:
			Log::error( "The operation did not complete because an application callback set by "
						"SSL_CTX_set_client_cert_cb() has asked to be called again." );
			break;
		case SSL_ERROR_SYSCALL:
			Log::error( "Some I/O error occurred. The OpenSSL error queue may contain more "
						"information on the error." );
			break;
		case SSL_ERROR_SSL:
			Log::error( "A failure in the SSL library occurred, usually a protocol error." );
			break;
	}
}

Socket::Status OpenSSLSocket::send( const void* data, std::size_t size ) {
	Uint8* buf = (Uint8*)data;

	while ( size > 0 ) {
		int ret = SSL_write( mSSL, buf, size );

		if ( ret <= 0 ) {
			printError( ret );

			disconnect();

			return Socket::Disconnected;
		}

		buf += ret;
		size -= ret;
	}

	return Socket::Done;
}

Socket::Status OpenSSLSocket::receive( void* data, std::size_t size, std::size_t& received ) {
	if ( size == 0 ) {
		received = 0;
		return Socket::Done;
	}

	size_t iniSize = size;

	Uint8* buf = (Uint8*)data;

	int ret = SSL_read( mSSL, buf, size );

	if ( ret < 0 ) {
		printError( ret );
		disconnect();
		return Socket::Disconnected;
	} else if ( 0 >= ret ) {
		if ( size == iniSize ) {
			return Socket::Disconnected;
		}

		received = iniSize - size;

		return Socket::Done;
	}

	received = ret;

	return Socket::Done;
}

}}} // namespace EE::Network::SSL

#endif
