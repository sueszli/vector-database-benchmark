/*
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */


/*
 * lib/crypto/aes/aes_s2k.c
 *
 * Copyright 2003 by the Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * Export of this software from the United States of America may
 *   require a specific license from the United States Government.
 *   It is the responsibility of any person or organization contemplating
 *   export to obtain such a license before exporting.
 *
 * WITHIN THAT CONSTRAINT, permission to use, copy, modify, and
 * distribute this software and its documentation for any purpose and
 * without fee is hereby granted, provided that the above copyright
 * notice appear in all copies and that both that copyright notice and
 * this permission notice appear in supporting documentation, and that
 * the name of M.I.T. not be used in advertising or publicity pertaining
 * to distribution of the software without specific, written prior
 * permission.  Furthermore if you modify this software you must label
 * your software as modified software and not distribute it in such a
 * fashion that it might be confused with the original M.I.T. software.
 * M.I.T. makes no representations about the suitability of
 * this software for any purpose.  It is provided "as is" without express
 * or implied warranty.
 *
 *
 * krb5int_aes_string_to_key
 */

#include "k5-int.h"
#include "dk.h"
#include "aes_s2k.h"

#define DEFAULT_ITERATION_COUNT		4096 /* was 0xb000L in earlier drafts */
#define MAX_ITERATION_COUNT		0x1000000L

krb5_error_code
krb5int_aes_string_to_key(krb5_context context,
			  const struct krb5_enc_provider *enc,
			  const krb5_data *string,
			  const krb5_data *salt,
			  const krb5_data *params,
			  krb5_keyblock *key)
{
    unsigned long iter_count;
    krb5_data out;
    static const krb5_data usage = { KV5M_DATA, 8, "kerberos" };
    krb5_error_code err;
    /* Solaris Kerberos */
    krb5_keyblock *inkey = NULL;

    if (params) {
	unsigned char *p = (unsigned char *) params->data;
	if (params->length != 4)
	    return KRB5_ERR_BAD_S2K_PARAMS;
	/* The first two need casts in case 'int' is 16 bits.  */
	iter_count = (((unsigned long)p[0] << 24)
		      | ((unsigned long)p[1] << 16)
		      | (p[2] <<  8)
		      | (p[3]));
	if (iter_count == 0) {
	/*
	    iter_count = (1L << 16) << 16;
	    if (((iter_count >> 16) >> 16) != 1)
		return KRB5_ERR_BAD_S2K_PARAMS;
	*/
	    iter_count = DEFAULT_ITERATION_COUNT;
	}
    } else
	iter_count = DEFAULT_ITERATION_COUNT;

    /* This is not a protocol specification constraint; this is an
       implementation limit, which should eventually be controlled by
       a config file.  */
    if (iter_count >= MAX_ITERATION_COUNT)
	return KRB5_ERR_BAD_S2K_PARAMS;

    /*
     * Dense key space, no parity bits or anything, so take a shortcut
     * and use the key contents buffer for the generated bytes.
     */
    /* Solaris Kerberos */
    if (key->length != 16 && key->length != 32)
	return KRB5_CRYPTO_INTERNAL;
    out.data = (char *) key->contents;
    out.length = key->length;
    /* Solaris Kerberos */
    err = krb5int_pbkdf2_hmac_sha1 (context, &out, iter_count, key->enctype,
		string, salt);
    if (err) {
	memset(out.data, 0, out.length);
	return err;
    }

    /*
     * Solaris Kerberos:
     * The derive key operation below will not work correctly
     * if the input and output key pointers are to the same
     * data.  This is because the key object handle (PKCS#11)
     * gets out-of-sync with the original key when the contents
     * are modified.  We copy the original key here for use
     * below in the derive_key step, then we free this key
     * before exiting.
     */
    err  = krb5_copy_keyblock(context, key, &inkey);
    if (err) {
	memset(out.data, 0, out.length);
	return err;
    }

    err = krb5_derive_key (context, enc, inkey, key, &usage);
    if (err) {
	memset(out.data, 0, out.length);
    }
    krb5_free_keyblock(context, inkey);

    return (err);
}
