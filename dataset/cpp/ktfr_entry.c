/*
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */


/*
 * lib/krb5/keytab/ktfr_entry.c
 *
 * Copyright 1990 by the Massachusetts Institute of Technology.
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
 * krb5_kt_free_entry()
 */

#include "k5-int.h"

krb5_error_code KRB5_CALLCONV
krb5_free_keytab_entry_contents (krb5_context context, krb5_keytab_entry *entry)
{
    if (!entry)
	return 0;

    krb5_free_principal(context, entry->principal);
    if (entry->key.contents) {
	memset((char *)entry->key.contents, 0, entry->key.length);
	/* Solaris Kerberos */
	krb5_free_keyblock_contents(context, &entry->key);
    }
    return 0;
}

krb5_error_code KRB5_CALLCONV
krb5_kt_free_entry (krb5_context context, krb5_keytab_entry *entry)
{
    return krb5_free_keytab_entry_contents (context, entry);
}
