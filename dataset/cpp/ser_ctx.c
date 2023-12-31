/*
 * lib/krb5/krb/ser_ctx.c
 *
 * Copyright 1995 by the Massachusetts Institute of Technology.
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
 */

/*
 * ser_ctx.c	- Routines to deal with serializing the krb5_context and
 *		  krb5_os_context structures.
 */
#include "k5-int.h"

/*
 * Routines to deal with externalizing the krb5_context:
 *	krb5_context_size();
 *	krb5_context_externalize();
 *	krb5_context_internalize();
 *
 * Routines to deal with externalizing the krb5_os_context:
 *	krb5_oscontext_size();
 *	krb5_oscontext_externalize();
 *	krb5_oscontext_internalize();
 *
 * Routines to deal with externalizing the profile.
 *	profile_ser_size();
 *	profile_ser_externalize();
 *	profile_ser_internalize();
 *
 * Interface to initialize serializing of krb5_context and krb5_os_context:
 *	krb5_ser_context_init();
 */
static krb5_error_code krb5_context_size
	(krb5_context, krb5_pointer, size_t *);
static krb5_error_code krb5_context_externalize
	(krb5_context, krb5_pointer, krb5_octet **, size_t *);
static krb5_error_code krb5_context_internalize
	(krb5_context,krb5_pointer *, krb5_octet **, size_t *);
static krb5_error_code krb5_oscontext_size
	(krb5_context, krb5_pointer, size_t *);
static krb5_error_code krb5_oscontext_externalize
	(krb5_context, krb5_pointer, krb5_octet **, size_t *);
static krb5_error_code krb5_oscontext_internalize
	(krb5_context,krb5_pointer *, krb5_octet **, size_t *);
#ifndef _KERNEL
krb5_error_code profile_ser_size
	(krb5_context, krb5_pointer, size_t *);
krb5_error_code profile_ser_externalize
	(krb5_context, krb5_pointer, krb5_octet **, size_t *);
krb5_error_code profile_ser_internalize
	(krb5_context,krb5_pointer *, krb5_octet **, size_t *);
#endif
/* Local data */
static const krb5_ser_entry krb5_context_ser_entry = {
    KV5M_CONTEXT,			/* Type 		*/
    krb5_context_size,			/* Sizer routine	*/
    krb5_context_externalize,		/* Externalize routine	*/
    krb5_context_internalize		/* Internalize routine	*/
};
static const krb5_ser_entry krb5_oscontext_ser_entry = {
    KV5M_OS_CONTEXT,			/* Type			*/
    krb5_oscontext_size,		/* Sizer routine	*/
    krb5_oscontext_externalize,		/* Externalize routine	*/
    krb5_oscontext_internalize		/* Internalize routine	*/
};
#ifndef _KERNEL
static const krb5_ser_entry krb5_profile_ser_entry = {
    PROF_MAGIC_PROFILE,			/* Type			*/
    profile_ser_size,			/* Sizer routine	*/
    profile_ser_externalize,		/* Externalize routine	*/
    profile_ser_internalize		/* Internalize routine	*/
};
#endif

/*
 * krb5_context_size()	- Determine the size required to externalize the
 *			  krb5_context.
 */
static krb5_error_code
krb5_context_size(krb5_context kcontext, krb5_pointer arg, size_t *sizep)
{
    krb5_error_code	kret;
    size_t		required;
    krb5_context	context;

    /*
     * The KRB5 context itself requires:
     *	krb5_int32			for KV5M_CONTEXT
     *	krb5_int32			for sizeof(default_realm)
     *	strlen(default_realm)		for default_realm.
     *	krb5_int32			for n_in_tkt_ktypes*sizeof(krb5_int32)
     *	nktypes*sizeof(krb5_int32)	for in_tkt_ktypes.
     *	krb5_int32			for n_tgs_ktypes*sizeof(krb5_int32)
     *	nktypes*sizeof(krb5_int32)	for tgs_ktypes.
     *  krb5_int32			for clockskew
     *  krb5_int32			for kdc_req_sumtype
     *  krb5_int32			for ap_req_sumtype
     *  krb5_int32			for safe_sumtype
     *  krb5_int32			for kdc_default_options
     *  krb5_int32			for library_options
     *  krb5_int32			for profile_secure
     * 	krb5_int32			for fcc_default_format
     *  krb5_int32			for scc_default_format
     *    <>				for os_context
     *    <>				for db_context
     *    <>				for profile
     *	krb5_int32			for trailer.
     */
    kret = EINVAL;
    if ((context = (krb5_context) arg)) {
	/* Calculate base length */
	required = (14 * sizeof(krb5_int32) +
		    (context->in_tkt_ktype_count * sizeof(krb5_int32)) +
		    (context->tgs_ktype_count * sizeof(krb5_int32)));

	if (context->default_realm)
	    required += strlen(context->default_realm);
	/* Calculate size required by os_context, if appropriate */
	if (context->os_context)
	    kret = krb5_size_opaque(kcontext,
				    KV5M_OS_CONTEXT,
				    (krb5_pointer) context->os_context,
				    &required);

	/* Calculate size required by db_context, if appropriate */
	if (!kret && context->db_context)
	    kret = krb5_size_opaque(kcontext,
				    KV5M_DB_CONTEXT,
				    (krb5_pointer) context->db_context,
				    &required);

	/* Finally, calculate size required by profile, if appropriate */
	if (!kret && context->profile)
	    kret = krb5_size_opaque(kcontext,
				    PROF_MAGIC_PROFILE,
				    (krb5_pointer) context->profile,
				    &required);
    }
    if (!kret)
	*sizep += required;
    return(kret);
}

/*
 * krb5_context_externalize()	- Externalize the krb5_context.
 */
static krb5_error_code
krb5_context_externalize(krb5_context kcontext, krb5_pointer arg, krb5_octet **buffer, size_t *lenremain)
{
    krb5_error_code	kret;
    krb5_context	context;
    size_t		required;
    krb5_octet		*bp;
    size_t		remain;
    int			i;

    required = 0;
    bp = *buffer;
    remain = *lenremain;
    context = (krb5_context) arg;
    if (!context)
	    return (EINVAL);
    KRB5_VERIFY_MAGIC(context, KV5M_CONTEXT);

    if ((kret = krb5_context_size(kcontext, arg, &required)))
	return (kret);

    if (required > remain)
	return (ENOMEM);

    /* First write our magic number */
    kret = krb5_ser_pack_int32(KV5M_CONTEXT, &bp, &remain);
    if (kret)
	return (kret);

    /* Now sizeof default realm */
    kret = krb5_ser_pack_int32((context->default_realm) ?
			       (krb5_int32) strlen(context->default_realm) : 0,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now default_realm bytes */
    if (context->default_realm) {
	kret = krb5_ser_pack_bytes((krb5_octet *) context->default_realm,
				   strlen(context->default_realm),
				   &bp, &remain);
	if (kret)
	    return (kret);
    }

    /* Now number of initial ticket ktypes */
    kret = krb5_ser_pack_int32((krb5_int32) context->in_tkt_ktype_count,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now serialize ktypes */
    for (i=0; i<context->in_tkt_ktype_count; i++) {
	kret = krb5_ser_pack_int32((krb5_int32) context->in_tkt_ktypes[i],
				   &bp, &remain);
	if (kret)
	    return (kret);
    }

    /* Now number of default ktypes */
    kret = krb5_ser_pack_int32((krb5_int32) context->tgs_ktype_count,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now serialize ktypes */
    for (i=0; i<context->tgs_ktype_count; i++) {
	kret = krb5_ser_pack_int32((krb5_int32) context->tgs_ktypes[i],
				   &bp, &remain);
	if (kret)
	    return (kret);
    }

    /* Now allowable clockskew */
    kret = krb5_ser_pack_int32((krb5_int32) context->clockskew,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now kdc_req_sumtype */
    kret = krb5_ser_pack_int32((krb5_int32) context->kdc_req_sumtype,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now default ap_req_sumtype */
    kret = krb5_ser_pack_int32((krb5_int32) context->default_ap_req_sumtype,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now default safe_sumtype */
    kret = krb5_ser_pack_int32((krb5_int32) context->default_safe_sumtype,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now kdc_default_options */
    kret = krb5_ser_pack_int32((krb5_int32) context->kdc_default_options,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now library_options */
    kret = krb5_ser_pack_int32((krb5_int32) context->library_options,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now profile_secure */
    kret = krb5_ser_pack_int32((krb5_int32) context->profile_secure,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now fcc_default_format */
    kret = krb5_ser_pack_int32((krb5_int32) context->fcc_default_format,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now scc_default_format */
    kret = krb5_ser_pack_int32((krb5_int32) context->scc_default_format,
			       &bp, &remain);
    if (kret)
	return (kret);

    /* Now handle os_context, if appropriate */
    if (context->os_context) {
	kret = krb5_externalize_opaque(kcontext, KV5M_OS_CONTEXT,
				       (krb5_pointer) context->os_context,
				       &bp, &remain);
	if (kret)
	    return (kret);
    }

    /* Now handle database context, if appropriate */
    if (context->db_context) {
	kret = krb5_externalize_opaque(kcontext, KV5M_DB_CONTEXT,
				       (krb5_pointer) context->db_context,
				       &bp, &remain);
	if (kret)
	    return (kret);
    }

    /* Finally, handle profile, if appropriate */
    if (context->profile) {
	kret = krb5_externalize_opaque(kcontext, PROF_MAGIC_PROFILE,
				       (krb5_pointer) context->profile,
				       &bp, &remain);
	if (kret)
	    return (kret);
    }

    /*
     * If we were successful, write trailer then update the pointer and
     * remaining length;
     */
    kret = krb5_ser_pack_int32(KV5M_CONTEXT, &bp, &remain);
    if (kret)
	return (kret);

    *buffer = bp;
    *lenremain = remain;

    return (0);
}

/*
 * krb5_context_internalize()	- Internalize the krb5_context.
 */
static krb5_error_code
krb5_context_internalize(krb5_context kcontext, krb5_pointer *argp, krb5_octet **buffer, size_t *lenremain)
{
    krb5_error_code	kret;
    krb5_context	context;
    krb5_int32		ibuf;
    krb5_octet		*bp;
    size_t		remain;
    int			i;

    bp = *buffer;
    remain = *lenremain;

    /* Read our magic number */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	return (EINVAL);

    if (ibuf != KV5M_CONTEXT)
	return (EINVAL);

    /* Get memory for the context */
    context = (krb5_context) MALLOC(sizeof(struct _krb5_context));
    if (!context)
	return (ENOMEM);
    (void) memset(context, 0, sizeof(struct _krb5_context));

    /* Get the size of the default realm */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;

    if (ibuf) {
	context->default_realm = (char *) MALLOC((size_t) ibuf+1);
	if (!context->default_realm) {
	    kret = ENOMEM;
	    goto cleanup;
	}

	kret = krb5_ser_unpack_bytes((krb5_octet *) context->default_realm,
				     (size_t) ibuf, &bp, &remain);
	if (kret)
	    goto cleanup;

	context->default_realm[ibuf] = '\0';
    }

    /* Get the number of in_tkt_ktypes */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;

    context->in_tkt_ktype_count = (int) ibuf;
    context->in_tkt_ktypes = (krb5_enctype *) MALLOC(sizeof(krb5_enctype) *
				     (context->in_tkt_ktype_count+1));
    if (!context->in_tkt_ktypes) {
	kret = ENOMEM;
	goto cleanup;
    }
    (void) memset(context->in_tkt_ktypes, 0, (sizeof(krb5_enctype) *
				       (context->in_tkt_ktype_count + 1)));

    for (i=0; i<context->in_tkt_ktype_count; i++) {
	if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	    goto cleanup;
	context->in_tkt_ktypes[i] = (krb5_enctype) ibuf;
    }

    /* Get the number of tgs_ktypes */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;

    context->tgs_ktype_count = (int) ibuf;
    context->tgs_ktypes = (krb5_enctype *) MALLOC(sizeof(krb5_enctype) *
				  (context->tgs_ktype_count+1));
    if (!context->tgs_ktypes) {
	kret = ENOMEM;
	goto cleanup;
    }
    (void) memset(context->tgs_ktypes, 0, (sizeof(krb5_enctype) *
				    (context->tgs_ktype_count + 1)));
    for (i=0; i<context->tgs_ktype_count; i++) {
	if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	    goto cleanup;
	context->tgs_ktypes[i] = (krb5_enctype) ibuf;
    }

    /* Allowable checksum */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->clockskew = (krb5_deltat) ibuf;

    /* kdc_req_sumtype */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->kdc_req_sumtype = (krb5_cksumtype) ibuf;

    /* default ap_req_sumtype */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->default_ap_req_sumtype = (krb5_cksumtype) ibuf;

    /* default_safe_sumtype */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->default_safe_sumtype = (krb5_cksumtype) ibuf;

    /* kdc_default_options */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->kdc_default_options = (krb5_flags) ibuf;

    /* library_options */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->library_options = (krb5_flags) ibuf;

    /* profile_secure */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->profile_secure = (krb5_boolean) ibuf;

    /* fcc_default_format */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->fcc_default_format = (int) ibuf;

    /* scc_default_format */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;
    context->scc_default_format = (int) ibuf;

    /* Attempt to read in the os_context.  It's an array now, but
       we still treat it in most places as a separate object with
       a pointer.  */
    {
	krb5_os_context osp = 0;
	kret = krb5_internalize_opaque(kcontext, KV5M_OS_CONTEXT,
				       (krb5_pointer *) &osp,
				       &bp, &remain);
	if (kret && (kret != EINVAL) && (kret != ENOENT))
	    goto cleanup;
	/* Put the newly allocated data into the krb5_context
	   structure where we're really keeping it these days.  */
	if (osp)
	    *context->os_context = *osp;
	free(osp);
    }

    /* Attempt to read in the db_context */
    kret = krb5_internalize_opaque(kcontext, KV5M_DB_CONTEXT,
				   (krb5_pointer *) &context->db_context,
				   &bp, &remain);
    if (kret && (kret != EINVAL) && (kret != ENOENT))
	goto cleanup;

#ifndef _KERNEL
    /* Attempt to read in the profile */
    kret = krb5_internalize_opaque(kcontext, PROF_MAGIC_PROFILE,
				   (krb5_pointer *) &context->profile,
				   &bp, &remain);
#endif
    if (kret && (kret != EINVAL) && (kret != ENOENT))
	goto cleanup;

    /* Finally, find the trailer */
    if ((kret = krb5_ser_unpack_int32(&ibuf, &bp, &remain)))
	goto cleanup;

    if (ibuf != KV5M_CONTEXT) {
	kret = EINVAL;
	goto cleanup;
    }

    context->magic = KV5M_CONTEXT;
    *buffer = bp;
    *lenremain = remain;
    *argp = (krb5_pointer) context;

    return 0;

cleanup:
    if (context)
	krb5_free_context(context);
    return(kret);
}

/*
 * krb5_oscontext_size()	- Determine the size required to externalize
 *				  the krb5_os_context.
 */
/*ARGSUSED*/
static krb5_error_code
krb5_oscontext_size(krb5_context kcontext, krb5_pointer arg, size_t *sizep)
{
    /*
     * We need five 32-bit integers:
     *	two for header and trailer
     *	one each for time_offset, usec_offset and os_flags
     */
    *sizep += (5*sizeof(krb5_int32));
    return(0);
}

/*
 * krb5_oscontext_externalize()	- Externalize the krb5_os_context.
 */
static krb5_error_code
krb5_oscontext_externalize(krb5_context kcontext, krb5_pointer arg, krb5_octet **buffer, size_t *lenremain)
{
    krb5_error_code	kret;
    krb5_os_context	os_ctx;
    size_t		required;
    krb5_octet		*bp;
    size_t		remain;

    required = 0;
    bp = *buffer;
    remain = *lenremain;
    kret = EINVAL;
    if ((os_ctx = (krb5_os_context) arg)) {
	kret = ENOMEM;
	if (!krb5_oscontext_size(kcontext, arg, &required) &&
	    (required <= remain)) {
	    (void) krb5_ser_pack_int32(KV5M_OS_CONTEXT, &bp, &remain);
	    (void) krb5_ser_pack_int32(os_ctx->time_offset, &bp, &remain);
	    (void) krb5_ser_pack_int32(os_ctx->usec_offset, &bp, &remain);
	    (void) krb5_ser_pack_int32(os_ctx->os_flags, &bp, &remain);
	    (void) krb5_ser_pack_int32(KV5M_OS_CONTEXT, &bp, &remain);

	    /* Handle any other OS context here */
	    kret = 0;
	    if (!kret) {
		*buffer = bp;
		*lenremain = remain;
	    }
	}
    }
    return(kret);
}

/*
 * krb5_oscontext_internalize()	- Internalize the krb5_os_context.
 */
/*ARGSUSED*/
static krb5_error_code
krb5_oscontext_internalize(krb5_context kcontext, krb5_pointer *argp, krb5_octet **buffer, size_t *lenremain)
{
    krb5_error_code	kret;
    krb5_os_context	os_ctx;
    krb5_int32		ibuf;
    krb5_octet		*bp;
    size_t		remain;

    bp = *buffer;
    remain = *lenremain;
    kret = EINVAL;
    os_ctx = (krb5_os_context) NULL;
    /* Read our magic number */
    if (krb5_ser_unpack_int32(&ibuf, &bp, &remain))
	ibuf = 0;
    if (ibuf == KV5M_OS_CONTEXT) {
	kret = ENOMEM;

	/* Get memory for the context */
	if ((os_ctx = (krb5_os_context)
	     MALLOC(sizeof(struct _krb5_os_context))) &&
	    (remain >= 4*sizeof(krb5_int32))) {
	    (void) memset(os_ctx, 0, sizeof(struct _krb5_os_context));
	    os_ctx->magic = KV5M_OS_CONTEXT;

	    /* Read out our context */
	    (void) krb5_ser_unpack_int32(&os_ctx->time_offset, &bp, &remain);
	    (void) krb5_ser_unpack_int32(&os_ctx->usec_offset, &bp, &remain);
	    (void) krb5_ser_unpack_int32(&os_ctx->os_flags, &bp, &remain);
	    (void) krb5_ser_unpack_int32(&ibuf, &bp, &remain);

	    if (ibuf == KV5M_OS_CONTEXT) {
		os_ctx->magic = KV5M_OS_CONTEXT;
		kret = 0;
		*buffer = bp;
		*lenremain = remain;
	    } else
		kret = EINVAL;
	}
    }
    if (!kret) {
	*argp = (krb5_pointer) os_ctx;
    }
    else {
	if (os_ctx)
	    FREE(os_ctx, sizeof(struct _krb5_os_context));
    }
    return(kret);
}

/*
 * Register the context serializers.
 */
krb5_error_code KRB5_CALLCONV
krb5_ser_context_init(krb5_context kcontext)
{
    krb5_error_code	kret;
    kret = krb5_register_serializer(kcontext, &krb5_context_ser_entry);
    if (!kret)
	kret = krb5_register_serializer(kcontext, &krb5_oscontext_ser_entry);

    /* Profile nformation need not be serialzied when we are importing the
     * context into kernel. Besides the function pointers to file access
     * routines can't be used in the kernel.

     * Any info needed from the profile is already a part of the
     * exported context obviating the need for importer to know about
     * profile config files.

    */

#ifndef _KERNEL
    if (!kret)
	kret = krb5_register_serializer(kcontext, &krb5_profile_ser_entry);
#endif
    return(kret);
}
