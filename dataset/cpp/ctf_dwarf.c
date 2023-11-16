/*
 * CDDL HEADER START
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can obtain a copy of the license at usr/src/OPENSOLARIS.LICENSE
 * or http://www.opensolaris.org/os/licensing.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL HEADER in each
 * file and include the License file at usr/src/OPENSOLARIS.LICENSE.
 * If applicable, add the following below this CDDL HEADER, with the
 * fields enclosed by brackets "[]" replaced with your own identifying
 * information: Portions Copyright [yyyy] [name of copyright owner]
 *
 * CDDL HEADER END
 */
/*
 * Copyright 2007 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */
/*
 * Copyright 2012 Jason King.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Copyright 2020 Joyent, Inc.
 * Copyright 2020 Robert Mustacchi
 * Copyright 2021 OmniOS Community Edition (OmniOSce) Association.
 */

/*
 * CTF DWARF conversion theory.
 *
 * DWARF data contains a series of compilation units. Each compilation unit
 * generally refers to an object file or what once was, in the case of linked
 * binaries and shared objects. Each compilation unit has a series of what DWARF
 * calls a DIE (Debugging Information Entry). The set of entries that we care
 * about have type information stored in a series of attributes. Each DIE also
 * has a tag that identifies the kind of attributes that it has.
 *
 * A given DIE may itself have children. For example, a DIE that represents a
 * structure has children which represent members. Whenever we encounter a DIE
 * that has children or other values or types associated with it, we recursively
 * process those children first so that way we can then refer to the generated
 * CTF type id while processing its parent. This reduces the amount of unknowns
 * and fixups that we need. It also ensures that we don't accidentally add types
 * that an overzealous compiler might add to the DWARF data but aren't used by
 * anything in the system.
 *
 * Once we do a conversion, we store a mapping in an AVL tree that goes from the
 * DWARF's die offset, which is relative to the given compilation unit, to a
 * ctf_id_t.
 *
 * Unfortunately, some compilers actually will emit duplicate entries for a
 * given type that look similar, but aren't quite. To that end, we go through
 * and do a variant on a merge once we're done processing a single compilation
 * unit which deduplicates all of the types that are in the unit.
 *
 * Finally, if we encounter an object that has multiple compilation units, then
 * we'll convert all of the compilation units separately and then do a merge, so
 * that way we can result in one single ctf_file_t that represents everything
 * for the object.
 *
 * Conversion Steps
 * ----------------
 *
 * Because a given object we've been given to convert may have multiple
 * compilation units, we break the work into two halves. The first half
 * processes each compilation unit (potentially in parallel) and then the second
 * half optionally merges all of the dies in the first half. First, we'll cover
 * what's involved in converting a single ctf_cu_t's dwarf to CTF. This covers
 * the work done in ctf_dwarf_convert_one().
 *
 * An individual ctf_cu_t, which represents a compilation unit, is converted to
 * CTF in a series of multiple passes.
 *
 * Pass 1: During the first pass we walk all of the top-level dies and if we
 * find a function, variable, struct, union, enum or typedef, we recursively
 * transform all of its types. We don't recurse or process everything, because
 * we don't want to add some of the types that compilers may add which are
 * effectively unused.
 *
 * During pass 1, if we encounter any structures or unions we mark them for
 * fixing up later. This is necessary because we may not be able to determine
 * the full size of a structure at the beginning of time. This will happen if
 * the DWARF attribute DW_AT_byte_size is not present for a member. Because of
 * this possibility we defer adding members to structures or even converting
 * them during pass 1 and save that for pass 2. Adding all of the base
 * structures without any of their members helps deal with any circular
 * dependencies that we might encounter.
 *
 * Pass 2: This pass is used to do the first half of fixing up structures and
 * unions. Rather than walk the entire type space again, we actually walk the
 * list of structures and unions that we marked for later fixing up. Here, we
 * iterate over every structure and add members to the underlying ctf_file_t,
 * but not to the structs themselves. One might wonder why we don't, and the
 * main reason is that libctf requires a ctf_update() be done before adding the
 * members to structures or unions.
 *
 * Pass 3: This pass is used to do the second half of fixing up structures and
 * unions. During this part we always go through and add members to structures
 * and unions that we added to the container in the previous pass. In addition,
 * we set the structure and union's actual size, which may have additional
 * padding added by the compiler, it isn't simply the last offset. DWARF always
 * guarantees an attribute exists for this. Importantly no ctf_id_t's change
 * during pass 2.
 *
 * Pass 4: The next phase is to add CTF entries for all of the symbols and
 * variables that are present in this die. During pass 1 we added entries to a
 * map for each variable and function. During this pass, we iterate over the
 * symbol table and when we encounter a symbol that we have in our lists of
 * translated information which matches, we then add it to the ctf_file_t.
 *
 * Pass 5: Here we go and look for any weak symbols and functions and see if
 * they match anything that we recognize. If so, then we add type information
 * for them at this point based on the matching type.
 *
 * Pass 6: This pass is actually a variant on a merge. The traditional merge
 * process expects there to be no duplicate types. As such, at the end of
 * conversion, we do a dedup on all of the types in the system. The
 * deduplication process is described in lib/libctf/common/ctf_merge.c.
 *
 * Once pass 6 is done, we've finished processing the individual compilation
 * unit.
 *
 * The following steps reflect the general process of doing a conversion.
 *
 * 1) Walk the dwarf section and determine the number of compilation units
 * 2) Create a ctf_cu_t for each compilation unit
 * 3) Add all ctf_cu_t's to a workq
 * 4) Have the workq process each die with ctf_dwarf_convert_one. This itself
 *    is comprised of several steps, which were already enumerated.
 * 5) If we have multiple cu's, we do a ctf merge of all the dies. The mechanics
 *    of the merge are discussed in lib/libctf/common/ctf_merge.c.
 * 6) Free everything up and return a ctf_file_t to the user. If we only had a
 *    single compilation unit, then we give that to the user. Otherwise, we
 *    return the merged ctf_file_t.
 *
 * Threading
 * ---------
 *
 * The process has been designed to be amenable to threading. Each compilation
 * unit has its own type stream, therefore the logical place to divide and
 * conquer is at the compilation unit. Each ctf_cu_t has been built to be able
 * to be processed independently of the others. It has its own libdwarf handle,
 * as a given libdwarf handle may only be used by a single thread at a time.
 * This allows the various ctf_cu_t's to be processed in parallel by different
 * threads.
 *
 * All of the ctf_cu_t's are loaded into a workq which allows for a number of
 * threads to be specified and used as a thread pool to process all of the
 * queued work. We set the number of threads to use in the workq equal to the
 * number of threads that the user has specified.
 *
 * After all of the compilation units have been drained, we use the same number
 * of threads when performing a merge of multiple compilation units, if they
 * exist.
 *
 * While all of these different parts do support and allow for multiple threads,
 * it's important that when only a single thread is specified, that it be the
 * calling thread. This allows the conversion routines to be used in a context
 * that doesn't allow additional threads, such as rtld.
 *
 * Common DWARF Mechanics and Notes
 * --------------------------------
 *
 * At this time, we really only support DWARFv2, though support for DWARFv4 is
 * mostly there. There is no intent to support DWARFv3.
 *
 * Generally types for something are stored in the DW_AT_type attribute. For
 * example, a function's return type will be stored in the local DW_AT_type
 * attribute while the arguments will be in child DIEs. There are also various
 * times when we don't have any DW_AT_type. In that case, the lack of a type
 * implies, at least for C, that its C type is void. Because DWARF doesn't emit
 * one, we have a synthetic void type that we create and manipulate instead and
 * pass it off to consumers on an as-needed basis. If nothing has a void type,
 * it will not be emitted.
 *
 * Architecture Specific Parts
 * ---------------------------
 *
 * The CTF tooling encodes various information about the various architectures
 * in the system. Importantly, the tool assumes that every architecture has a
 * data model where long and pointer are the same size. This is currently the
 * case, as the two data models illumos supports are ILP32 and LP64.
 *
 * In addition, we encode the mapping of various floating point sizes to various
 * types for each architecture. If a new architecture is being added, it should
 * be added to the list. The general design of the ctf conversion tools is to be
 * architecture independent. eg. any of the tools here should be able to convert
 * any architecture's DWARF into ctf; however, this has not been rigorously
 * tested and more importantly, the ctf routines don't currently write out the
 * data in an endian-aware form, they only use that of the currently running
 * library.
 */

#include <libctf_impl.h>
#include <sys/avl.h>
#include <sys/debug.h>
#include <sys/list.h>
#include <gelf.h>
#include <libdwarf.h>
#include <dwarf.h>
#include <libgen.h>
#include <workq.h>
#include <thread.h>
#include <macros.h>
#include <errno.h>

#define	DWARF_VERSION_TWO	2
#define	DWARF_VERSION_FOUR	4
#define	DWARF_VARARGS_NAME	"..."

/*
 * Dwarf may refer recursively to other types that we've already processed. To
 * see if we've already converted them, we look them up in an AVL tree that's
 * sorted by the DWARF id.
 */
typedef struct ctf_dwmap {
	avl_node_t	cdm_avl;
	Dwarf_Off	cdm_off;
	Dwarf_Die	cdm_die;
	ctf_id_t	cdm_id;
	boolean_t	cdm_fix;
} ctf_dwmap_t;

typedef struct ctf_dwvar {
	ctf_list_t	cdv_list;
	char		*cdv_name;
	ctf_id_t	cdv_type;
	boolean_t	cdv_global;
} ctf_dwvar_t;

typedef struct ctf_dwfunc {
	ctf_list_t	cdf_list;
	char		*cdf_name;
	ctf_funcinfo_t	cdf_fip;
	ctf_id_t	*cdf_argv;
	boolean_t	cdf_global;
} ctf_dwfunc_t;

typedef struct ctf_dwbitf {
	ctf_list_t	cdb_list;
	ctf_id_t	cdb_base;
	uint_t		cdb_nbits;
	ctf_id_t	cdb_id;
} ctf_dwbitf_t;

/*
 * The ctf_cu_t represents a single top-level DWARF die unit. While generally,
 * the typical object file has only a single die, if we're asked to convert
 * something that's been linked from multiple sources, multiple dies will exist.
 */
typedef struct ctf_die {
	Elf		*cu_elf;	/* shared libelf handle */
	int		cu_fd;		/* shared file descriptor */
	char		*cu_name;	/* basename of the DIE */
	ctf_merge_t	*cu_cmh;	/* merge handle */
	ctf_list_t	cu_vars;	/* List of variables */
	ctf_list_t	cu_funcs;	/* List of functions */
	ctf_list_t	cu_bitfields;	/* Bit field members */
	Dwarf_Debug	cu_dwarf;	/* libdwarf handle */
	mutex_t		*cu_dwlock;	/* libdwarf lock */
	Dwarf_Die	cu_cu;		/* libdwarf compilation unit */
	Dwarf_Off	cu_cuoff;	/* cu's offset */
	Dwarf_Off	cu_maxoff;	/* maximum offset */
	Dwarf_Half	cu_vers;	/* Dwarf Version */
	Dwarf_Half	cu_addrsz;	/* Dwarf Address Size */
	ctf_file_t	*cu_ctfp;	/* output CTF file */
	avl_tree_t	cu_map;		/* map die offsets to CTF types */
	char		*cu_errbuf;	/* error message buffer */
	size_t		cu_errlen;	/* error message buffer length */
	ctf_convert_t	*cu_handle;	/* ctf convert handle */
	size_t		cu_ptrsz;	/* object's pointer size */
	boolean_t	cu_bigend;	/* is it big endian */
	boolean_t	cu_doweaks;	/* should we convert weak symbols? */
	uint_t		cu_mach;	/* machine type */
	ctf_id_t	cu_voidtid;	/* void pointer */
	ctf_id_t	cu_longtid;	/* id for a 'long' */
} ctf_cu_t;

static int ctf_dwarf_init_die(ctf_cu_t *);
static int ctf_dwarf_offset(ctf_cu_t *, Dwarf_Die, Dwarf_Off *);
static int ctf_dwarf_convert_die(ctf_cu_t *, Dwarf_Die);
static int ctf_dwarf_convert_type(ctf_cu_t *, Dwarf_Die, ctf_id_t *, int);

static int ctf_dwarf_function_count(ctf_cu_t *, Dwarf_Die, ctf_funcinfo_t *,
    boolean_t);
static int ctf_dwarf_convert_fargs(ctf_cu_t *, Dwarf_Die, ctf_funcinfo_t *,
    ctf_id_t *);

#define	DWARF_LOCK(cup) \
	if ((cup)->cu_dwlock != NULL) \
		mutex_enter((cup)->cu_dwlock)
#define	DWARF_UNLOCK(cup) \
	if ((cup)->cu_dwlock != NULL) \
		mutex_exit((cup)->cu_dwlock)

/*
 * This is a generic way to set a CTF Conversion backend error depending on what
 * we were doing. Unless it was one of a specific set of errors that don't
 * indicate a programming / translation bug, eg. ENOMEM, then we transform it
 * into a CTF backend error and fill in the error buffer.
 */
static int
ctf_dwarf_error(ctf_cu_t *cup, ctf_file_t *cfp, int err, const char *fmt, ...)
{
	va_list ap;
	int ret;
	size_t off = 0;
	ssize_t rem = cup->cu_errlen;
	if (cfp != NULL)
		err = ctf_errno(cfp);

	if (err == ENOMEM)
		return (err);

	ret = snprintf(cup->cu_errbuf, rem, "die %s: ",
	    cup->cu_name != NULL ? cup->cu_name : "NULL");
	if (ret < 0)
		goto err;
	off += ret;
	rem = MAX(rem - ret, 0);

	va_start(ap, fmt);
	ret = vsnprintf(cup->cu_errbuf + off, rem, fmt, ap);
	va_end(ap);
	if (ret < 0)
		goto err;

	off += ret;
	rem = MAX(rem - ret, 0);
	if (fmt[strlen(fmt) - 1] != '\n') {
		(void) snprintf(cup->cu_errbuf + off, rem,
		    ": %s\n", ctf_errmsg(err));
	}
	va_end(ap);
	return (ECTF_CONVBKERR);

err:
	cup->cu_errbuf[0] = '\0';
	return (ECTF_CONVBKERR);
}

/*
 * DWARF often opts to put no explicit type to describe a void type. eg. if we
 * have a reference type whose DW_AT_type member doesn't exist, then we should
 * instead assume it points to void. Because this isn't represented, we
 * instead cause it to come into existence.
 */
static ctf_id_t
ctf_dwarf_void(ctf_cu_t *cup)
{
	if (cup->cu_voidtid == CTF_ERR) {
		ctf_encoding_t enc = { CTF_INT_SIGNED, 0, 0 };
		cup->cu_voidtid = ctf_add_integer(cup->cu_ctfp, CTF_ADD_ROOT,
		    "void", &enc);
		if (cup->cu_voidtid == CTF_ERR) {
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "failed to create void type: %s\n",
			    ctf_errmsg(ctf_errno(cup->cu_ctfp)));
		}
	}

	return (cup->cu_voidtid);
}

/*
 * There are many different forms that an array index may take. However, we just
 * always force it to be of a type long no matter what. Therefore we use this to
 * have a single instance of long across everything.
 */
static ctf_id_t
ctf_dwarf_long(ctf_cu_t *cup)
{
	if (cup->cu_longtid == CTF_ERR) {
		ctf_encoding_t enc;

		enc.cte_format = CTF_INT_SIGNED;
		enc.cte_offset = 0;
		/* All illumos systems are LP */
		enc.cte_bits = cup->cu_ptrsz * 8;
		cup->cu_longtid = ctf_add_integer(cup->cu_ctfp, CTF_ADD_NONROOT,
		    "long", &enc);
		if (cup->cu_longtid == CTF_ERR) {
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "failed to create long type: %s\n",
			    ctf_errmsg(ctf_errno(cup->cu_ctfp)));
		}

	}

	return (cup->cu_longtid);
}

static int
ctf_dwmap_comp(const void *a, const void *b)
{
	const ctf_dwmap_t *ca = a;
	const ctf_dwmap_t *cb = b;

	if (ca->cdm_off > cb->cdm_off)
		return (1);
	if (ca->cdm_off < cb->cdm_off)
		return (-1);
	return (0);
}

static int
ctf_dwmap_add(ctf_cu_t *cup, ctf_id_t id, Dwarf_Die die, boolean_t fix)
{
	int ret;
	avl_index_t index;
	ctf_dwmap_t *dwmap;
	Dwarf_Off off;

	VERIFY(id > 0 && id < CTF_MAX_TYPE);

	if ((ret = ctf_dwarf_offset(cup, die, &off)) != 0)
		return (ret);

	if ((dwmap = ctf_alloc(sizeof (ctf_dwmap_t))) == NULL)
		return (ENOMEM);

	dwmap->cdm_die = die;
	dwmap->cdm_off = off;
	dwmap->cdm_id = id;
	dwmap->cdm_fix = fix;

	ctf_dprintf("dwmap: %p %" DW_PR_DUx "->%d\n", dwmap, off, id);
	VERIFY(avl_find(&cup->cu_map, dwmap, &index) == NULL);
	avl_insert(&cup->cu_map, dwmap, index);
	return (0);
}

static int
ctf_dwarf_attribute(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name,
    Dwarf_Attribute *attrp)
{
	int ret;
	Dwarf_Error derr;

	DWARF_LOCK(cup);
	ret = dwarf_attr(die, name, attrp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK)
		return (0);
	if (ret == DW_DLV_NO_ENTRY) {
		*attrp = NULL;
		return (ENOENT);
	}
	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get attribute for type: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static void
ctf_dwarf_dealloc(ctf_cu_t *cup, Dwarf_Ptr ptr, Dwarf_Unsigned type)
{
	DWARF_LOCK(cup);
	dwarf_dealloc(cup->cu_dwarf, ptr, type);
	DWARF_UNLOCK(cup);
}

static int
ctf_dwarf_ref(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name, Dwarf_Off *refp)
{
	int ret;
	Dwarf_Attribute attr;
	Dwarf_Error derr;

	if ((ret = ctf_dwarf_attribute(cup, die, name, &attr)) != 0)
		return (ret);

	DWARF_LOCK(cup);
	ret = dwarf_formref(attr, refp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK) {
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (0);
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get attribute descriptor offset: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_refdie(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name,
    Dwarf_Die *diep)
{
	int ret;
	Dwarf_Off off;
	Dwarf_Error derr;

	if ((ret = ctf_dwarf_ref(cup, die, name, &off)) != 0)
		return (ret);

	off += cup->cu_cuoff;
	DWARF_LOCK(cup);
	ret = dwarf_offdie(cup->cu_dwarf, off, diep, &derr);
	DWARF_UNLOCK(cup);
	if (ret != DW_DLV_OK) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to get die from offset %" DW_PR_DUu ": %s\n",
		    off, dwarf_errmsg(derr));
		return (ECTF_CONVBKERR);
	}

	return (0);
}

static int
ctf_dwarf_signed(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name,
    Dwarf_Signed *valp)
{
	int ret;
	Dwarf_Attribute attr;
	Dwarf_Error derr;

	if ((ret = ctf_dwarf_attribute(cup, die, name, &attr)) != 0)
		return (ret);

	DWARF_LOCK(cup);
	ret = dwarf_formsdata(attr, valp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK) {
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (0);
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get signed attribute for type: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_unsigned(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name,
    Dwarf_Unsigned *valp)
{
	int ret;
	Dwarf_Attribute attr;
	Dwarf_Error derr;

	if ((ret = ctf_dwarf_attribute(cup, die, name, &attr)) != 0)
		return (ret);

	DWARF_LOCK(cup);
	ret = dwarf_formudata(attr, valp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK) {
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (0);
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get unsigned attribute for type: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_boolean(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name,
    Dwarf_Bool *val)
{
	int ret;
	Dwarf_Attribute attr;
	Dwarf_Error derr;

	if ((ret = ctf_dwarf_attribute(cup, die, name, &attr)) != 0)
		return (ret);

	DWARF_LOCK(cup);
	ret = dwarf_formflag(attr, val, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK) {
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (0);
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get boolean attribute for type: %s\n",
	    dwarf_errmsg(derr));

	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_string(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half name, char **strp)
{
	int ret;
	char *s;
	Dwarf_Attribute attr;
	Dwarf_Error derr;

	*strp = NULL;
	if ((ret = ctf_dwarf_attribute(cup, die, name, &attr)) != 0)
		return (ret);

	DWARF_LOCK(cup);
	ret = dwarf_formstring(attr, &s, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK) {
		if ((*strp = ctf_strdup(s)) == NULL)
			ret = ENOMEM;
		else
			ret = 0;
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (ret);
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get string attribute for type: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

/*
 * The encoding of a DW_AT_data_member_location has changed between different
 * revisions of the specification. It may be a general udata form or it may be
 * location data information. In DWARF 2, it is only the latter. In later
 * revisions of the spec, it may be either. To determine the form, we ask the
 * class, which will be of type CONSTANT.
 */
static int
ctf_dwarf_member_location(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Unsigned *valp)
{
	int ret;
	Dwarf_Error derr;
	Dwarf_Attribute attr;
	Dwarf_Locdesc *loc;
	Dwarf_Signed locnum;
	Dwarf_Half form;
	enum Dwarf_Form_Class class;

	if ((ret = ctf_dwarf_attribute(cup, die, DW_AT_data_member_location,
	    &attr)) != 0) {
		return (ret);
	}

	DWARF_LOCK(cup);
	ret = dwarf_whatform(attr, &form, &derr);
	DWARF_UNLOCK(cup);
	if (ret != DW_DLV_OK) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to get dwarf attribute for for member "
		    "location: %s\n",
		    dwarf_errmsg(derr));
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (ECTF_CONVBKERR);
	}

	DWARF_LOCK(cup);
	class = dwarf_get_form_class(cup->cu_vers, DW_AT_data_member_location,
	    cup->cu_addrsz, form);
	if (class == DW_FORM_CLASS_CONSTANT) {
		Dwarf_Signed sign;

		/*
		 * We have a constant. We need to try to get both this as signed
		 * and unsigned data, as unfortunately, DWARF doesn't define the
		 * sign. Which is a joy. We try unsigned first. If neither
		 * match, fall through to the normal path.
		 */
		if (dwarf_formudata(attr, valp, &derr) == DW_DLV_OK) {
			DWARF_UNLOCK(cup);
			ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
			return (0);
		}

		if (dwarf_formsdata(attr, &sign, &derr) == DW_DLV_OK) {
			DWARF_UNLOCK(cup);
			ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
			if (sign < 0) {
				(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
				    "encountered negative member data "
				    "location: %lld\n", sign);
			}
			*valp = (Dwarf_Unsigned)sign;
			return (0);
		}
	}

	if (dwarf_loclist(attr, &loc, &locnum, &derr) != DW_DLV_OK) {
		DWARF_UNLOCK(cup);
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to obtain location list for member offset: %s\n",
		    dwarf_errmsg(derr));
		ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
		return (ECTF_CONVBKERR);
	}
	DWARF_UNLOCK(cup);
	ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);

	if (locnum != 1 || loc->ld_s->lr_atom != DW_OP_plus_uconst) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to parse location structure for member\n");
		ctf_dwarf_dealloc(cup, loc->ld_s, DW_DLA_LOC_BLOCK);
		ctf_dwarf_dealloc(cup, loc, DW_DLA_LOCDESC);
		return (ECTF_CONVBKERR);
	}

	*valp = loc->ld_s->lr_number;

	ctf_dwarf_dealloc(cup, loc->ld_s, DW_DLA_LOC_BLOCK);
	ctf_dwarf_dealloc(cup, loc, DW_DLA_LOCDESC);
	return (0);
}


static int
ctf_dwarf_offset(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Off *offsetp)
{
	Dwarf_Error derr;
	int ret;

	DWARF_LOCK(cup);
	ret = dwarf_dieoffset(die, offsetp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK)
		return (0);

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get die offset: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

/* simpler variant for debugging output */
static Dwarf_Off
ctf_die_offset(ctf_cu_t *cup, Dwarf_Die die)
{
	Dwarf_Off off = -1;
	Dwarf_Error derr;

	DWARF_LOCK(cup);
	(void) dwarf_dieoffset(die, &off, &derr);
	DWARF_UNLOCK(cup);
	return (off);
}

static int
ctf_dwarf_tag(ctf_cu_t *cup, Dwarf_Die die, Dwarf_Half *tagp)
{
	Dwarf_Error derr;
	int ret;

	DWARF_LOCK(cup);
	ret = dwarf_tag(die, tagp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK)
		return (0);

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get tag type: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_sib(ctf_cu_t *cup, Dwarf_Die base, Dwarf_Die *sibp)
{
	Dwarf_Error derr;
	int ret;

	*sibp = NULL;
	DWARF_LOCK(cup);
	ret = dwarf_siblingof(cup->cu_dwarf, base, sibp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK || ret == DW_DLV_NO_ENTRY)
		return (0);

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to sibling from die: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

static int
ctf_dwarf_child(ctf_cu_t *cup, Dwarf_Die base, Dwarf_Die *childp)
{
	Dwarf_Error derr;
	int ret;

	*childp = NULL;
	DWARF_LOCK(cup);
	ret = dwarf_child(base, childp, &derr);
	DWARF_UNLOCK(cup);
	if (ret == DW_DLV_OK || ret == DW_DLV_NO_ENTRY)
		return (0);

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to child from die: %s\n",
	    dwarf_errmsg(derr));
	return (ECTF_CONVBKERR);
}

/*
 * Compilers disagree on what to do to determine if something has global
 * visiblity. Traditionally gcc has used DW_AT_external to indicate this while
 * Studio has used DW_AT_visibility. We check DW_AT_visibility first and then
 * fall back to DW_AT_external. Lack of DW_AT_external implies that it is not.
 */
static int
ctf_dwarf_isglobal(ctf_cu_t *cup, Dwarf_Die die, boolean_t *igp)
{
	int ret;
	Dwarf_Signed vis;
	Dwarf_Bool ext;

	if ((ret = ctf_dwarf_signed(cup, die, DW_AT_visibility, &vis)) == 0) {
		*igp = vis == DW_VIS_exported;
		return (0);
	} else if (ret != ENOENT) {
		return (ret);
	}

	if ((ret = ctf_dwarf_boolean(cup, die, DW_AT_external, &ext)) != 0) {
		if (ret == ENOENT) {
			*igp = B_FALSE;
			return (0);
		}
		return (ret);
	}
	*igp = ext != 0 ? B_TRUE : B_FALSE;
	return (0);
}

static int
ctf_dwarf_die_elfenc(Elf *elf, ctf_cu_t *cup, char *errbuf, size_t errlen)
{
	GElf_Ehdr ehdr;

	if (gelf_getehdr(elf, &ehdr) == NULL) {
		(void) snprintf(errbuf, errlen,
		    "failed to get ELF header: %s\n",
		    elf_errmsg(elf_errno()));
		return (ECTF_CONVBKERR);
	}

	cup->cu_mach = ehdr.e_machine;

	if (ehdr.e_ident[EI_CLASS] == ELFCLASS32) {
		cup->cu_ptrsz = 4;
		VERIFY(ctf_setmodel(cup->cu_ctfp, CTF_MODEL_ILP32) == 0);
	} else if (ehdr.e_ident[EI_CLASS] == ELFCLASS64) {
		cup->cu_ptrsz = 8;
		VERIFY(ctf_setmodel(cup->cu_ctfp, CTF_MODEL_LP64) == 0);
	} else {
		(void) snprintf(errbuf, errlen,
		    "unknown ELF class %d\n", ehdr.e_ident[EI_CLASS]);
		return (ECTF_CONVBKERR);
	}

	if (ehdr.e_ident[EI_DATA] == ELFDATA2LSB) {
		cup->cu_bigend = B_FALSE;
	} else if (ehdr.e_ident[EI_DATA] == ELFDATA2MSB) {
		cup->cu_bigend = B_TRUE;
	} else {
		(void) snprintf(errbuf, errlen,
		    "unknown ELF data encoding: %hhu\n", ehdr.e_ident[EI_DATA]);
		return (ECTF_CONVBKERR);
	}

	return (0);
}

typedef struct ctf_dwarf_fpent {
	size_t	cdfe_size;
	uint_t	cdfe_enc[3];
} ctf_dwarf_fpent_t;

typedef struct ctf_dwarf_fpmap {
	uint_t			cdf_mach;
	ctf_dwarf_fpent_t	cdf_ents[5];
} ctf_dwarf_fpmap_t;

static const ctf_dwarf_fpmap_t ctf_dwarf_fpmaps[] = {
	{ EM_SPARC, {
		{ 4, { CTF_FP_SINGLE, CTF_FP_CPLX, CTF_FP_IMAGRY } },
		{ 8, { CTF_FP_DOUBLE, CTF_FP_DCPLX, CTF_FP_DIMAGRY } },
		{ 16, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		{ 0, { 0 } }
	} },
	{ EM_SPARC32PLUS, {
		{ 4, { CTF_FP_SINGLE, CTF_FP_CPLX, CTF_FP_IMAGRY } },
		{ 8, { CTF_FP_DOUBLE, CTF_FP_DCPLX, CTF_FP_DIMAGRY } },
		{ 16, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		{ 0, { 0 } }
	} },
	{ EM_SPARCV9, {
		{ 4, { CTF_FP_SINGLE, CTF_FP_CPLX, CTF_FP_IMAGRY } },
		{ 8, { CTF_FP_DOUBLE, CTF_FP_DCPLX, CTF_FP_DIMAGRY } },
		{ 16, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		{ 0, { 0 } }
	} },
	{ EM_386, {
		{ 4, { CTF_FP_SINGLE, CTF_FP_CPLX, CTF_FP_IMAGRY } },
		{ 8, { CTF_FP_DOUBLE, CTF_FP_DCPLX, CTF_FP_DIMAGRY } },
		{ 12, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		/*
		 * ISO/IEC TS-18661-3:2015 defines several types with analogues
		 * to existing C types. However, in the i386 ABI there is no
		 * corresponding type for a _Float128. While, ideally we would
		 * add this as a discrete type, when C2x formally standardizes
		 * this and a number of additional extensions, we'll want to
		 * change that around. In the interim, we'll encode it as a
		 * weirdly sized long-double, even though not all the tools
		 * will expect an off-abi encoding.
		 */
		{ 16, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		{ 0, { 0 } }
	} },
	{ EM_X86_64, {
		{ 4, { CTF_FP_SINGLE, CTF_FP_CPLX, CTF_FP_IMAGRY } },
		{ 8, { CTF_FP_DOUBLE, CTF_FP_DCPLX, CTF_FP_DIMAGRY } },
		{ 16, { CTF_FP_LDOUBLE, CTF_FP_LDCPLX, CTF_FP_LDIMAGRY } },
		{ 0, { 0 } }
	} },
	{ EM_NONE }
};

/*
 * We want to normalize the type names that are used between compilers in the
 * case of complex. gcc prefixes things with types like 'long complex' where as
 * clang only calls them 'complex' in the dwarf even if in the C they are long
 * complex or similar.
 */
static int
ctf_dwarf_fixup_complex(ctf_cu_t *cup, ctf_encoding_t *enc, char **namep)
{
	const char *name;
	*namep = NULL;

	switch (enc->cte_format) {
	case CTF_FP_CPLX:
		name = "complex float";
		break;
	case CTF_FP_DCPLX:
		name = "complex double";
		break;
	case CTF_FP_LDCPLX:
		name = "complex long double";
		break;
	default:
		return (0);
	}

	*namep = ctf_strdup(name);
	if (*namep == NULL) {
		return (ENOMEM);
	}

	return (0);
}

static int
ctf_dwarf_float_base(ctf_cu_t *cup, Dwarf_Signed type, ctf_encoding_t *enc)
{
	const ctf_dwarf_fpmap_t *map = &ctf_dwarf_fpmaps[0];
	const ctf_dwarf_fpent_t *ent;
	uint_t col = 0, mult = 1;

	for (map = &ctf_dwarf_fpmaps[0]; map->cdf_mach != EM_NONE; map++) {
		if (map->cdf_mach == cup->cu_mach)
			break;
	}

	if (map->cdf_mach == EM_NONE) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "Unsupported machine type: %d\n", cup->cu_mach);
		return (ENOTSUP);
	}

	if (type == DW_ATE_complex_float) {
		mult = 2;
		col = 1;
	} else if (type == DW_ATE_imaginary_float ||
	    type == DW_ATE_SUN_imaginary_float) {
		col = 2;
	}

	ent = &map->cdf_ents[0];
	for (ent = &map->cdf_ents[0]; ent->cdfe_size != 0; ent++) {
		if (ent->cdfe_size * mult * 8 == enc->cte_bits) {
			enc->cte_format = ent->cdfe_enc[col];
			return (0);
		}
	}

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to find valid fp mapping for encoding %lld, size %d bits\n",
	    type, enc->cte_bits);
	return (EINVAL);
}

static int
ctf_dwarf_dwarf_base(ctf_cu_t *cup, Dwarf_Die die, int *kindp,
    ctf_encoding_t *enc)
{
	int ret;
	Dwarf_Signed type;

	if ((ret = ctf_dwarf_signed(cup, die, DW_AT_encoding, &type)) != 0)
		return (ret);

	switch (type) {
	case DW_ATE_unsigned:
	case DW_ATE_address:
		*kindp = CTF_K_INTEGER;
		enc->cte_format = 0;
		break;
	case DW_ATE_unsigned_char:
		*kindp = CTF_K_INTEGER;
		enc->cte_format = CTF_INT_CHAR;
		break;
	case DW_ATE_signed:
		*kindp = CTF_K_INTEGER;
		enc->cte_format = CTF_INT_SIGNED;
		break;
	case DW_ATE_signed_char:
		*kindp = CTF_K_INTEGER;
		enc->cte_format = CTF_INT_SIGNED | CTF_INT_CHAR;
		break;
	case DW_ATE_boolean:
		*kindp = CTF_K_INTEGER;
		enc->cte_format = CTF_INT_SIGNED | CTF_INT_BOOL;
		break;
	case DW_ATE_float:
	case DW_ATE_complex_float:
	case DW_ATE_imaginary_float:
	case DW_ATE_SUN_imaginary_float:
	case DW_ATE_SUN_interval_float:
		*kindp = CTF_K_FLOAT;
		if ((ret = ctf_dwarf_float_base(cup, type, enc)) != 0)
			return (ret);
		break;
	default:
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "encountered unknown DWARF encoding: %lld\n", type);
		return (ECTF_CONVBKERR);
	}

	return (0);
}

/*
 * Different compilers (at least GCC and Studio) use different names for types.
 * This parses the types and attempts to unify them. If this fails, we just fall
 * back to using the DWARF itself.
 */
static int
ctf_dwarf_parse_int(const char *name, int *kindp, ctf_encoding_t *enc,
    char **newnamep)
{
	char buf[256];
	char *base, *c, *last;
	int nlong = 0, nshort = 0, nchar = 0, nint = 0;
	int sign = 1;

	if (strlen(name) + 1 > sizeof (buf))
		return (EINVAL);

	(void) strlcpy(buf, name, sizeof (buf));
	for (c = strtok_r(buf, " ", &last); c != NULL;
	    c = strtok_r(NULL, " ", &last)) {
		if (strcmp(c, "signed") == 0) {
			sign = 1;
		} else if (strcmp(c, "unsigned") == 0) {
			sign = 0;
		} else if (strcmp(c, "long") == 0) {
			nlong++;
		} else if (strcmp(c, "char") == 0) {
			nchar++;
		} else if (strcmp(c, "short") == 0) {
			nshort++;
		} else if (strcmp(c, "int") == 0) {
			nint++;
		} else {
			/*
			 * If we don't recognize any of the tokens, we'll tell
			 * the caller to fall back to the dwarf-provided
			 * encoding information.
			 */
			return (EINVAL);
		}
	}

	if (nchar > 1 || nshort > 1 || nint > 1 || nlong > 2)
		return (EINVAL);

	if (nchar > 0) {
		if (nlong > 0 || nshort > 0 || nint > 0)
			return (EINVAL);
		base = "char";
	} else if (nshort > 0) {
		if (nlong > 0)
			return (EINVAL);
		base = "short";
	} else if (nlong > 0) {
		base = "long";
	} else {
		base = "int";
	}

	if (nchar > 0)
		enc->cte_format = CTF_INT_CHAR;
	else
		enc->cte_format = 0;

	if (sign > 0)
		enc->cte_format |= CTF_INT_SIGNED;

	(void) snprintf(buf, sizeof (buf), "%s%s%s",
	    (sign ? "" : "unsigned "),
	    (nlong > 1 ? "long " : ""),
	    base);

	*newnamep = ctf_strdup(buf);
	if (*newnamep == NULL)
		return (ENOMEM);
	*kindp = CTF_K_INTEGER;
	return (0);
}

static int
ctf_dwarf_create_base(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp, int isroot,
    Dwarf_Off off)
{
	int ret;
	char *name, *nname = NULL;
	Dwarf_Unsigned sz;
	int kind;
	ctf_encoding_t enc;
	ctf_id_t id;

	if ((ret = ctf_dwarf_string(cup, die, DW_AT_name, &name)) != 0)
		return (ret);
	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_byte_size, &sz)) != 0) {
		goto out;
	}
	ctf_dprintf("Creating base type %s from off %llu, size: %d\n", name,
	    off, sz);

	bzero(&enc, sizeof (ctf_encoding_t));
	enc.cte_bits = sz * 8;
	if ((ret = ctf_dwarf_parse_int(name, &kind, &enc, &nname)) == 0) {
		ctf_strfree(name);
		name = nname;
	} else {
		if (ret != EINVAL) {
			goto out;
		}
		ctf_dprintf("falling back to dwarf for base type %s\n", name);
		if ((ret = ctf_dwarf_dwarf_base(cup, die, &kind, &enc)) != 0) {
			goto out;
		}

		if (kind == CTF_K_FLOAT && (ret = ctf_dwarf_fixup_complex(cup,
		    &enc, &nname)) != 0) {
			goto out;
		} else if (nname != NULL) {
			ctf_strfree(name);
			name = nname;
		}
	}

	id = ctf_add_encoded(cup->cu_ctfp, isroot, name, &enc, kind);
	if (id == CTF_ERR) {
		ret = ctf_errno(cup->cu_ctfp);
	} else {
		*idp = id;
		ret = ctf_dwmap_add(cup, id, die, B_FALSE);
	}
out:
	ctf_strfree(name);
	return (ret);
}

/*
 * Getting a member's offset is a surprisingly intricate dance. It works as
 * follows:
 *
 * 1) If we're in DWARFv4, then we either have a DW_AT_data_bit_offset or we
 * have a DW_AT_data_member_location. We won't have both. Thus we check first
 * for DW_AT_data_bit_offset, and if it exists, we're set.
 *
 * Next, if we have a bitfield and we don't have a DW_AT_data_bit_offset, then
 * we have to grab the data location and use the following dance:
 *
 * 2) Gather the set of DW_AT_byte_size, DW_AT_bit_offset, and DW_AT_bit_size.
 * Of course, the DW_AT_byte_size may be omitted, even though it isn't always.
 * When it's been omitted, we then have to say that the size is that of the
 * underlying type, which forces that to be after a ctf_update(). Here, we have
 * to do different things based on whether or not we're using big endian or
 * little endian to obtain the proper offset.
 */
static int
ctf_dwarf_member_offset(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t mid,
    ulong_t *offp)
{
	int ret;
	Dwarf_Unsigned loc, bitsz, bytesz;
	Dwarf_Signed bitoff;
	size_t off;
	ssize_t tsz;

	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_data_bit_offset,
	    &loc)) == 0) {
		*offp = loc;
		return (0);
	} else if (ret != ENOENT) {
		return (ret);
	}

	if ((ret = ctf_dwarf_member_location(cup, die, &loc)) != 0)
		return (ret);
	off = loc * 8;

	if ((ret = ctf_dwarf_signed(cup, die, DW_AT_bit_offset,
	    &bitoff)) != 0) {
		if (ret != ENOENT)
			return (ret);
		*offp = off;
		return (0);
	}

	/* At this point we have to have DW_AT_bit_size */
	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_bit_size, &bitsz)) != 0)
		return (ret);

	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_byte_size,
	    &bytesz)) != 0) {
		if (ret != ENOENT)
			return (ret);
		if ((tsz = ctf_type_size(cup->cu_ctfp, mid)) == CTF_ERR) {
			int e = ctf_errno(cup->cu_ctfp);
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "failed to get type size: %s\n", ctf_errmsg(e));
			return (ECTF_CONVBKERR);
		}
	} else {
		tsz = bytesz;
	}
	tsz *= 8;
	if (cup->cu_bigend == B_TRUE) {
		*offp = off + bitoff;
	} else {
		*offp = off + tsz - bitoff - bitsz;
	}

	return (0);
}

/*
 * We need to determine if the member in question is a bitfield. If it is, then
 * we need to go through and create a new type that's based on the actual base
 * type, but has a different size. We also rename the type as a result to help
 * deal with future collisions.
 *
 * Here we need to look and see if we have a DW_AT_bit_size value. If we have a
 * bit size member and it does not equal the byte size member, then we need to
 * create a bitfield type based on this.
 *
 * Note: When we support DWARFv4, there may be a chance that we need to also
 * search for the DW_AT_byte_size if we don't have a DW_AT_bit_size member.
 */
static int
ctf_dwarf_member_bitfield(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp)
{
	int ret;
	Dwarf_Unsigned bitsz;
	ctf_encoding_t e;
	ctf_dwbitf_t *cdb;
	ctf_dtdef_t *dtd;
	ctf_id_t base = *idp;
	int kind;

	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_bit_size, &bitsz)) != 0) {
		if (ret == ENOENT)
			return (0);
		return (ret);
	}

	ctf_dprintf("Trying to deal with bitfields on %d:%d\n", base, bitsz);
	/*
	 * Given that we now have a bitsize, time to go do something about it.
	 * We're going to create a new type based on the current one, but first
	 * we need to find the base type. This means we need to traverse any
	 * typedef's, consts, and volatiles until we get to what should be
	 * something of type integer or enumeration.
	 */
	VERIFY(bitsz < UINT32_MAX);
	dtd = ctf_dtd_lookup(cup->cu_ctfp, base);
	VERIFY(dtd != NULL);
	kind = CTF_INFO_KIND(dtd->dtd_data.ctt_info);
	while (kind == CTF_K_TYPEDEF || kind == CTF_K_CONST ||
	    kind == CTF_K_VOLATILE) {
		dtd = ctf_dtd_lookup(cup->cu_ctfp, dtd->dtd_data.ctt_type);
		VERIFY(dtd != NULL);
		kind = CTF_INFO_KIND(dtd->dtd_data.ctt_info);
	}
	ctf_dprintf("got kind %d\n", kind);
	VERIFY(kind == CTF_K_INTEGER || kind == CTF_K_ENUM);

	/*
	 * As surprising as it may be, it is strictly possible to create a
	 * bitfield that is based on an enum. Of course, the C standard leaves
	 * enums sizing as an ABI concern more or less. To that effect, today on
	 * all illumos platforms the size of an enum is generally that of an
	 * int as our supported data models and ABIs all agree on that. So what
	 * we'll do is fake up a CTF encoding here to use. In this case, we'll
	 * treat it as an unsigned value of whatever size the underlying enum
	 * currently has (which is in the ctt_size member of its dynamic type
	 * data).
	 */
	if (kind == CTF_K_INTEGER) {
		e = dtd->dtd_u.dtu_enc;
	} else {
		bzero(&e, sizeof (ctf_encoding_t));
		e.cte_bits = dtd->dtd_data.ctt_size * NBBY;
	}

	for (cdb = ctf_list_next(&cup->cu_bitfields); cdb != NULL;
	    cdb = ctf_list_next(cdb)) {
		if (cdb->cdb_base == base && cdb->cdb_nbits == bitsz)
			break;
	}

	/*
	 * Create a new type if none exists. We name all types in a way that is
	 * guaranteed not to conflict with the corresponding C type. We do this
	 * by using the ':' operator.
	 */
	if (cdb == NULL) {
		size_t namesz;
		char *name;

		e.cte_bits = bitsz;
		namesz = snprintf(NULL, 0, "%s:%d", dtd->dtd_name,
		    (uint32_t)bitsz);
		name = ctf_alloc(namesz + 1);
		if (name == NULL)
			return (ENOMEM);
		cdb = ctf_alloc(sizeof (ctf_dwbitf_t));
		if (cdb == NULL) {
			ctf_free(name, namesz + 1);
			return (ENOMEM);
		}
		(void) snprintf(name, namesz + 1, "%s:%d", dtd->dtd_name,
		    (uint32_t)bitsz);

		cdb->cdb_base = base;
		cdb->cdb_nbits = bitsz;
		cdb->cdb_id = ctf_add_integer(cup->cu_ctfp, CTF_ADD_NONROOT,
		    name, &e);
		if (cdb->cdb_id == CTF_ERR) {
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "failed to get add bitfield type %s: %s\n", name,
			    ctf_errmsg(ctf_errno(cup->cu_ctfp)));
			ctf_free(name, namesz + 1);
			ctf_free(cdb, sizeof (ctf_dwbitf_t));
			return (ECTF_CONVBKERR);
		}
		ctf_free(name, namesz + 1);
		ctf_list_append(&cup->cu_bitfields, cdb);
	}

	*idp = cdb->cdb_id;

	return (0);
}

static int
ctf_dwarf_fixup_sou(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t base, boolean_t add)
{
	int ret, kind;
	Dwarf_Die child, memb;
	Dwarf_Unsigned size;

	kind = ctf_type_kind(cup->cu_ctfp, base);
	VERIFY(kind != CTF_ERR);
	VERIFY(kind == CTF_K_STRUCT || kind == CTF_K_UNION);

	/*
	 * Members are in children. However, gcc also allows empty ones.
	 */
	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0)
		return (ret);
	if (child == NULL)
		return (0);

	memb = child;
	while (memb != NULL) {
		Dwarf_Die sib, tdie;
		Dwarf_Half tag;
		ctf_id_t mid;
		char *mname;
		ulong_t memboff = 0;

		if ((ret = ctf_dwarf_tag(cup, memb, &tag)) != 0)
			return (ret);

		if (tag != DW_TAG_member)
			goto next;

		if ((ret = ctf_dwarf_refdie(cup, memb, DW_AT_type, &tdie)) != 0)
			return (ret);

		if ((ret = ctf_dwarf_convert_type(cup, tdie, &mid,
		    CTF_ADD_NONROOT)) != 0)
			return (ret);
		ctf_dprintf("Got back type id: %d\n", mid);

		/*
		 * If we're not adding a member, just go ahead and return.
		 */
		if (add == B_FALSE) {
			if ((ret = ctf_dwarf_member_bitfield(cup, memb,
			    &mid)) != 0)
				return (ret);
			goto next;
		}

		if ((ret = ctf_dwarf_string(cup, memb, DW_AT_name,
		    &mname)) != 0 && ret != ENOENT)
			return (ret);
		if (ret == ENOENT)
			mname = NULL;

		if (kind == CTF_K_UNION) {
			memboff = 0;
		} else if ((ret = ctf_dwarf_member_offset(cup, memb, mid,
		    &memboff)) != 0) {
			if (mname != NULL)
				ctf_strfree(mname);
			return (ret);
		}

		if ((ret = ctf_dwarf_member_bitfield(cup, memb, &mid)) != 0)
			return (ret);

		ret = ctf_add_member(cup->cu_ctfp, base, mname, mid, memboff);
		if (ret == CTF_ERR) {
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "failed to add member %s: %s\n",
			    mname, ctf_errmsg(ctf_errno(cup->cu_ctfp)));
			if (mname != NULL)
				ctf_strfree(mname);
			return (ECTF_CONVBKERR);
		}

		if (mname != NULL)
			ctf_strfree(mname);

next:
		if ((ret = ctf_dwarf_sib(cup, memb, &sib)) != 0)
			return (ret);
		memb = sib;
	}

	/*
	 * If we're not adding members, then we don't know the final size of the
	 * structure, so end here.
	 */
	if (add == B_FALSE)
		return (0);

	/* Finally set the size of the structure to the actual byte size */
	if ((ret = ctf_dwarf_unsigned(cup, die, DW_AT_byte_size, &size)) != 0)
		return (ret);
	if ((ctf_set_size(cup->cu_ctfp, base, size)) == CTF_ERR) {
		int e = ctf_errno(cup->cu_ctfp);
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to set type size for %d to 0x%x: %s\n", base,
		    (uint32_t)size, ctf_errmsg(e));
		return (ECTF_CONVBKERR);
	}

	return (0);
}

static int
ctf_dwarf_create_sou(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp,
    int kind, int isroot)
{
	int ret;
	char *name;
	ctf_id_t base;
	Dwarf_Die child;
	Dwarf_Bool decl;

	/*
	 * Deal with the terribly annoying case of anonymous structs and unions.
	 * If they don't have a name, set the name to the empty string.
	 */
	if ((ret = ctf_dwarf_string(cup, die, DW_AT_name, &name)) != 0 &&
	    ret != ENOENT)
		return (ret);
	if (ret == ENOENT)
		name = NULL;

	/*
	 * We need to check if we just have a declaration here. If we do, then
	 * instead of creating an actual structure or union, we're just going to
	 * go ahead and create a forward. During a dedup or merge, the forward
	 * will be replaced with the real thing.
	 */
	if ((ret = ctf_dwarf_boolean(cup, die, DW_AT_declaration,
	    &decl)) != 0) {
		if (ret != ENOENT)
			return (ret);
		decl = 0;
	}

	if (decl == B_TRUE) {
		base = ctf_add_forward(cup->cu_ctfp, isroot, name, kind);
	} else if (kind == CTF_K_STRUCT) {
		base = ctf_add_struct(cup->cu_ctfp, isroot, name);
	} else {
		base = ctf_add_union(cup->cu_ctfp, isroot, name);
	}
	ctf_dprintf("added sou %s (%d) (%ld) forward=%d\n",
	    name, kind, base, decl == B_TRUE);
	if (name != NULL)
		ctf_strfree(name);
	if (base == CTF_ERR)
		return (ctf_errno(cup->cu_ctfp));
	*idp = base;

	/*
	 * If it's just a declaration, we're not going to mark it for fix up or
	 * do anything else.
	 */
	if (decl == B_TRUE)
		return (ctf_dwmap_add(cup, base, die, B_FALSE));
	if ((ret = ctf_dwmap_add(cup, base, die, B_TRUE)) != 0)
		return (ret);

	/*
	 * The children of a structure or union are generally members. However,
	 * some compilers actually insert structs and unions there and not as a
	 * top-level die. Therefore, to make sure we honor our pass 1 contract
	 * of having all the base types, but not members, we need to walk this
	 * for instances of a DW_TAG_union_type.
	 */
	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0)
		return (ret);

	while (child != NULL) {
		Dwarf_Half tag;
		Dwarf_Die sib;

		if ((ret = ctf_dwarf_tag(cup, child, &tag)) != 0)
			return (ret);

		switch (tag) {
		case DW_TAG_union_type:
		case DW_TAG_structure_type:
			ret = ctf_dwarf_convert_type(cup, child, NULL,
			    CTF_ADD_NONROOT);
			if (ret != 0) {
				return (ret);
			}
			break;
		default:
			break;
		}

		if ((ret = ctf_dwarf_sib(cup, child, &sib)) != 0)
			return (ret);
		child = sib;
	}

	return (0);
}

static int
ctf_dwarf_array_upper_bound(ctf_cu_t *cup, Dwarf_Die range, ctf_arinfo_t *ar)
{
	Dwarf_Attribute attr;
	Dwarf_Unsigned uval;
	Dwarf_Signed sval;
	Dwarf_Half attrnum, form;
	Dwarf_Error derr;
	enum Dwarf_Form_Class class;
	const char *formstr = NULL;
	uint_t adj = 0;
	int ret = 0;

	ctf_dprintf("setting array upper bound\n");

	ar->ctr_nelems = 0;

	/*
	 * Different compilers use different attributes to indicate the size of
	 * an array. GCC has traditionally used DW_AT_upper_bound, while Clang
	 * uses DW_AT_count. They have slightly different semantics. DW_AT_count
	 * indicates the total number of elements that are present, while
	 * DW_AT_upper_bound indicates the last index, hence we need to add one
	 * to that index to get the count.
	 *
	 * We first search for DW_AT_count and then for DW_AT_upper_bound. If we
	 * find neither, then we treat the lack of this as a zero element array.
	 * Our value is initialized assuming we find a DW_AT_count value.
	 */
	attrnum = DW_AT_count;
	ret = ctf_dwarf_attribute(cup, range, DW_AT_count, &attr);
	if (ret != 0 && ret != ENOENT) {
		return (ret);
	} else if (ret == ENOENT) {
		attrnum = DW_AT_upper_bound;
		ret = ctf_dwarf_attribute(cup, range, DW_AT_upper_bound, &attr);
		if (ret != 0 && ret != ENOENT) {
			return (ret);
		} else if (ret == ENOENT) {
			return (0);
		} else {
			adj = 1;
		}
	}

	DWARF_LOCK(cup);
	ret = dwarf_whatform(attr, &form, &derr);
	if (ret != DW_DLV_OK) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "failed to get DW_AT_upper_bound attribute form: %s\n",
		    dwarf_errmsg(derr));
		ret = ECTF_CONVBKERR;
		goto done;
	}

	/*
	 * Compilers can indicate array bounds using signed or unsigned values.
	 * Additionally, some compilers may also store the array bounds
	 * using as DW_FORM_data{1,2,4,8} (which DWARF treats as raw data and
	 * expects the caller to understand how to interpret the value).
	 *
	 * GCC 4.4.4 appears to always use unsigned values to encode the
	 * array size (using '(unsigned)-1' to represent a zero-length or
	 * unknown length array). Later versions of GCC use a signed value of
	 * -1 for zero/unknown length arrays, and unsigned values to encode
	 * known array sizes.
	 *
	 * Both dwarf_formsdata() and dwarf_formudata() will retrieve values
	 * as their respective signed/unsigned forms, but both will also
	 * retreive DW_FORM_data{1,2,4,8} values and treat them as signed or
	 * unsigned integers (i.e. dwarf_formsdata() treats DW_FORM_dataXX
	 * as signed integers and dwarf_formudata() treats DW_FORM_dataXX as
	 * unsigned integers). Both will return an error if the form is not
	 * their respective signed/unsigned form, or DW_FORM_dataXX.
	 *
	 * To obtain the upper bound, we use the appropriate
	 * dwarf_form[su]data() function based on the form of DW_AT_upper_bound.
	 * Additionally, we let dwarf_formudata() handle the DW_FORM_dataXX
	 * forms (via the default option in the switch). If the value is in an
	 * unexpected form (i.e. not DW_FORM_udata or DW_FORM_dataXX),
	 * dwarf_formudata() will return failure (i.e. not DW_DLV_OK) and set
	 * derr with the specific error value.
	 */
	switch (form) {
	case DW_FORM_sdata:
		if (dwarf_formsdata(attr, &sval, &derr) == DW_DLV_OK) {
			ar->ctr_nelems = sval + adj;
			goto done;
		}
		break;
	case DW_FORM_udata:
	default:
		if (dwarf_formudata(attr, &uval, &derr) == DW_DLV_OK) {
			ar->ctr_nelems = uval + adj;
			goto done;
		}
		break;
	}

	/*
	 * The value of attr is not always a constant value. For things
	 * such as C99 variable length arrays, it can be a reference to another
	 * entity or a DWARF expression. In either of these cases, we treat
	 * this as a zero length array.
	 */
	class = dwarf_get_form_class(cup->cu_vers, attrnum, cup->cu_addrsz,
	    form);

	if (class == DW_FORM_CLASS_REFERENCE || class == DW_FORM_CLASS_BLOCK)
		goto done;

	if (dwarf_get_FORM_name(form, &formstr) != DW_DLV_OK)
		formstr = "unknown DWARF form";

	(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
	    "failed to get %s (%hu) value for DW_AT_upper_bound: %s\n",
	    formstr, form, dwarf_errmsg(derr));
	ret = ECTF_CONVBKERR;

done:
	DWARF_UNLOCK(cup);
	ctf_dwarf_dealloc(cup, attr, DW_DLA_ATTR);
	return (ret);
}

static int
ctf_dwarf_create_array_range(ctf_cu_t *cup, Dwarf_Die range, ctf_id_t *idp,
    ctf_id_t base, int isroot)
{
	int ret;
	Dwarf_Die sib;
	ctf_arinfo_t ar;

	ctf_dprintf("creating array range\n");

	if ((ret = ctf_dwarf_sib(cup, range, &sib)) != 0)
		return (ret);
	if (sib != NULL) {
		ctf_id_t id;
		if ((ret = ctf_dwarf_create_array_range(cup, sib, &id,
		    base, CTF_ADD_NONROOT)) != 0)
			return (ret);
		ar.ctr_contents = id;
	} else {
		ar.ctr_contents = base;
	}

	if ((ar.ctr_index = ctf_dwarf_long(cup)) == CTF_ERR)
		return (ctf_errno(cup->cu_ctfp));

	if ((ret = ctf_dwarf_array_upper_bound(cup, range, &ar)) != 0)
		return (ret);

	if ((*idp = ctf_add_array(cup->cu_ctfp, isroot, &ar)) == CTF_ERR)
		return (ctf_errno(cup->cu_ctfp));

	return (0);
}

/*
 * Try and create an array type. First, the kind of the array is specified in
 * the DW_AT_type entry. Next, the number of entries is stored in a more
 * complicated form, we should have a child that has the DW_TAG_subrange type.
 */
static int
ctf_dwarf_create_array(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp, int isroot)
{
	int ret;
	Dwarf_Die tdie, rdie;
	ctf_id_t tid;
	Dwarf_Half rtag;

	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_type, &tdie)) != 0)
		return (ret);
	if ((ret = ctf_dwarf_convert_type(cup, tdie, &tid,
	    CTF_ADD_NONROOT)) != 0)
		return (ret);

	if ((ret = ctf_dwarf_child(cup, die, &rdie)) != 0)
		return (ret);
	if ((ret = ctf_dwarf_tag(cup, rdie, &rtag)) != 0)
		return (ret);
	if (rtag != DW_TAG_subrange_type) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "encountered array without DW_TAG_subrange_type child\n");
		return (ECTF_CONVBKERR);
	}

	/*
	 * The compiler may opt to describe a multi-dimensional array as one
	 * giant array or it may opt to instead encode it as a series of
	 * subranges. If it's the latter, then for each subrange we introduce a
	 * type. We can always use the base type.
	 */
	if ((ret = ctf_dwarf_create_array_range(cup, rdie, idp, tid,
	    isroot)) != 0)
		return (ret);
	ctf_dprintf("Got back id %d\n", *idp);
	return (ctf_dwmap_add(cup, *idp, die, B_FALSE));
}

/*
 * Given "const int const_array3[11]", GCC7 at least will create a DIE tree of
 * DW_TAG_const_type:DW_TAG_array_type:DW_Tag_const_type:<member_type>.
 *
 * Given C's syntax, this renders out as "const const int const_array3[11]".  To
 * get closer to round-tripping (and make the unit tests work), we'll peek for
 * this case, and avoid adding the extraneous qualifier if we see that the
 * underlying array referent already has the same qualifier.
 *
 * This is unfortunately less trivial than it could be: this issue applies to
 * qualifier sets like "const volatile", as well as multi-dimensional arrays, so
 * we need to descend down those.
 *
 * Returns CTF_ERR on error, or a boolean value otherwise.
 */
static int
needed_array_qualifier(ctf_cu_t *cup, int kind, ctf_id_t ref_id)
{
	const ctf_type_t *t;
	ctf_arinfo_t arinfo;
	int akind;

	if (kind != CTF_K_CONST && kind != CTF_K_VOLATILE &&
	    kind != CTF_K_RESTRICT)
		return (1);

	if ((t = ctf_dyn_lookup_by_id(cup->cu_ctfp, ref_id)) == NULL)
		return (CTF_ERR);

	if (LCTF_INFO_KIND(cup->cu_ctfp, t->ctt_info) != CTF_K_ARRAY)
		return (1);

	if (ctf_dyn_array_info(cup->cu_ctfp, ref_id, &arinfo) != 0)
		return (CTF_ERR);

	ctf_id_t id = arinfo.ctr_contents;

	for (;;) {
		if ((t = ctf_dyn_lookup_by_id(cup->cu_ctfp, id)) == NULL)
			return (CTF_ERR);

		akind = LCTF_INFO_KIND(cup->cu_ctfp, t->ctt_info);

		if (akind == kind)
			break;

		if (akind == CTF_K_ARRAY) {
			if (ctf_dyn_array_info(cup->cu_ctfp,
			    id, &arinfo) != 0)
				return (CTF_ERR);
			id = arinfo.ctr_contents;
			continue;
		}

		if (akind != CTF_K_CONST && akind != CTF_K_VOLATILE &&
		    akind != CTF_K_RESTRICT)
			break;

		id = t->ctt_type;
	}

	if (kind == akind) {
		ctf_dprintf("ignoring extraneous %s qualifier for array %d\n",
		    ctf_kind_name(cup->cu_ctfp, kind), ref_id);
	}

	return (kind != akind);
}

static int
ctf_dwarf_create_reference(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp,
    int kind, int isroot)
{
	int ret;
	ctf_id_t id;
	Dwarf_Die tdie;
	char *name;

	if ((ret = ctf_dwarf_string(cup, die, DW_AT_name, &name)) != 0 &&
	    ret != ENOENT)
		return (ret);
	if (ret == ENOENT)
		name = NULL;

	ctf_dprintf("reference kind %d %s\n", kind, name != NULL ? name : "<>");

	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_type, &tdie)) != 0) {
		if (ret != ENOENT) {
			ctf_strfree(name);
			return (ret);
		}
		if ((id = ctf_dwarf_void(cup)) == CTF_ERR) {
			ctf_strfree(name);
			return (ctf_errno(cup->cu_ctfp));
		}
	} else {
		if ((ret = ctf_dwarf_convert_type(cup, tdie, &id,
		    CTF_ADD_NONROOT)) != 0) {
			ctf_strfree(name);
			return (ret);
		}
	}

	if ((ret = needed_array_qualifier(cup, kind, id)) <= 0) {
		if (ret != 0) {
			ret = (ctf_errno(cup->cu_ctfp));
		} else {
			*idp = id;
		}

		ctf_strfree(name);
		return (ret);
	}

	if ((*idp = ctf_add_reftype(cup->cu_ctfp, isroot, name, id, kind)) ==
	    CTF_ERR) {
		ctf_strfree(name);
		return (ctf_errno(cup->cu_ctfp));
	}

	ctf_strfree(name);
	return (ctf_dwmap_add(cup, *idp, die, B_FALSE));
}

static int
ctf_dwarf_create_enum(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp, int isroot)
{
	size_t size = 0;
	Dwarf_Die child;
	Dwarf_Unsigned dw;
	ctf_id_t id;
	char *enumname;
	int ret;

	ret = ctf_dwarf_string(cup, die, DW_AT_name, &enumname);
	if (ret != 0 && ret != ENOENT)
		return (ret);
	if (ret == ENOENT)
		enumname = NULL;

	/*
	 * Enumerations may have a size associated with them, particularly if
	 * they're packed. Note, a Dwarf_Unsigned is larger than a size_t on an
	 * ILP32 system.
	 */
	if (ctf_dwarf_unsigned(cup, die, DW_AT_byte_size, &dw) == 0 &&
	    dw < SIZE_MAX) {
		size = (size_t)dw;
	}

	id = ctf_add_enum(cup->cu_ctfp, isroot, enumname, size);
	ctf_dprintf("added enum %s (%d)\n",
	    enumname == NULL ? "<anon>" : enumname, id);
	if (id == CTF_ERR) {
		ret = ctf_errno(cup->cu_ctfp);
		goto out;
	}
	*idp = id;
	if ((ret = ctf_dwmap_add(cup, id, die, B_FALSE)) != 0)
		goto out;

	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0) {
		if (ret == ENOENT)
			ret = 0;
		goto out;
	}

	while (child != NULL) {
		Dwarf_Half tag;
		Dwarf_Signed sval;
		Dwarf_Unsigned uval;
		Dwarf_Die arg = child;
		char *name;
		int eval;

		if ((ret = ctf_dwarf_sib(cup, arg, &child)) != 0)
			break;

		if ((ret = ctf_dwarf_tag(cup, arg, &tag)) != 0)
			break;

		if (tag != DW_TAG_enumerator) {
			if ((ret = ctf_dwarf_convert_type(cup, arg, NULL,
			    CTF_ADD_NONROOT)) != 0) {
				break;
			}
			continue;
		}

		/*
		 * DWARF v4 section 5.7 tells us we'll always have names.
		 */
		if ((ret = ctf_dwarf_string(cup, arg, DW_AT_name, &name)) != 0)
			break;

		/*
		 * We have to be careful here: newer GCCs generate DWARF where
		 * an unsigned value will happily pass ctf_dwarf_signed().
		 * Since negative values will fail ctf_dwarf_unsigned(), we try
		 * that first to make sure we get the right value.
		 */
		if ((ret = ctf_dwarf_unsigned(cup, arg, DW_AT_const_value,
		    &uval)) == 0) {
			eval = (int)uval;
		} else {
			/*
			 * ctf_dwarf_unsigned will have left an error in the
			 * buffer
			 */
			*cup->cu_errbuf = '\0';

			if ((ret = ctf_dwarf_signed(cup, arg, DW_AT_const_value,
			    &sval)) == 0) {
				eval = sval;
			}
		}

		if (ret != 0) {
			if (ret == ENOENT) {
				(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
				    "encountered enumeration without constant "
				    "value\n");
				ret = ECTF_CONVBKERR;
			}
			ctf_strfree(name);
			break;
		}

		ret = ctf_add_enumerator(cup->cu_ctfp, id, name, eval);
		if (ret == CTF_ERR) {
			ret = ctf_errno(cup->cu_ctfp);

			if (ret == ECTF_DTFULL && (cup->cu_handle->cch_flags &
			    CTF_ALLOW_TRUNCATION)) {
				if (cup->cu_handle->cch_warncb != NULL) {
					cup->cu_handle->cch_warncb(
					    cup->cu_handle->cch_warncb_arg,
					    "truncating enumeration %s at %s\n",
					    name, enumname == NULL ? "<anon>" :
					    enumname);
				}
				ret = 0;
			} else {
				(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
				    "failed to add enumerator %s (%d) "
				    "to %s (%d)\n", name, eval,
				    enumname == NULL ? "<anon>" : enumname, id);
			}
			ctf_strfree(name);
			break;
		}
		ctf_strfree(name);
	}

out:

	if (enumname != NULL)
		ctf_strfree(enumname);

	return (ret);
}

/*
 * For a function pointer, walk over and process all of its children, unless we
 * encounter one that's just a declaration. In which case, we error on it.
 */
static int
ctf_dwarf_create_fptr(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp, int isroot)
{
	int ret;
	Dwarf_Bool b;
	ctf_funcinfo_t fi;
	Dwarf_Die retdie;
	ctf_id_t *argv = NULL;

	bzero(&fi, sizeof (ctf_funcinfo_t));

	if ((ret = ctf_dwarf_boolean(cup, die, DW_AT_declaration, &b)) != 0) {
		if (ret != ENOENT)
			return (ret);
	} else {
		if (b != 0)
			return (EPROTOTYPE);
	}

	/*
	 * Return type is in DW_AT_type, if none, it returns void.
	 */
	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_type, &retdie)) != 0) {
		if (ret != ENOENT)
			return (ret);
		if ((fi.ctc_return = ctf_dwarf_void(cup)) == CTF_ERR)
			return (ctf_errno(cup->cu_ctfp));
	} else {
		if ((ret = ctf_dwarf_convert_type(cup, retdie, &fi.ctc_return,
		    CTF_ADD_NONROOT)) != 0)
			return (ret);
	}

	if ((ret = ctf_dwarf_function_count(cup, die, &fi, B_TRUE)) != 0) {
		return (ret);
	}

	if (fi.ctc_argc != 0) {
		argv = ctf_alloc(sizeof (ctf_id_t) * fi.ctc_argc);
		if (argv == NULL)
			return (ENOMEM);

		if ((ret = ctf_dwarf_convert_fargs(cup, die, &fi, argv)) != 0) {
			ctf_free(argv, sizeof (ctf_id_t) * fi.ctc_argc);
			return (ret);
		}
	}

	if ((*idp = ctf_add_funcptr(cup->cu_ctfp, isroot, &fi, argv)) ==
	    CTF_ERR) {
		ctf_free(argv, sizeof (ctf_id_t) * fi.ctc_argc);
		return (ctf_errno(cup->cu_ctfp));
	}

	ctf_free(argv, sizeof (ctf_id_t) * fi.ctc_argc);
	return (ctf_dwmap_add(cup, *idp, die, B_FALSE));
}

static int
ctf_dwarf_convert_type(ctf_cu_t *cup, Dwarf_Die die, ctf_id_t *idp,
    int isroot)
{
	int ret;
	Dwarf_Off offset;
	Dwarf_Half tag;
	ctf_dwmap_t lookup, *map;
	ctf_id_t id;

	if (idp == NULL)
		idp = &id;

	if ((ret = ctf_dwarf_offset(cup, die, &offset)) != 0)
		return (ret);

	if (offset > cup->cu_maxoff) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "die offset %llu beyond maximum for header %llu\n",
		    offset, cup->cu_maxoff);
		return (ECTF_CONVBKERR);
	}

	/*
	 * If we've already added an entry for this offset, then we're done.
	 */
	lookup.cdm_off = offset;
	if ((map = avl_find(&cup->cu_map, &lookup, NULL)) != NULL) {
		*idp = map->cdm_id;
		return (0);
	}

	if ((ret = ctf_dwarf_tag(cup, die, &tag)) != 0)
		return (ret);

	ret = ENOTSUP;
	switch (tag) {
	case DW_TAG_base_type:
		ctf_dprintf("base\n");
		ret = ctf_dwarf_create_base(cup, die, idp, isroot, offset);
		break;
	case DW_TAG_array_type:
		ctf_dprintf("array\n");
		ret = ctf_dwarf_create_array(cup, die, idp, isroot);
		break;
	case DW_TAG_enumeration_type:
		ctf_dprintf("enum\n");
		ret = ctf_dwarf_create_enum(cup, die, idp, isroot);
		break;
	case DW_TAG_pointer_type:
		ctf_dprintf("pointer\n");
		ret = ctf_dwarf_create_reference(cup, die, idp, CTF_K_POINTER,
		    isroot);
		break;
	case DW_TAG_structure_type:
		ctf_dprintf("struct\n");
		ret = ctf_dwarf_create_sou(cup, die, idp, CTF_K_STRUCT,
		    isroot);
		break;
	case DW_TAG_subroutine_type:
		ctf_dprintf("fptr\n");
		ret = ctf_dwarf_create_fptr(cup, die, idp, isroot);
		break;
	case DW_TAG_typedef:
		ctf_dprintf("typedef\n");
		ret = ctf_dwarf_create_reference(cup, die, idp, CTF_K_TYPEDEF,
		    isroot);
		break;
	case DW_TAG_union_type:
		ctf_dprintf("union\n");
		ret = ctf_dwarf_create_sou(cup, die, idp, CTF_K_UNION,
		    isroot);
		break;
	case DW_TAG_const_type:
		ctf_dprintf("const\n");
		ret = ctf_dwarf_create_reference(cup, die, idp, CTF_K_CONST,
		    isroot);
		break;
	case DW_TAG_volatile_type:
		ctf_dprintf("volatile\n");
		ret = ctf_dwarf_create_reference(cup, die, idp, CTF_K_VOLATILE,
		    isroot);
		break;
	case DW_TAG_restrict_type:
		ctf_dprintf("restrict\n");
		ret = ctf_dwarf_create_reference(cup, die, idp, CTF_K_RESTRICT,
		    isroot);
		break;
	default:
		ctf_dprintf("ignoring tag type %x\n", tag);
		*idp = CTF_ERR;
		ret = 0;
		break;
	}
	ctf_dprintf("ctf_dwarf_convert_type tag specific handler returned %d\n",
	    ret);

	return (ret);
}

static int
ctf_dwarf_walk_lexical(ctf_cu_t *cup, Dwarf_Die die)
{
	int ret;
	Dwarf_Die child;

	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0)
		return (ret);

	if (child == NULL)
		return (0);

	return (ctf_dwarf_convert_die(cup, die));
}

static int
ctf_dwarf_function_count(ctf_cu_t *cup, Dwarf_Die die, ctf_funcinfo_t *fip,
    boolean_t fptr)
{
	int ret;
	Dwarf_Die child, sib, arg;

	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0)
		return (ret);

	arg = child;
	while (arg != NULL) {
		Dwarf_Half tag;

		if ((ret = ctf_dwarf_tag(cup, arg, &tag)) != 0)
			return (ret);

		/*
		 * We have to check for a varargs type declaration. This will
		 * happen in one of two ways. If we have a function pointer
		 * type, then it'll be done with a tag of type
		 * DW_TAG_unspecified_parameters. However, it only means we have
		 * a variable number of arguments, if we have more than one
		 * argument found so far. Otherwise, when we have a function
		 * type, it instead uses a formal parameter whose name is '...'
		 * to indicate a variable arguments member.
		 *
		 * Also, if we have a function pointer, then we have to expect
		 * that we might not get a name at all.
		 */
		if (tag == DW_TAG_formal_parameter && fptr == B_FALSE) {
			char *name;
			if ((ret = ctf_dwarf_string(cup, die, DW_AT_name,
			    &name)) != 0)
				return (ret);
			if (strcmp(name, DWARF_VARARGS_NAME) == 0)
				fip->ctc_flags |= CTF_FUNC_VARARG;
			else
				fip->ctc_argc++;
			ctf_strfree(name);
		} else if (tag == DW_TAG_formal_parameter) {
			fip->ctc_argc++;
		} else if (tag == DW_TAG_unspecified_parameters &&
		    fip->ctc_argc > 0) {
			fip->ctc_flags |= CTF_FUNC_VARARG;
		}
		if ((ret = ctf_dwarf_sib(cup, arg, &sib)) != 0)
			return (ret);
		arg = sib;
	}

	return (0);
}

static int
ctf_dwarf_convert_fargs(ctf_cu_t *cup, Dwarf_Die die, ctf_funcinfo_t *fip,
    ctf_id_t *argv)
{
	int ret;
	int i = 0;
	Dwarf_Die child, sib, arg;

	if ((ret = ctf_dwarf_child(cup, die, &child)) != 0)
		return (ret);

	arg = child;
	while (arg != NULL) {
		Dwarf_Half tag;

		if ((ret = ctf_dwarf_tag(cup, arg, &tag)) != 0)
			return (ret);
		if (tag == DW_TAG_formal_parameter) {
			Dwarf_Die tdie;

			if ((ret = ctf_dwarf_refdie(cup, arg, DW_AT_type,
			    &tdie)) != 0)
				return (ret);

			if ((ret = ctf_dwarf_convert_type(cup, tdie, &argv[i],
			    CTF_ADD_ROOT)) != 0)
				return (ret);
			i++;

			/*
			 * Once we hit argc entries, we're done. This ensures we
			 * don't accidentally hit a varargs which should be the
			 * last entry.
			 */
			if (i == fip->ctc_argc)
				break;
		}

		if ((ret = ctf_dwarf_sib(cup, arg, &sib)) != 0)
			return (ret);
		arg = sib;
	}

	return (0);
}

static int
ctf_dwarf_convert_function(ctf_cu_t *cup, Dwarf_Die die)
{
	ctf_dwfunc_t *cdf;
	Dwarf_Die tdie;
	Dwarf_Bool b;
	char *name;
	int ret;

	/*
	 * Functions that don't have a name are generally functions that have
	 * been inlined and thus most information about them has been lost. If
	 * we can't get a name, then instead of returning ENOENT, we silently
	 * swallow the error.
	 */
	if ((ret = ctf_dwarf_string(cup, die, DW_AT_name, &name)) != 0) {
		if (ret == ENOENT)
			return (0);
		return (ret);
	}

	ctf_dprintf("beginning work on function %s (die %llx)\n",
	    name, ctf_die_offset(cup, die));

	if ((ret = ctf_dwarf_boolean(cup, die, DW_AT_declaration, &b)) != 0) {
		if (ret != ENOENT) {
			ctf_strfree(name);
			return (ret);
		}
	} else if (b != 0) {
		/*
		 * GCC7 at least creates empty DW_AT_declarations for functions
		 * defined in headers.  As they lack details on the function
		 * prototype, we need to ignore them.  If we later actually
		 * see the relevant function's definition, we will see another
		 * DW_TAG_subprogram that is more complete.
		 */
		ctf_dprintf("ignoring declaration of function %s (die %llx)\n",
		    name, ctf_die_offset(cup, die));
		ctf_strfree(name);
		return (0);
	}

	if ((cdf = ctf_alloc(sizeof (ctf_dwfunc_t))) == NULL) {
		ctf_strfree(name);
		return (ENOMEM);
	}
	bzero(cdf, sizeof (ctf_dwfunc_t));
	cdf->cdf_name = name;

	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_type, &tdie)) == 0) {
		if ((ret = ctf_dwarf_convert_type(cup, tdie,
		    &(cdf->cdf_fip.ctc_return), CTF_ADD_ROOT)) != 0) {
			ctf_strfree(name);
			ctf_free(cdf, sizeof (ctf_dwfunc_t));
			return (ret);
		}
	} else if (ret != ENOENT) {
		ctf_strfree(name);
		ctf_free(cdf, sizeof (ctf_dwfunc_t));
		return (ret);
	} else {
		if ((cdf->cdf_fip.ctc_return = ctf_dwarf_void(cup)) ==
		    CTF_ERR) {
			ctf_strfree(name);
			ctf_free(cdf, sizeof (ctf_dwfunc_t));
			return (ctf_errno(cup->cu_ctfp));
		}
	}

	/*
	 * A function has a number of children, some of which may not be ones we
	 * care about. Children that we care about have a type of
	 * DW_TAG_formal_parameter. We're going to do two passes, the first to
	 * count the arguments, the second to process them. Afterwards, we
	 * should be good to go ahead and add this function.
	 *
	 * Note, we already got the return type by going in and grabbing it out
	 * of the DW_AT_type.
	 */
	if ((ret = ctf_dwarf_function_count(cup, die, &cdf->cdf_fip,
	    B_FALSE)) != 0) {
		ctf_strfree(name);
		ctf_free(cdf, sizeof (ctf_dwfunc_t));
		return (ret);
	}

	ctf_dprintf("beginning to convert function arguments %s\n", name);
	if (cdf->cdf_fip.ctc_argc != 0) {
		uint_t argc = cdf->cdf_fip.ctc_argc;
		cdf->cdf_argv = ctf_alloc(sizeof (ctf_id_t) * argc);
		if (cdf->cdf_argv == NULL) {
			ctf_strfree(name);
			ctf_free(cdf, sizeof (ctf_dwfunc_t));
			return (ENOMEM);
		}
		if ((ret = ctf_dwarf_convert_fargs(cup, die,
		    &cdf->cdf_fip, cdf->cdf_argv)) != 0) {
			ctf_free(cdf->cdf_argv, sizeof (ctf_id_t) * argc);
			ctf_strfree(name);
			ctf_free(cdf, sizeof (ctf_dwfunc_t));
			return (ret);
		}
	} else {
		cdf->cdf_argv = NULL;
	}

	if ((ret = ctf_dwarf_isglobal(cup, die, &cdf->cdf_global)) != 0) {
		ctf_free(cdf->cdf_argv, sizeof (ctf_id_t) *
		    cdf->cdf_fip.ctc_argc);
		ctf_strfree(name);
		ctf_free(cdf, sizeof (ctf_dwfunc_t));
		return (ret);
	}

	ctf_list_append(&cup->cu_funcs, cdf);
	return (ret);
}

/*
 * Convert variables, but only if they're not prototypes and have names.
 */
static int
ctf_dwarf_convert_variable(ctf_cu_t *cup, Dwarf_Die die)
{
	int ret;
	char *name;
	Dwarf_Bool b;
	Dwarf_Die tdie;
	ctf_id_t id;
	ctf_dwvar_t *cdv;

	/* Skip "Non-Defining Declarations" */
	if ((ret = ctf_dwarf_boolean(cup, die, DW_AT_declaration, &b)) == 0) {
		if (b != 0)
			return (0);
	} else if (ret != ENOENT) {
		return (ret);
	}

	/*
	 * If we find a DIE of "Declarations Completing Non-Defining
	 * Declarations", we will use the referenced type's DIE.  This isn't
	 * quite correct, e.g. DW_AT_decl_line will be the forward declaration
	 * not this site.  It's sufficient for what we need, however: in
	 * particular, we should find DW_AT_external as needed there.
	 */
	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_specification,
	    &tdie)) == 0) {
		Dwarf_Off offset;
		if ((ret = ctf_dwarf_offset(cup, tdie, &offset)) != 0)
			return (ret);
		ctf_dprintf("die 0x%llx DW_AT_specification -> die 0x%llx\n",
		    ctf_die_offset(cup, die), ctf_die_offset(cup, tdie));
		die = tdie;
	} else if (ret != ENOENT) {
		return (ret);
	}

	if ((ret = ctf_dwarf_string(cup, die, DW_AT_name, &name)) != 0 &&
	    ret != ENOENT)
		return (ret);
	if (ret == ENOENT)
		return (0);

	if ((ret = ctf_dwarf_refdie(cup, die, DW_AT_type, &tdie)) != 0) {
		ctf_strfree(name);
		return (ret);
	}

	if ((ret = ctf_dwarf_convert_type(cup, tdie, &id,
	    CTF_ADD_ROOT)) != 0)
		return (ret);

	if ((cdv = ctf_alloc(sizeof (ctf_dwvar_t))) == NULL) {
		ctf_strfree(name);
		return (ENOMEM);
	}

	cdv->cdv_name = name;
	cdv->cdv_type = id;

	if ((ret = ctf_dwarf_isglobal(cup, die, &cdv->cdv_global)) != 0) {
		ctf_free(cdv, sizeof (ctf_dwvar_t));
		ctf_strfree(name);
		return (ret);
	}

	ctf_list_append(&cup->cu_vars, cdv);
	return (0);
}

/*
 * Walk through our set of top-level types and process them.
 */
static int
ctf_dwarf_walk_toplevel(ctf_cu_t *cup, Dwarf_Die die)
{
	int ret;
	Dwarf_Off offset;
	Dwarf_Half tag;

	if ((ret = ctf_dwarf_offset(cup, die, &offset)) != 0)
		return (ret);

	if (offset > cup->cu_maxoff) {
		(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
		    "die offset %llu beyond maximum for header %llu\n",
		    offset, cup->cu_maxoff);
		return (ECTF_CONVBKERR);
	}

	if ((ret = ctf_dwarf_tag(cup, die, &tag)) != 0)
		return (ret);

	ret = 0;
	switch (tag) {
	case DW_TAG_subprogram:
		ctf_dprintf("top level func\n");
		ret = ctf_dwarf_convert_function(cup, die);
		break;
	case DW_TAG_variable:
		ctf_dprintf("top level var\n");
		ret = ctf_dwarf_convert_variable(cup, die);
		break;
	case DW_TAG_lexical_block:
		ctf_dprintf("top level block\n");
		ret = ctf_dwarf_walk_lexical(cup, die);
		break;
	case DW_TAG_enumeration_type:
	case DW_TAG_structure_type:
	case DW_TAG_typedef:
	case DW_TAG_union_type:
		ctf_dprintf("top level type\n");
		ret = ctf_dwarf_convert_type(cup, die, NULL, B_TRUE);
		break;
	default:
		break;
	}

	return (ret);
}


/*
 * We're given a node. At this node we need to convert it and then proceed to
 * convert any siblings that are associaed with this die.
 */
static int
ctf_dwarf_convert_die(ctf_cu_t *cup, Dwarf_Die die)
{
	while (die != NULL) {
		int ret;
		Dwarf_Die sib;

		if ((ret = ctf_dwarf_walk_toplevel(cup, die)) != 0)
			return (ret);

		if ((ret = ctf_dwarf_sib(cup, die, &sib)) != 0)
			return (ret);
		die = sib;
	}
	return (0);
}

static int
ctf_dwarf_fixup_die(ctf_cu_t *cup, boolean_t addpass)
{
	ctf_dwmap_t *map;

	for (map = avl_first(&cup->cu_map); map != NULL;
	    map = AVL_NEXT(&cup->cu_map, map)) {
		int ret;
		if (map->cdm_fix == B_FALSE)
			continue;
		if ((ret = ctf_dwarf_fixup_sou(cup, map->cdm_die, map->cdm_id,
		    addpass)) != 0)
			return (ret);
	}

	return (0);
}

/*
 * The DWARF information about a symbol and the information in the symbol table
 * may not be the same due to symbol reduction that is performed by ld due to a
 * mapfile or other such directive. We process weak symbols at a later time.
 *
 * The following are the rules that we employ:
 *
 * 1. A DWARF function that is considered exported matches STB_GLOBAL entries
 * with the same name.
 *
 * 2. A DWARF function that is considered exported matches STB_LOCAL entries
 * with the same name and the same file. This case may happen due to mapfile
 * reduction.
 *
 * 3. A DWARF function that is not considered exported matches STB_LOCAL entries
 * with the same name and the same file.
 *
 * 4. A DWARF function that has the same name as the symbol table entry, but the
 * files do not match. This is considered a 'fuzzy' match. This may also happen
 * due to a mapfile reduction. Fuzzy matching is only used when we know that the
 * file in question refers to the primary object. This is because when a symbol
 * is reduced in a mapfile, it's always going to be tagged as a local value in
 * the generated output and it is considered as to belong to the primary file
 * which is the first STT_FILE symbol we see.
 */
static boolean_t
ctf_dwarf_symbol_match(const char *symtab_file, const char *symtab_name,
    uint_t symtab_bind, const char *dwarf_file, const char *dwarf_name,
    boolean_t dwarf_global, boolean_t *is_fuzzy)
{
	*is_fuzzy = B_FALSE;

	if (symtab_bind != STB_LOCAL && symtab_bind != STB_GLOBAL) {
		return (B_FALSE);
	}

	if (strcmp(symtab_name, dwarf_name) != 0) {
		return (B_FALSE);
	}

	if (symtab_bind == STB_GLOBAL) {
		return (dwarf_global);
	}

	if (strcmp(symtab_file, dwarf_file) == 0) {
		return (B_TRUE);
	}

	if (dwarf_global) {
		*is_fuzzy = B_TRUE;
		return (B_TRUE);
	}

	return (B_FALSE);
}

static ctf_dwfunc_t *
ctf_dwarf_match_func(ctf_cu_t *cup, const char *file, const char *name,
    uint_t bind, boolean_t primary)
{
	ctf_dwfunc_t *cdf, *fuzzy = NULL;

	if (bind == STB_WEAK)
		return (NULL);

	if (bind == STB_LOCAL && (file == NULL || cup->cu_name == NULL))
		return (NULL);

	for (cdf = ctf_list_next(&cup->cu_funcs); cdf != NULL;
	    cdf = ctf_list_next(cdf)) {
		boolean_t is_fuzzy = B_FALSE;

		if (ctf_dwarf_symbol_match(file, name, bind, cup->cu_name,
		    cdf->cdf_name, cdf->cdf_global, &is_fuzzy)) {
			if (is_fuzzy) {
				if (primary) {
					fuzzy = cdf;
				}
				continue;
			} else {
				return (cdf);
			}
		}
	}

	return (fuzzy);
}

static ctf_dwvar_t *
ctf_dwarf_match_var(ctf_cu_t *cup, const char *file, const char *name,
    uint_t bind, boolean_t primary)
{
	ctf_dwvar_t *cdv, *fuzzy = NULL;

	if (bind == STB_WEAK)
		return (NULL);

	if (bind == STB_LOCAL && (file == NULL || cup->cu_name == NULL))
		return (NULL);

	for (cdv = ctf_list_next(&cup->cu_vars); cdv != NULL;
	    cdv = ctf_list_next(cdv)) {
		boolean_t is_fuzzy = B_FALSE;

		if (ctf_dwarf_symbol_match(file, name, bind, cup->cu_name,
		    cdv->cdv_name, cdv->cdv_global, &is_fuzzy)) {
			if (is_fuzzy) {
				if (primary) {
					fuzzy = cdv;
				}
			} else {
				return (cdv);
			}
		}
	}

	return (fuzzy);
}

static int
ctf_dwarf_conv_funcvars_cb(const Elf64_Sym *symp, ulong_t idx,
    const char *file, const char *name, boolean_t primary, void *arg)
{
	int ret;
	uint_t bind, type;
	ctf_cu_t *cup = arg;

	bind = GELF_ST_BIND(symp->st_info);
	type = GELF_ST_TYPE(symp->st_info);

	/*
	 * Come back to weak symbols in another pass
	 */
	if (bind == STB_WEAK)
		return (0);

	if (type == STT_OBJECT) {
		ctf_dwvar_t *cdv = ctf_dwarf_match_var(cup, file, name,
		    bind, primary);
		if (cdv == NULL)
			return (0);
		ret = ctf_add_object(cup->cu_ctfp, idx, cdv->cdv_type);
		ctf_dprintf("added object %s->%ld\n", name, cdv->cdv_type);
	} else {
		ctf_dwfunc_t *cdf = ctf_dwarf_match_func(cup, file, name,
		    bind, primary);
		if (cdf == NULL)
			return (0);
		ret = ctf_add_function(cup->cu_ctfp, idx, &cdf->cdf_fip,
		    cdf->cdf_argv);
		ctf_dprintf("added function %s\n", name);
	}

	if (ret == CTF_ERR) {
		return (ctf_errno(cup->cu_ctfp));
	}

	return (0);
}

static int
ctf_dwarf_conv_funcvars(ctf_cu_t *cup)
{
	return (ctf_symtab_iter(cup->cu_ctfp, ctf_dwarf_conv_funcvars_cb, cup));
}

/*
 * If we have a weak symbol, attempt to find the strong symbol it will resolve
 * to.  Note: the code where this actually happens is in sym_process() in
 * cmd/sgs/libld/common/syms.c
 *
 * Finding the matching symbol is unfortunately not trivial.  For a symbol to be
 * a candidate, it must:
 *
 * - have the same type (function, object)
 * - have the same value (address)
 * - have the same size
 * - not be another weak symbol
 * - belong to the same section (checked via section index)
 *
 * To perform this check, we first iterate over the symbol table. For each weak
 * symbol that we encounter, we then do a second walk over the symbol table,
 * calling ctf_dwarf_conv_check_weak(). If a symbol matches the above, then it's
 * either a local or global symbol. If we find a global symbol then we go with
 * it and stop searching for additional matches.
 *
 * If instead, we find a local symbol, things are more complicated. The first
 * thing we do is to try and see if we have file information about both symbols
 * (STT_FILE). If they both have file information and it matches, then we treat
 * that as a good match and stop searching for additional matches.
 *
 * Otherwise, this means we have a non-matching file and a local symbol. We
 * treat this as a candidate and if we find a better match (one of the two cases
 * above), use that instead. There are two different ways this can happen.
 * Either this is a completely different symbol, or it's a once-global symbol
 * that was scoped to local via a mapfile.  In the former case, curfile is
 * likely inaccurate since the linker does not preserve the needed curfile in
 * the order of the symbol table (see the comments about locally scoped symbols
 * in libld's update_osym()).  As we can't tell this case from the former one,
 * we use this symbol iff no other matching symbol is found.
 *
 * What we really need here is a SUNW section containing weak<->strong mappings
 * that we can consume.
 */
typedef struct ctf_dwarf_weak_arg {
	const Elf64_Sym *cweak_symp;
	const char *cweak_file;
	boolean_t cweak_candidate;
	ulong_t cweak_idx;
} ctf_dwarf_weak_arg_t;

static int
ctf_dwarf_conv_check_weak(const Elf64_Sym *symp, ulong_t idx, const char *file,
    const char *name, boolean_t primary, void *arg)
{
	ctf_dwarf_weak_arg_t *cweak = arg;

	const Elf64_Sym *wsymp = cweak->cweak_symp;

	ctf_dprintf("comparing weak to %s\n", name);

	if (GELF_ST_BIND(symp->st_info) == STB_WEAK) {
		return (0);
	}

	if (GELF_ST_TYPE(wsymp->st_info) != GELF_ST_TYPE(symp->st_info)) {
		return (0);
	}

	if (wsymp->st_value != symp->st_value) {
		return (0);
	}

	if (wsymp->st_size != symp->st_size) {
		return (0);
	}

	if (wsymp->st_shndx != symp->st_shndx) {
		return (0);
	}

	/*
	 * Check if it's a weak candidate.
	 */
	if (GELF_ST_BIND(symp->st_info) == STB_LOCAL &&
	    (file == NULL || cweak->cweak_file == NULL ||
	    strcmp(file, cweak->cweak_file) != 0)) {
		cweak->cweak_candidate = B_TRUE;
		cweak->cweak_idx = idx;
		return (0);
	}

	/*
	 * Found a match, break.
	 */
	cweak->cweak_idx = idx;
	return (1);
}

static int
ctf_dwarf_duplicate_sym(ctf_cu_t *cup, ulong_t idx, ulong_t matchidx)
{
	ctf_id_t id = ctf_lookup_by_symbol(cup->cu_ctfp, matchidx);

	/*
	 * If we matched something that for some reason didn't have type data,
	 * we don't consider that a fatal error and silently swallow it.
	 */
	if (id == CTF_ERR) {
		if (ctf_errno(cup->cu_ctfp) == ECTF_NOTYPEDAT)
			return (0);
		else
			return (ctf_errno(cup->cu_ctfp));
	}

	if (ctf_add_object(cup->cu_ctfp, idx, id) == CTF_ERR)
		return (ctf_errno(cup->cu_ctfp));

	return (0);
}

static int
ctf_dwarf_duplicate_func(ctf_cu_t *cup, ulong_t idx, ulong_t matchidx)
{
	int ret;
	ctf_funcinfo_t fip;
	ctf_id_t *args = NULL;

	if (ctf_func_info(cup->cu_ctfp, matchidx, &fip) == CTF_ERR) {
		if (ctf_errno(cup->cu_ctfp) == ECTF_NOFUNCDAT)
			return (0);
		else
			return (ctf_errno(cup->cu_ctfp));
	}

	if (fip.ctc_argc != 0) {
		args = ctf_alloc(sizeof (ctf_id_t) * fip.ctc_argc);
		if (args == NULL)
			return (ENOMEM);

		if (ctf_func_args(cup->cu_ctfp, matchidx, fip.ctc_argc, args) ==
		    CTF_ERR) {
			ctf_free(args, sizeof (ctf_id_t) * fip.ctc_argc);
			return (ctf_errno(cup->cu_ctfp));
		}
	}

	ret = ctf_add_function(cup->cu_ctfp, idx, &fip, args);
	if (args != NULL)
		ctf_free(args, sizeof (ctf_id_t) * fip.ctc_argc);
	if (ret == CTF_ERR)
		return (ctf_errno(cup->cu_ctfp));

	return (0);
}

static int
ctf_dwarf_conv_weaks_cb(const Elf64_Sym *symp, ulong_t idx, const char *file,
    const char *name, boolean_t primary, void *arg)
{
	int ret, type;
	ctf_dwarf_weak_arg_t cweak;
	ctf_cu_t *cup = arg;

	/*
	 * We only care about weak symbols.
	 */
	if (GELF_ST_BIND(symp->st_info) != STB_WEAK)
		return (0);

	type = GELF_ST_TYPE(symp->st_info);
	ASSERT(type == STT_OBJECT || type == STT_FUNC);

	/*
	 * For each weak symbol we encounter, we need to do a second iteration
	 * to try and find a match. We should probably think about other
	 * techniques to try and save us time in the future.
	 */
	cweak.cweak_symp = symp;
	cweak.cweak_file = file;
	cweak.cweak_candidate = B_FALSE;
	cweak.cweak_idx = 0;

	ctf_dprintf("Trying to find weak equiv for %s\n", name);

	ret = ctf_symtab_iter(cup->cu_ctfp, ctf_dwarf_conv_check_weak, &cweak);
	VERIFY(ret == 0 || ret == 1);

	/*
	 * Nothing was ever found, we're not going to add anything for this
	 * entry.
	 */
	if (ret == 0 && cweak.cweak_candidate == B_FALSE) {
		ctf_dprintf("found no weak match for %s\n", name);
		return (0);
	}

	/*
	 * Now, finally go and add the type based on the match.
	 */
	ctf_dprintf("matched weak symbol %lu to %lu\n", idx, cweak.cweak_idx);
	if (type == STT_OBJECT) {
		ret = ctf_dwarf_duplicate_sym(cup, idx, cweak.cweak_idx);
	} else {
		ret = ctf_dwarf_duplicate_func(cup, idx, cweak.cweak_idx);
	}

	return (ret);
}

static int
ctf_dwarf_conv_weaks(ctf_cu_t *cup)
{
	return (ctf_symtab_iter(cup->cu_ctfp, ctf_dwarf_conv_weaks_cb, cup));
}

static int
ctf_dwarf_convert_one(void *arg, void *unused)
{
	int ret;
	ctf_file_t *dedup;
	ctf_cu_t *cup = arg;
	const char *name = cup->cu_name != NULL ? cup->cu_name : "NULL";

	VERIFY(cup != NULL);

	if ((ret = ctf_dwarf_init_die(cup)) != 0)
		return (ret);

	ctf_dprintf("converting die: %s - max offset: %x\n", name,
	    cup->cu_maxoff);

	ret = ctf_dwarf_convert_die(cup, cup->cu_cu);
	ctf_dprintf("ctf_dwarf_convert_die (%s) returned %d\n", name,
	    ret);
	if (ret != 0)
		return (ret);

	if (ctf_update(cup->cu_ctfp) != 0) {
		return (ctf_dwarf_error(cup, cup->cu_ctfp, 0,
		    "failed to update output ctf container"));
	}

	ret = ctf_dwarf_fixup_die(cup, B_FALSE);
	ctf_dprintf("ctf_dwarf_fixup_die (%s, FALSE) returned %d\n", name,
	    ret);
	if (ret != 0)
		return (ret);

	if (ctf_update(cup->cu_ctfp) != 0) {
		return (ctf_dwarf_error(cup, cup->cu_ctfp, 0,
		    "failed to update output ctf container"));
	}

	ret = ctf_dwarf_fixup_die(cup, B_TRUE);
	ctf_dprintf("ctf_dwarf_fixup_die (%s, TRUE) returned %d\n", name,
	    ret);
	if (ret != 0)
		return (ret);

	if (ctf_update(cup->cu_ctfp) != 0) {
		return (ctf_dwarf_error(cup, cup->cu_ctfp, 0,
		    "failed to update output ctf container"));
	}

	if ((ret = ctf_dwarf_conv_funcvars(cup)) != 0) {
		ctf_dprintf("ctf_dwarf_conv_funcvars (%s) returned %d\n",
		    name, ret);
		return (ctf_dwarf_error(cup, NULL, ret,
		    "failed to convert strong functions and variables"));
	}

	if (ctf_update(cup->cu_ctfp) != 0) {
		return (ctf_dwarf_error(cup, cup->cu_ctfp, 0,
		    "failed to update output ctf container"));
	}

	if (cup->cu_doweaks == B_TRUE) {
		if ((ret = ctf_dwarf_conv_weaks(cup)) != 0) {
			ctf_dprintf("ctf_dwarf_conv_weaks (%s) returned %d\n",
			    name, ret);
			return (ctf_dwarf_error(cup, NULL, ret,
			    "failed to convert weak functions and variables"));
		}

		if (ctf_update(cup->cu_ctfp) != 0) {
			return (ctf_dwarf_error(cup, cup->cu_ctfp, 0,
			    "failed to update output ctf container"));
		}
	}

	ctf_phase_dump(cup->cu_ctfp, "pre-dwarf-dedup", name);
	ctf_dprintf("adding inputs for dedup\n");
	if ((ret = ctf_merge_add(cup->cu_cmh, cup->cu_ctfp)) != 0) {
		return (ctf_dwarf_error(cup, NULL, ret,
		    "failed to add inputs for merge"));
	}

	ctf_dprintf("starting dedup of %s\n", name);
	if ((ret = ctf_merge_dedup(cup->cu_cmh, &dedup)) != 0) {
		return (ctf_dwarf_error(cup, NULL, ret,
		    "failed to deduplicate die"));
	}

	ctf_close(cup->cu_ctfp);
	cup->cu_ctfp = dedup;
	ctf_phase_dump(cup->cu_ctfp, "post-dwarf-dedup", name);

	return (0);
}

static void
ctf_dwarf_free_die(ctf_cu_t *cup)
{
	ctf_dwfunc_t *cdf, *ndf;
	ctf_dwvar_t *cdv, *ndv;
	ctf_dwbitf_t *cdb, *ndb;
	ctf_dwmap_t *map;
	void *cookie;

	ctf_dprintf("Beginning to free die: %p\n", cup);

	VERIFY3P(cup->cu_elf, !=, NULL);
	cup->cu_elf = NULL;

	ctf_dprintf("Trying to free name: %p\n", cup->cu_name);
	if (cup->cu_name != NULL) {
		ctf_strfree(cup->cu_name);
		cup->cu_name = NULL;
	}

	ctf_dprintf("Trying to free merge handle: %p\n", cup->cu_cmh);
	if (cup->cu_cmh != NULL) {
		ctf_merge_fini(cup->cu_cmh);
		cup->cu_cmh = NULL;
	}

	ctf_dprintf("Trying to free functions\n");
	for (cdf = ctf_list_next(&cup->cu_funcs); cdf != NULL; cdf = ndf) {
		ndf = ctf_list_next(cdf);
		ctf_strfree(cdf->cdf_name);
		if (cdf->cdf_fip.ctc_argc != 0) {
			ctf_free(cdf->cdf_argv,
			    sizeof (ctf_id_t) * cdf->cdf_fip.ctc_argc);
		}
		ctf_free(cdf, sizeof (ctf_dwfunc_t));
	}

	ctf_dprintf("Trying to free variables\n");
	for (cdv = ctf_list_next(&cup->cu_vars); cdv != NULL; cdv = ndv) {
		ndv = ctf_list_next(cdv);
		ctf_strfree(cdv->cdv_name);
		ctf_free(cdv, sizeof (ctf_dwvar_t));
	}

	ctf_dprintf("Trying to free bitfields\n");
	for (cdb = ctf_list_next(&cup->cu_bitfields); cdb != NULL; cdb = ndb) {
		ndb = ctf_list_next(cdb);
		ctf_free(cdb, sizeof (ctf_dwbitf_t));
	}

	if (cup->cu_ctfp != NULL) {
		ctf_close(cup->cu_ctfp);
		cup->cu_ctfp = NULL;
	}

	cookie = NULL;
	while ((map = avl_destroy_nodes(&cup->cu_map, &cookie)) != NULL)
		ctf_free(map, sizeof (ctf_dwmap_t));
	avl_destroy(&cup->cu_map);
	cup->cu_errbuf = NULL;

	if (cup->cu_cu != NULL) {
		ctf_dwarf_dealloc(cup, cup->cu_cu, DW_DLA_DIE);
		cup->cu_cu = NULL;
	}
}

static int
ctf_dwarf_count_dies(Dwarf_Debug dw, Dwarf_Error *derr, uint_t *ndies,
    char *errbuf, size_t errlen)
{
	int ret;
	Dwarf_Half vers;
	Dwarf_Unsigned nexthdr;

	while ((ret = dwarf_next_cu_header(dw, NULL, &vers, NULL, NULL,
	    &nexthdr, derr)) != DW_DLV_NO_ENTRY) {
		if (ret != DW_DLV_OK) {
			(void) snprintf(errbuf, errlen,
			    "file does not contain valid DWARF data: %s\n",
			    dwarf_errmsg(*derr));
			return (ECTF_CONVBKERR);
		}

		switch (vers) {
		case DWARF_VERSION_TWO:
		case DWARF_VERSION_FOUR:
			break;
		default:
			(void) snprintf(errbuf, errlen,
			    "unsupported DWARF version: %d\n", vers);
			return (ECTF_CONVBKERR);
		}
		*ndies = *ndies + 1;
	}

	return (0);
}

/*
 * Fill out just enough of each ctf_cu_t for the conversion process to
 * be able to finish the rest in a (potentially) multithreaded context.
 */
static int
ctf_dwarf_preinit_dies(ctf_convert_t *cch, int fd, Elf *elf, Dwarf_Debug dw,
    mutex_t *dwlock, Dwarf_Error *derr, uint_t ndies, ctf_cu_t *cdies,
    char *errbuf, size_t errlen)
{
	Dwarf_Unsigned hdrlen, abboff, nexthdr;
	Dwarf_Half addrsz, vers;
	Dwarf_Unsigned offset = 0;
	uint_t added = 0;
	int ret, i = 0;

	while ((ret = dwarf_next_cu_header(dw, &hdrlen, &vers, &abboff,
	    &addrsz, &nexthdr, derr)) != DW_DLV_NO_ENTRY) {
		Dwarf_Die cu;
		ctf_cu_t *cup;
		char *name;

		VERIFY3U(i, <, ndies);

		cup = &cdies[i++];

		cup->cu_handle = cch;
		cup->cu_fd = fd;
		cup->cu_elf = elf;
		cup->cu_dwarf = dw;
		cup->cu_errbuf = errbuf;
		cup->cu_errlen = errlen;
		cup->cu_dwarf = dw;
		if (ndies > 1) {
			/*
			 * Only need to lock calls into libdwarf if there are
			 * multiple CUs.
			 */
			cup->cu_dwlock = dwlock;
			cup->cu_doweaks = B_FALSE;
		} else {
			cup->cu_doweaks = B_TRUE;
		}

		cup->cu_voidtid = CTF_ERR;
		cup->cu_longtid = CTF_ERR;
		cup->cu_cuoff = offset;
		cup->cu_maxoff = nexthdr - 1;
		cup->cu_vers = vers;
		cup->cu_addrsz = addrsz;

		if ((ret = ctf_dwarf_sib(cup, NULL, &cu)) != 0) {
			ctf_dprintf("cu %d - no cu %d\n", i, ret);
			return (ret);
		}

		if (cu == NULL) {
			ctf_dprintf("cu %d - no cu data\n", i);
			(void) snprintf(cup->cu_errbuf, cup->cu_errlen,
			    "file does not contain DWARF data\n");
			return (ECTF_CONVNODEBUG);
		}

		if (ctf_dwarf_string(cup, cu, DW_AT_name, &name) == 0) {
			char *b = basename(name);

			cup->cu_name = strdup(b);
			ctf_strfree(name);
			if (cup->cu_name == NULL)
				return (ENOMEM);
		}

		ret = ctf_dwarf_child(cup, cu, &cup->cu_cu);
		dwarf_dealloc(cup->cu_dwarf, cu, DW_DLA_DIE);
		if (ret != 0) {
			ctf_dprintf("cu %d - no child '%s' %d\n",
			    i, cup->cu_name != NULL ? cup->cu_name : "NULL",
			    ret);
			return (ret);
		}

		if (cup->cu_cu == NULL) {
			size_t len;

			ctf_dprintf("cu %d - no child data '%s' %d\n",
			    i, cup->cu_name != NULL ? cup->cu_name : "NULL",
			    ret);
			if (cup->cu_name != NULL &&
			    (len = strlen(cup->cu_name)) > 2 &&
			    strncmp(".c", &cup->cu_name[len - 2], 2) == 0) {
				/*
				 * Missing DEBUG data for a .c file, return an
				 * error unless this is permitted.
				 */
				if (!(cch->cch_flags &
				    CTF_ALLOW_MISSING_DEBUG)) {
					(void) snprintf(
					    cup->cu_errbuf, cup->cu_errlen,
					    "missing debug information "
					    "(first seen in %s)\n",
					    cup->cu_name);
					return (ECTF_CONVNODEBUG);
				}
				if (cch->cch_warncb != NULL) {
					cch->cch_warncb(cch->cch_warncb_arg,
					    "file %s is missing debug "
					    "information\n", cup->cu_name);
				}
			}
		} else {
			added++;
		}

		ctf_dprintf("Pre-initialised cu %d - '%s'\n", i,
		    cup->cu_name != NULL ? cup->cu_name : "NULL");

		offset = nexthdr;
	}

	/*
	 * If none of the CUs had debug data, return an error.
	 */
	if (added == 0)
		return (ECTF_CONVNODEBUG);

	return (0);
}

static int
ctf_dwarf_init_die(ctf_cu_t *cup)
{
	int ret;

	cup->cu_ctfp = ctf_fdcreate(cup->cu_fd, &ret);
	if (cup->cu_ctfp == NULL)
		return (ret);

	avl_create(&cup->cu_map, ctf_dwmap_comp, sizeof (ctf_dwmap_t),
	    offsetof(ctf_dwmap_t, cdm_avl));

	if ((ret = ctf_dwarf_die_elfenc(cup->cu_elf, cup,
	    cup->cu_errbuf, cup->cu_errlen)) != 0) {
		return (ret);
	}

	if ((cup->cu_cmh = ctf_merge_init(cup->cu_fd, &ret)) == NULL)
		return (ret);

	return (0);
}

/*
 * This is our only recourse to identify a C source file that is missing debug
 * info: it will be mentioned as an STT_FILE, but not have a compile unit entry.
 * (A traditional ctfmerge works on individual files, so can identify missing
 * DWARF more directly, via ctf_has_c_source() on the .o file.)
 *
 * As we operate on basenames, this can of course miss some cases, but it's
 * better than not checking at all.
 *
 * We explicitly whitelist some CRT components.  Failing that, there's always
 * the -m option.
 */
static boolean_t
c_source_has_debug(ctf_convert_t *cch, const char *file,
    ctf_cu_t *cus, size_t nr_cus)
{
	const char *basename = strrchr(file, '/');
	ctf_convert_filelist_t *ccf;

	if (basename == NULL)
		basename = file;
	else
		basename++;

	if (strcmp(basename, "common-crt.c") == 0 ||
	    strcmp(basename, "gmon.c") == 0 ||
	    strcmp(basename, "dlink_init.c") == 0 ||
	    strcmp(basename, "dlink_common.c") == 0 ||
	    strcmp(basename, "ssp_ns.c") == 0 ||
	    strncmp(basename, "crt", strlen("crt")) == 0 ||
	    strncmp(basename, "values-", strlen("values-")) == 0)
		return (B_TRUE);

	for (ccf = list_head(&cch->cch_nodebug); ccf != NULL;
	    ccf = list_next(&cch->cch_nodebug, ccf)) {
		if (ccf->ccf_basename != NULL &&
		    strcmp(basename, ccf->ccf_basename) == 0) {
			return (B_TRUE);
		}
	}

	for (size_t i = 0; i < nr_cus; i++) {
		if (cus[i].cu_name != NULL &&
		    strcmp(basename, cus[i].cu_name) == 0) {
			return (B_TRUE);
		}
	}

	return (B_FALSE);
}

static int
ctf_dwarf_check_missing(ctf_convert_t *cch, ctf_cu_t *cus, size_t nr_cus,
    Elf *elf, char *errmsg, size_t errlen)
{
	Elf_Scn *scn, *strscn;
	Elf_Data *data, *strdata;
	GElf_Shdr shdr;
	ulong_t i;
	int ret = 0;

	scn = NULL;
	while ((scn = elf_nextscn(elf, scn)) != NULL) {
		if (gelf_getshdr(scn, &shdr) == NULL) {
			(void) snprintf(errmsg, errlen,
			    "failed to get section header: %s\n",
			    elf_errmsg(elf_errno()));
			return (EINVAL);
		}

		if (shdr.sh_type == SHT_SYMTAB)
			break;
	}

	if (scn == NULL)
		return (0);

	if ((strscn = elf_getscn(elf, shdr.sh_link)) == NULL) {
		(void) snprintf(errmsg, errlen,
		    "failed to get str section: %s\n",
		    elf_errmsg(elf_errno()));
		return (EINVAL);
	}

	if ((data = elf_getdata(scn, NULL)) == NULL) {
		(void) snprintf(errmsg, errlen, "failed to read section: %s\n",
		    elf_errmsg(elf_errno()));
		return (EINVAL);
	}

	if ((strdata = elf_getdata(strscn, NULL)) == NULL) {
		(void) snprintf(errmsg, errlen,
		    "failed to read string table: %s\n",
		    elf_errmsg(elf_errno()));
		return (EINVAL);
	}

	for (i = 0; i < shdr.sh_size / shdr.sh_entsize; i++) {
		GElf_Sym sym;
		const char *file;
		size_t len;

		if (gelf_getsym(data, i, &sym) == NULL) {
			(void) snprintf(errmsg, errlen,
			    "failed to read sym %lu: %s\n",
			    i, elf_errmsg(elf_errno()));
			return (EINVAL);
		}

		if (GELF_ST_TYPE(sym.st_info) != STT_FILE)
			continue;

		file = (const char *)((uintptr_t)strdata->d_buf + sym.st_name);
		len = strlen(file);
		if (len < 2 || strncmp(".c", &file[len - 2], 2) != 0)
			continue;

		if (!c_source_has_debug(cch, file, cus, nr_cus)) {
			if (cch->cch_warncb != NULL) {
				cch->cch_warncb(
				    cch->cch_warncb_arg,
				    "file %s is missing debug information\n",
				    file);
			}
			if (ret != ECTF_CONVNODEBUG) {
				(void) snprintf(errmsg, errlen,
				    "missing debug information "
				    "(first seen in %s)\n", file);
				ret = ECTF_CONVNODEBUG;
			}
		}
	}

	return (ret);
}

static int
ctf_dwarf_convert_batch(uint_t start, uint_t end, int fd, uint_t nthrs,
    workq_t *wqp, ctf_cu_t *cdies, ctf_file_t **fpp)
{
	ctf_file_t *fpprev = NULL;
	uint_t i, added;
	ctf_cu_t *cup;
	int ret, err;

	ctf_dprintf("Processing CU batch %u - %u\n", start, end - 1);

	added = 0;
	for (i = start; i < end; i++) {
		cup = &cdies[i];
		if (cup->cu_cu == NULL)
			continue;
		ctf_dprintf("adding cu %s: %p, %x %x\n",
		    cup->cu_name != NULL ? cup->cu_name : "NULL",
		    cup->cu_cu, cup->cu_cuoff, cup->cu_maxoff);
		if (workq_add(wqp, cup) == -1) {
			err = errno;
			goto out;
		}
		added++;
	}

	/*
	 * No debug data found in this batch, move on to the next.
	 * NB:	ctf_dwarf_preinit_dies() has already checked that there is at
	 *	least one CU with debug data present.
	 */
	if (added == 0) {
		err = 0;
		goto out;
	}

	ctf_dprintf("Running conversion phase\n");

	/* Run the conversions */
	ret = workq_work(wqp, ctf_dwarf_convert_one, NULL, &err);
	if (ret == WORKQ_ERROR) {
		err = errno;
		goto out;
	} else if (ret == WORKQ_UERROR) {
		ctf_dprintf("internal convert failed: %s\n",
		    ctf_errmsg(err));
		goto out;
	}

	ctf_dprintf("starting merge phase\n");

	ctf_merge_t *cmp = ctf_merge_init(fd, &err);
	if (cmp == NULL)
		goto out;

	if ((err = ctf_merge_set_nthreads(cmp, nthrs)) != 0) {
		ctf_merge_fini(cmp);
		goto out;
	}

	/*
	 * If we have the result of a previous merge then add it as an input to
	 * the next one.
	 */
	if (*fpp != NULL) {
		ctf_dprintf("adding previous merge CU\n");
		fpprev = *fpp;
		*fpp = NULL;
		if ((err = ctf_merge_add(cmp, fpprev)) != 0) {
			ctf_merge_fini(cmp);
			goto out;
		}
	}

	ctf_dprintf("adding CUs to merge\n");
	for (i = start; i < end; i++) {
		cup = &cdies[i];
		if (cup->cu_cu == NULL)
			continue;
		if ((err = ctf_merge_add(cmp, cup->cu_ctfp)) != 0) {
			ctf_merge_fini(cmp);
			*fpp = NULL;
			goto out;
		}
	}

	ctf_dprintf("performing merge\n");
	err = ctf_merge_merge(cmp, fpp);
	if (err != 0) {
		ctf_dprintf("failed merge!\n");
		*fpp = NULL;
		ctf_merge_fini(cmp);
		goto out;
	}

	ctf_merge_fini(cmp);

	ctf_dprintf("freeing CUs\n");
	for (i = start; i < end; i++) {
		cup = &cdies[i];
		ctf_dprintf("freeing cu %d\n", i);
		ctf_dwarf_free_die(cup);
	}

out:
	ctf_close(fpprev);
	return (err);
}

int
ctf_dwarf_convert(ctf_convert_t *cch, int fd, Elf *elf, ctf_file_t **fpp,
    char *errbuf, size_t errlen)
{
	int err, ret;
	uint_t ndies, i, bsize, nthrs;
	Dwarf_Debug dw;
	Dwarf_Error derr;
	ctf_cu_t *cdies = NULL, *cup;
	workq_t *wqp = NULL;
	mutex_t dwlock = ERRORCHECKMUTEX;

	*fpp = NULL;

	ret = dwarf_elf_init(elf, DW_DLC_READ, NULL, NULL, &dw, &derr);
	if (ret != DW_DLV_OK) {
		if (ret == DW_DLV_NO_ENTRY ||
		    dwarf_errno(derr) == DW_DLE_DEBUG_INFO_NULL) {
			(void) snprintf(errbuf, errlen,
			    "file does not contain DWARF data\n");
			return (ECTF_CONVNODEBUG);
		}

		(void) snprintf(errbuf, errlen,
		    "dwarf_elf_init() failed: %s\n", dwarf_errmsg(derr));
		return (ECTF_CONVBKERR);
	}

	/*
	 * Iterate over all of the compilation units and create a ctf_cu_t for
	 * each of them.  This is used to determine if we have zero, one, or
	 * multiple dies to convert. If we have zero, that's an error. If
	 * there's only one die, that's the simple case.  No merge needed and
	 * only a single Dwarf_Debug as well.
	 */
	ndies = 0;
	err = ctf_dwarf_count_dies(dw, &derr, &ndies, errbuf, errlen);

	ctf_dprintf("found %d DWARF CUs\n", ndies);

	if (ndies == 0) {
		(void) snprintf(errbuf, errlen,
		    "file does not contain DWARF data\n");
		(void) dwarf_finish(dw, &derr);
		return (ECTF_CONVNODEBUG);
	}

	cdies = ctf_alloc(sizeof (ctf_cu_t) * ndies);
	if (cdies == NULL) {
		(void) dwarf_finish(dw, &derr);
		return (ENOMEM);
	}

	bzero(cdies, sizeof (ctf_cu_t) * ndies);

	if ((err = ctf_dwarf_preinit_dies(cch, fd, elf, dw, &dwlock, &derr,
	    ndies, cdies, errbuf, errlen)) != 0) {
		goto out;
	}

	if ((err = ctf_dwarf_check_missing(cch, cdies, ndies, elf,
	    errbuf, errlen)) != 0) {
		if (!(cch->cch_flags & CTF_ALLOW_MISSING_DEBUG)) {
			goto out;
		}
		if (err != ECTF_CONVNODEBUG && *errbuf != '\0' &&
		    cch->cch_warncb != NULL) {
			cch->cch_warncb(cch->cch_warncb_arg, "%s", errbuf);
			*errbuf = '\0';
		}
	}

	/* Only one cu, no merge required */
	if (ndies == 1) {
		cup = cdies;

		if ((err = ctf_dwarf_convert_one(cup, NULL)) != 0)
			goto out;

		*fpp = cup->cu_ctfp;
		cup->cu_ctfp = NULL;
		ctf_dwarf_free_die(cup);
		goto success;
	}

	/*
	 * There's no need to have either more threads or a batch size larger
	 * than the total number of dies, even if the user requested them.
	 */
	nthrs = min(ndies, cch->cch_nthreads);
	bsize = min(ndies, cch->cch_batchsize);

	if (workq_init(&wqp, nthrs) == -1) {
		err = errno;
		goto out;
	}

	/*
	 * In order to avoid exhausting memory limits when converting files
	 * with a large number of dies, we process them in batches.
	 */
	for (i = 0; i < ndies; i += bsize) {
		err = ctf_dwarf_convert_batch(i, min(i + bsize, ndies),
		    fd, nthrs, wqp, cdies, fpp);
		if (err != 0) {
			*fpp = NULL;
			goto out;
		}
	}

success:
	err = 0;
	ctf_dprintf("successfully converted!\n");

out:
	(void) dwarf_finish(dw, &derr);
	workq_fini(wqp);
	ctf_free(cdies, sizeof (ctf_cu_t) * ndies);
	return (err);
}
