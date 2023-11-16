/*
 * This file is part of cparser.
 * Copyright (C) 2013 Matthias Braun <matze@braunis.de>
 */
#include "predefs.h"

#include <stdarg.h>
#include <string.h>

#include "adt/panic.h"
#include "adt/strutil.h"
#include "ast/constfoldbits.h"
#include "ast/dialect.h"
#include "ast/type_t.h"
#include "ast/types.h"
#include "firm/ast2firm.h"
#include "firm/firm_opt.h"
#include "parser/preprocessor.h"
#include "target.h"
#include "version.h"

static void add_define_prop_fmt(const char *name_template, const char *name,
                                const char *value_fmt, ...)
{
	char name_prop[64];
	snprintf(name_prop, sizeof(name_prop), name_template, name);

	va_list ap;
	va_start(ap, value_fmt);
	char value[128];
	vsnprintf(value, sizeof(value), value_fmt, ap);
	add_define(name_prop, value, false);
	va_end(ap);
}

static const char *get_max_string(atomic_type_kind_t akind)
{
	/* float not implemented yet */
	unsigned flags = get_atomic_type_flags(akind);
	assert(flags & ATOMIC_TYPE_FLAG_INTEGER);

	unsigned bits = get_atomic_type_size(akind) * BITS_PER_BYTE;
	if (flags & ATOMIC_TYPE_FLAG_SIGNED)
		--bits;
	switch (bits) {
	case 7:  return "127";
	case 8:  return "255";
	case 15: return "32767";
	case 16: return "65535";
	case 31: return "2147483647";
	case 32: return "4294967295";
	case 63: return "9223372036854775807";
	case 64: return "18446744073709551615";
	}
	panic("unexpected number of bits requested");
}

static const char *get_literal_suffix(atomic_type_kind_t kind)
{
	switch (kind) {
	case ATOMIC_TYPE_BOOL:
	case ATOMIC_TYPE_CHAR:
	case ATOMIC_TYPE_SCHAR:
	case ATOMIC_TYPE_UCHAR:
	case ATOMIC_TYPE_SHORT:
	case ATOMIC_TYPE_USHORT:
	case ATOMIC_TYPE_INT:
	case ATOMIC_TYPE_DOUBLE:
	case ATOMIC_TYPE_WCHAR_T:
		return "";
	case ATOMIC_TYPE_UINT:        return "U";
	case ATOMIC_TYPE_LONG:        return "L";
	case ATOMIC_TYPE_ULONG:       return "UL";
	case ATOMIC_TYPE_LONGLONG:    return "LL";
	case ATOMIC_TYPE_ULONGLONG:   return "ULL";
	case ATOMIC_TYPE_FLOAT:       return "F";
	case ATOMIC_TYPE_LONG_DOUBLE: return "L";
	}
	panic("invalid kind in get_literal_suffix");
}

static void define_type_max(const char *name, atomic_type_kind_t akind)
{
	add_define_prop_fmt("__%s_MAX__", name, "%s%s", get_max_string(akind),
	                    get_literal_suffix(akind));
}

static void define_type_min(const char *name, atomic_type_kind_t akind)
{
	unsigned flags = get_atomic_type_flags(akind);
	if (flags & ATOMIC_TYPE_FLAG_SIGNED) {
		/* float not implemented yet */
		assert(flags & ATOMIC_TYPE_FLAG_INTEGER);
		add_define_prop_fmt("__%s_MIN__", name, "(-__%s_MAX__ - 1)", name);
	} else {
		add_define_prop_fmt("__%s_MIN__", name, "0%s",
		                    get_literal_suffix(akind));
	}
}

static void define_type_type_max(const char *name, atomic_type_kind_t akind)
{
	const char *type = get_atomic_kind_name(akind);
	add_define_prop_fmt("__%s_TYPE__", name, "%s", type);
	define_type_max(name, akind);
}

static void add_define_int(const char *name, int value)
{
	add_define_prop_fmt("%s", name, "%d", value);
}

static void define_sizeof(const char *name, atomic_type_kind_t akind)
{
	int size = get_atomic_type_size(akind);
	add_define_prop_fmt("__SIZEOF_%s__", name, "%d", size);
}

static void define_type_c(const char *name, atomic_type_kind_t akind)
{
	char buf[32];
	const char *suffix = get_literal_suffix(akind);
	const char *val;
	if (suffix[0] != '\0') {
		snprintf(buf, sizeof(buf), "c ## %s", suffix);
		val = buf;
	} else {
		val = "c";
	}
	add_define_macro(name, "c", val, false);
}

static void define_int_n_types(unsigned size, atomic_type_kind_t unsigned_kind,
                               atomic_type_kind_t signed_kind)
{
	char buf[32];

	assert(size == get_atomic_type_size(signed_kind) * BITS_PER_BYTE);
	snprintf(buf, sizeof(buf), "INT%u", size);
	define_type_type_max(buf, signed_kind);
	snprintf(buf, sizeof(buf), "__INT%u_C", size);
	define_type_c(buf, signed_kind);
	snprintf(buf, sizeof(buf), "INT_LEAST%u", size);
	define_type_type_max(buf, signed_kind);
	snprintf(buf, sizeof(buf), "INT_FAST%u", size);
	define_type_type_max(buf, signed_kind);

	assert(size == get_atomic_type_size(unsigned_kind) * BITS_PER_BYTE);
	snprintf(buf, sizeof(buf), "UINT%u", size);
	define_type_type_max(buf, unsigned_kind);
	snprintf(buf, sizeof(buf), "__UINT%u_C", size);
	define_type_c(buf, unsigned_kind);
	snprintf(buf, sizeof(buf), "UINT_LEAST%u", size);
	define_type_type_max(buf, unsigned_kind);
	snprintf(buf, sizeof(buf), "UINT_FAST%u", size);
	define_type_type_max(buf, unsigned_kind);
}

static void define_float_properties(const char *prefix,
                                    atomic_type_kind_t akind)
{
	typedef struct float_properties {
		char const *mant_digs;
		char const *digs;
		char const *min_exps;
		char const *min_10_exps;
		char const *decimal_digs;
		char const *max_exps;
		char const *max_10_exps;
		char const *maxs;
		char const *mins;
		char const *epsilons;
		char const *denorm_mins;
	} float_properties;

	float_properties   const *p;
	ir_mode           *const  mode      = atomic_modes[akind];
	ir_mode_arithmetic const  arith     = get_mode_arithmetic(mode);
	unsigned           const  exp_size  = get_mode_exponent_size(mode);
	unsigned           const  mant_size = get_mode_mantissa_size(mode);
	if (arith == irma_ieee754 && exp_size == 8 && mant_size == 23) {
		static float_properties const prop_f32 = {
			.mant_digs    = "24",
			.digs         = "6",
			.min_exps     = "(-125)",
			.min_10_exps  = "(-37)",
			.decimal_digs = "9",
			.max_exps     = "128",
			.max_10_exps  = "38",
			.maxs         = "3.40282346638528859812e+38F",
			.mins         = "1.17549435082228750797e-38F",
			.epsilons     = "1.19209289550781250000e-7F",
			.denorm_mins  = "1.40129846432481707092e-45F",
		};
		p = &prop_f32;
	} else if (arith == irma_ieee754 && exp_size == 11 && mant_size == 52) {
		static float_properties const prop_f64 = {
			.mant_digs    = "53",
			.digs         = "15",
			.min_exps     = "(-1021)",
			.min_10_exps  = "(-307)",
			.decimal_digs = "17",
			.max_exps     = "1024",
			.max_10_exps  = "308",
			.maxs         = "((double)1.79769313486231570815e+308L)",
			.mins         = "((double)2.22507385850720138309e-308L)",
			.epsilons     = "((double)2.22044604925031308085e-16L)",
			.denorm_mins  = "((double)4.94065645841246544177e-324L)",
		};
		p = &prop_f64;
	} else if (arith == irma_x86_extended_float && exp_size == 15 && mant_size == 64) {
		static float_properties const prop_f80 = {
			.mant_digs    = "64",
			.digs         = "18",
			.min_exps     = "(-16381)",
			.min_10_exps  = "(-4931)",
			.decimal_digs = "21",
			.max_exps     = "16384",
			.max_10_exps  = "4932",
			.maxs         = "1.18973149535723176502e+4932L",
			.mins         = "3.36210314311209350626e-4932L",
			.epsilons     = "1.08420217248550443401e-19L",
			.denorm_mins  = "3.64519953188247460253e-4951L",
		};
		p = &prop_f80;
	} else if (arith == irma_ieee754 && exp_size == 15 && mant_size == 112) {
		static float_properties const prop_f128 = {
			.mant_digs    = "113",
			.digs         = "33",
			.min_exps     = "(-16381)",
			.min_10_exps  = "(-4931)",
			.decimal_digs = "36",
			.max_exps     = "16384",
			.max_10_exps  = "4932",
			.maxs         = "1.189731495357231765085759326628007016E+4932L",
			.mins         = "3.36210314311209350626267781732175260e-4932L",
			.epsilons     = "1.92592994438723585305597794258492732e-34L",
			.denorm_mins  = "6.47517511943802511092443895822764655e-4966L",
		};
		p = &prop_f128;
	} else {
		panic("unexpected long double mode");
	}

	add_define_prop_fmt("__%s_MANT_DIG__",      prefix, "%s", p->mant_digs);
	add_define_prop_fmt("__%s_DIG__",           prefix, "%s", p->digs);
	add_define_prop_fmt("__%s_MIN_EXP__",       prefix, "%s", p->min_exps);
	add_define_prop_fmt("__%s_MIN_10_EXP__",    prefix, "%s", p->min_10_exps);
	add_define_prop_fmt("__%s_MAX_EXP__",       prefix, "%s", p->max_exps);
	add_define_prop_fmt("__%s_MAX_10_EXP__",    prefix, "%s", p->max_10_exps);
	add_define_prop_fmt("__%s_DECIMAL_DIG__",   prefix, "%s", p->decimal_digs);
	add_define_prop_fmt("__%s_MAX__",           prefix, "%s", p->maxs);
	add_define_prop_fmt("__%s_MIN__",           prefix, "%s", p->mins);
	add_define_prop_fmt("__%s_EPSILON__",       prefix, "%s", p->epsilons);
	add_define_prop_fmt("__%s_DENORM_MIN__",    prefix, "%s", p->denorm_mins);
	add_define_prop_fmt("__%s_HAS_DENORM__",    prefix, "1");
	add_define_prop_fmt("__%s_HAS_INFINITY__",  prefix, "1");
	add_define_prop_fmt("__%s_HAS_QUIET_NAN__", prefix, "1");
}

static bool is_ILP(unsigned const size_int, unsigned const size_long, unsigned const size_pointer)
{
	return
		get_atomic_type_size(ATOMIC_TYPE_INT)  == size_int &&
		get_atomic_type_size(ATOMIC_TYPE_LONG) == size_long &&
		get_ctype_size(type_void_ptr)          == size_pointer;
}

void add_predefined_macros(void)
{
	add_define("__STDC__", "1", true);
	/* C99 predefined macros, but defining them for other language standards too
	 * shouldn't hurt */
	add_define("__STDC_HOSTED__", dialect.freestanding ? "0" : "1", true);

	if (dialect.c99)
		add_define("__STDC_VERSION__", "199901L", true);
	if (dialect.cpp)
		add_define("__cplusplus", "1", true);
	if (!dialect.gnu && !dialect.ms && !dialect.cpp)
		add_define("__STRICT_ANSI__", "1", false);

	add_define_string("__VERSION__", CPARSER_VERSION, false);

	/* we are cparser */
	add_define("__CPARSER__",            CPARSER_MAJOR, false);
	add_define("__CPARSER_MINOR__",      CPARSER_MINOR, false);
	add_define("__CPARSER_PATCHLEVEL__", CPARSER_PATCHLEVEL, false);

	/* let's pretend we are a GCC compiler */
	add_define("__GNUC__",            "4", false);
	add_define("__GNUC_MINOR__",      "2", false);
	add_define("__GNUC_PATCHLEVEL__", "0", false);
	if (dialect.cpp)
		add_define("__GNUG__", "4", false);

	if (!firm_is_inlining_enabled())
		add_define("__NO_INLINE__", "1", false);
	if (dialect.c99) {
		add_define("__GNUC_STDC_INLINE__", "1", false);
	} else {
		add_define("__GNUC_GNU_INLINE__", "1", false);
	}

	/* TODO: I'd really like to enable these, but for now they enable some form
	 * of x87 inline assembly in the glibc/linux headers which we don't support
	 * yet */
#if 0
	if (opt_level != OPT_O0 && opt_level != OPT_Og)
		add_define("__OPTIMIZE__", "1", false);
	if (opt_level == OPT_Os || opt_level == OPT_Oz)
		add_define("__OPTIMIZE_SIZE__", "1", false);
#endif

	/* no support for the XXX_chk functions in cparser yet */
	add_define("_FORTIFY_SOURCE", "0", false);

	char const *const big    = "__ORDER_BIG_ENDIAN__";
	char const *const little = "__ORDER_LITTLE_ENDIAN__";
	add_define(big,    "4321", false);
	add_define(little, "1234", false);
	add_define("__ORDER_PDP_ENDIAN__", "3412", false);
	char const *const order = target.byte_order_big_endian ? big : little;
	add_define("__BYTE_ORDER__",       order, false);
	add_define("__FLOAT_WORD_ORDER__", order, false);

	add_define("__FINITE_MATH_ONLY__", "0", false);

	if (is_ILP(4, 8, 8)) {
		add_define("_LP64",    "1", false);
		add_define("__LP64__", "1", false);
	} else if (is_ILP(4, 4, 4)) {
		add_define("_ILP32",    "1", false);
		add_define("__ILP32__", "1", false);
	}

	ir_mode *float_mode = ir_target_float_arithmetic_mode();
	const char *flt_eval_metod
		= float_mode == NULL ? "0"
		: get_mode_size_bytes(float_mode) > get_ctype_size(type_double)  ? "2"
		: get_mode_size_bytes(float_mode) == get_ctype_size(type_double) ? "1"
		: "-1";
	add_define("__FLT_EVAL_METHOD__", flt_eval_metod, false);

	char user_label_prefix_str[] = { target.user_label_prefix, '\0' };
	add_define("__USER_LABEL_PREFIX__", user_label_prefix_str, false);
	add_define("__REGISTER_PREFIX__", "", false);
	/* TODO: GCC_HAVE_SYNC_COMPARE_AND_SWAP_XX */

	atomic_type_properties_t *props = atomic_type_properties;
	if (!(props[ATOMIC_TYPE_CHAR].flags & ATOMIC_TYPE_FLAG_SIGNED))
		add_define("__CHAR_UNSIGNED__", "1", false);
	if (!(props[ATOMIC_TYPE_WCHAR_T].flags & ATOMIC_TYPE_FLAG_SIGNED))
		add_define("__WCHAR_UNSIGNED__", "1", false);
	add_define_int("__CHAR_BIT__", BITS_PER_BYTE);

	assert(type_size_t->kind    == TYPE_ATOMIC);
	assert(type_wint_t->kind    == TYPE_ATOMIC);
	assert(type_ptrdiff_t->kind == TYPE_ATOMIC);

	define_sizeof("SHORT",       ATOMIC_TYPE_SHORT);
	define_sizeof("INT",         ATOMIC_TYPE_INT);
	define_sizeof("LONG",        ATOMIC_TYPE_LONG);
	define_sizeof("LONG_LONG",   ATOMIC_TYPE_LONGLONG);
	define_sizeof("FLOAT",       ATOMIC_TYPE_FLOAT);
	define_sizeof("DOUBLE",      ATOMIC_TYPE_DOUBLE);
	define_sizeof("LONG_DOUBLE", ATOMIC_TYPE_LONG_DOUBLE);
	define_sizeof("SIZE_T",      type_size_t->atomic.akind);
	define_sizeof("WCHAR_T",     type_wchar_t->atomic.akind);
	define_sizeof("WINT_T",      type_wint_t->atomic.akind);
	define_sizeof("PTRDIFF_T",   type_ptrdiff_t->atomic.akind);
	add_define_int("__SIZEOF_POINTER__", get_ctype_size(type_void_ptr));

	define_type_max("SCHAR",     ATOMIC_TYPE_SCHAR);
	define_type_max("SHRT",      ATOMIC_TYPE_SHORT);
	define_type_max("INT",       ATOMIC_TYPE_INT);
	define_type_max("LONG",      ATOMIC_TYPE_LONG);
	define_type_max("LONG_LONG", ATOMIC_TYPE_LONGLONG);

	define_type_type_max("WCHAR",   type_wchar_t->atomic.akind);
	define_type_min(     "WCHAR",   type_wchar_t->atomic.akind);
	define_type_type_max("SIZE",    type_size_t->atomic.akind);
	define_type_type_max("WINT",    type_wint_t->atomic.akind);
	define_type_min(     "WINT",    type_wint_t->atomic.akind);
	define_type_type_max("PTRDIFF", type_ptrdiff_t->atomic.akind);

	/* TODO: what to choose here... */
	define_type_type_max("SIG_ATOMIC", ATOMIC_TYPE_INT);
	define_type_min(     "SIG_ATOMIC", ATOMIC_TYPE_INT);

	define_int_n_types(8,  ATOMIC_TYPE_UCHAR,  ATOMIC_TYPE_SCHAR);
	define_int_n_types(16, ATOMIC_TYPE_USHORT, ATOMIC_TYPE_SHORT);
	define_int_n_types(32, ATOMIC_TYPE_UINT,   ATOMIC_TYPE_INT);
	atomic_type_kind_t akind_uintptr;
	atomic_type_kind_t akind_intptr;
	if (get_ctype_size(type_void_ptr) == 4
	 && get_atomic_type_size(ATOMIC_TYPE_INT) == 4) {
		akind_intptr = ATOMIC_TYPE_INT;
		akind_uintptr = ATOMIC_TYPE_UINT;
	} else if (get_ctype_size(type_void_ptr) == 8
	        && get_atomic_type_size(ATOMIC_TYPE_LONG) == 8) {
		akind_intptr = ATOMIC_TYPE_LONG;
		akind_uintptr = ATOMIC_TYPE_ULONG;
	} else if (get_ctype_size(type_void_ptr) == 8
	        && get_atomic_type_size(ATOMIC_TYPE_LONGLONG) == 8) {
		akind_intptr = ATOMIC_TYPE_LONGLONG;
		akind_uintptr = ATOMIC_TYPE_ULONGLONG;
	} else {
		panic("Couldn't determine uintptr type for target");
	}
	define_type_type_max("UINTPTR", akind_uintptr);
	define_type_type_max("INTPTR",  akind_intptr);
	if (props[ATOMIC_TYPE_LONG].size == 8) {
		define_int_n_types(64, ATOMIC_TYPE_ULONG, ATOMIC_TYPE_LONG);
	} else if (props[ATOMIC_TYPE_LONGLONG].size == 8) {
		define_int_n_types(64, ATOMIC_TYPE_ULONGLONG, ATOMIC_TYPE_LONGLONG);
	}

	unsigned           intmax_size  = 0;
	atomic_type_kind_t intmax_kind  = ATOMIC_TYPE_FIRST;
	unsigned           uintmax_size = 0;
	atomic_type_kind_t uintmax_kind = ATOMIC_TYPE_FIRST;
	unsigned           biggest_alignment = target.biggest_alignment;
	for (atomic_type_kind_t i = ATOMIC_TYPE_FIRST; i <= ATOMIC_TYPE_LAST; ++i) {
		assert(get_atomic_type_alignment(i) <= biggest_alignment);
		unsigned flags = get_atomic_type_flags(i);
		if (!(flags & ATOMIC_TYPE_FLAG_INTEGER))
			continue;
		unsigned size = get_atomic_type_size(i);
		if (flags & ATOMIC_TYPE_FLAG_SIGNED) {
			if (size > intmax_size) {
				intmax_kind = i;
				intmax_size = size;
			}
		} else {
			if (size > uintmax_size) {
				uintmax_kind = i;
				uintmax_size = size;
			}
		}
	}
	define_type_type_max("UINTMAX", uintmax_kind);
	define_type_c("__UINTMAX_C", uintmax_kind);
	define_type_type_max("INTMAX",  intmax_kind);
	define_type_c("__INTMAX_C", intmax_kind);
	add_define_int("__BIGGEST_ALIGNMENT__", biggest_alignment);

	/* TODO: less hardcoding for the following... */
	define_float_properties("FLT",  ATOMIC_TYPE_FLOAT);
	define_float_properties("DBL",  ATOMIC_TYPE_DOUBLE);
	define_float_properties("LDBL", ATOMIC_TYPE_LONG_DOUBLE);
	add_define("__FLT_RADIX__",   "2",                    false);
	add_define("__DECIMAL_DIG__", "__LDBL_DECIMAL_DIG__", false);

	/* TODO: __CHAR16_TYPE__, __CHAR32_TYPE__ */

	define_pragma_macro();

	/* Add target specific defines */
	for (ir_platform_define_t const *define = ir_platform_define_first();
		 define != NULL; define = ir_platform_define_next(define)) {
		char const *const name = ir_platform_define_name(define);
		if (name[0] != '_' && !dialect.gnu)
			continue;
		char const *const value = ir_platform_define_value(define);
		add_define(name, value, false);
	}
}
