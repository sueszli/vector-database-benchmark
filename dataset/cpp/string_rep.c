/*
 * This file is part of cparser.
 * Copyright (C) 2012 Matthias Braun <matze@braunis.de>
 */
#include "string_rep.h"

#include "adt/panic.h"
#include "adt/unicode.h"

static inline size_t wstrlen(const string_t *string)
{
	size_t      result = 0;
	const char *p      = string->begin;
	const char *end    = p + string->size;
	while (p < end) {
		read_utf8_char(&p);
		++result;
	}
	return result;
}

size_t get_string_len(string_t const *const str)
{
	switch (str->encoding) {
	case STRING_ENCODING_CHAR:
	case STRING_ENCODING_UTF8:   return str->size;
	case STRING_ENCODING_CHAR16:
	case STRING_ENCODING_CHAR32:
	case STRING_ENCODING_WIDE:   return wstrlen(str);
	}
	panic("invalid string encoding");
}
