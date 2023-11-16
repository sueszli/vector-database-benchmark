/*
 * Copyright 2022, Trung Nguyen, trungnt282910@gmail.com
 * All rights reserved. Distributed under the terms of the MIT License.
 */


#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <string.h>
#include <wctype.h>

#include <errno_private.h>

#include "LocaleBackend.h"


using BPrivate::Libroot::GetCurrentLocaleBackend;
using BPrivate::Libroot::LocaleBackend;
using BPrivate::Libroot::LocaleBackendData;


int
iswctype_l(wint_t wc, wctype_t charClass, locale_t l)
{
	LocaleBackendData* locale = (LocaleBackendData*)l;
	LocaleBackend* backend = locale->backend;

	if (backend == NULL) {
		if (wc < 0 || wc > 127)
			return 0;
		return __isctype(wc, charClass);
	}

	return backend->IsWCType(wc, charClass);
}


int
iswalnum_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISalnum, locale);
}


int
iswalpha_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISalpha, locale);
}


int
iswblank_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISblank, locale);
}


int
iswcntrl_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _IScntrl, locale);
}


int
iswdigit_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISdigit, locale);
}


int
iswgraph_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISgraph, locale);
}


int
iswlower_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISlower, locale);
}


int
iswprint_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISprint, locale);
}


int
iswpunct_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISpunct, locale);
}


int
iswspace_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISspace, locale);
}


int
iswupper_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISupper, locale);
}


int
iswxdigit_l(wint_t wc, locale_t locale)
{
	return iswctype_l(wc, _ISxdigit, locale);
}


wint_t
towlower_l(wint_t wc, locale_t l)
{
	LocaleBackendData* locale = (LocaleBackendData*)l;
	LocaleBackend* backend = locale->backend;

	if (backend == NULL) {
		if (wc < 0 || wc > 127)
			return wc;
		return tolower(wc);
	}

	wint_t result = wc;
	backend->ToWCTrans(wc, _ISlower, result);

	return result;
}


wint_t
towupper_l(wint_t wc, locale_t l)
{
	LocaleBackendData* locale = (LocaleBackendData*)l;
	LocaleBackend* backend = locale->backend;

	if (backend == NULL) {
		if (wc < 0 || wc > 127)
			return wc;
		return toupper(wc);
	}

	wint_t result = wc;
	backend->ToWCTrans(wc, _ISupper, result);

	return result;
}


wint_t
towctrans_l(wint_t wc, wctrans_t transition, locale_t l)
{
	LocaleBackendData* locale = (LocaleBackendData*)l;
	LocaleBackend* backend = locale->backend;

	if (backend == NULL) {
		if (transition == _ISlower)
			return tolower(wc);
		if (transition == _ISupper)
			return toupper(wc);

		__set_errno(EINVAL);
		return wc;
	}

	wint_t result = wc;
	status_t status = backend->ToWCTrans(wc, transition, result);
	if (status != B_OK)
		__set_errno(EINVAL);

	return result;
}


wctrans_t
wctrans_l(const char *charClass, locale_t locale)
{
	(void)locale;

	if (charClass != NULL) {
		// we do not know any locale-specific character classes
		if (strcmp(charClass, "tolower") == 0)
			return _ISlower;
		if (strcmp(charClass, "toupper") == 0)
			return _ISupper;
	}

	__set_errno(EINVAL);
	return 0;
}


wctype_t
wctype_l(const char *property, locale_t locale)
{
	(void)locale;

	// currently, we do not support any locale-specific properties
	if (strcmp(property, "alnum") == 0)
		return _ISalnum;
	if (strcmp(property, "alpha") == 0)
		return _ISalpha;
	if (strcmp(property, "blank") == 0)
		return _ISblank;
	if (strcmp(property, "cntrl") == 0)
		return _IScntrl;
	if (strcmp(property, "digit") == 0)
		return _ISdigit;
	if (strcmp(property, "graph") == 0)
		return _ISgraph;
	if (strcmp(property, "lower") == 0)
		return _ISlower;
	if (strcmp(property, "print") == 0)
		return _ISprint;
	if (strcmp(property, "punct") == 0)
		return _ISpunct;
	if (strcmp(property, "space") == 0)
		return _ISspace;
	if (strcmp(property, "upper") == 0)
		return _ISupper;
	if (strcmp(property, "xdigit") == 0)
		return _ISxdigit;

	return 0;
}
