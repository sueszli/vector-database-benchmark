/*
** Copyright 2011, Oliver Tappe, zooey@hirschkaefer.de. All rights reserved.
** Distributed under the terms of the MIT License.
*/

#include <errno.h>
#include <string.h>
#include <wchar.h>

#include <errno_private.h>
#include <LocaleBackend.h>
#include <wchar_private.h>


//#define TRACE_WCSRTOMBS
#ifdef TRACE_WCSRTOMBS
#	include <OS.h>
#	define TRACE(x) debug_printf x
#else
#	define TRACE(x) ;
#endif


using BPrivate::Libroot::GetCurrentLocaleBackend;
using BPrivate::Libroot::LocaleBackend;


extern "C" size_t
__wcsnrtombs(char* dst, const wchar_t** src, size_t nwc, size_t len,
	mbstate_t* ps)
{
	LocaleBackend* backend = GetCurrentLocaleBackend();

	TRACE(("wcsnrtombs(%p, %p, %lu, %lu) - lb:%p\n", dst, *src, nwc, len,
		backend));

	if (ps == NULL) {
		static mbstate_t internalMbState;
		ps = &internalMbState;
	}

	if (backend == NULL) {
		/*
		 * The POSIX locale is active. Since the POSIX locale only contains
		 * chars 0-127 and those ASCII chars are compatible with the UTF32
		 * values used in wint_t, we can just copy the codepoint values.
		 */
		size_t count = 0;
		const wchar_t* srcEnd = *src + nwc;
		if (dst == NULL) {
			// only count number of required wide characters
			const wchar_t* wcSrc = *src;
			for (; wcSrc < srcEnd; ++wcSrc, ++count) {
				if (*wcSrc < 0) {
					// char is non-ASCII
					__set_errno(EILSEQ);
					count = (size_t)-1;
					break;
				}
				if (*wcSrc == 0) {
					memset(ps, 0, sizeof(mbstate_t));
					break;
				}
			}
		} else {
			// "convert" the characters
			for (; *src < srcEnd && count < len; ++*src, ++count) {
				if (**src < 0) {
					// char is non-ASCII
					__set_errno(EILSEQ);
					count = (size_t)-1;
					break;
				}
				*dst++ = (char)**src;
				if (**src == 0) {
					memset(ps, 0, sizeof(mbstate_t));
					*src = NULL;
					break;
				}
			}
		}

		TRACE(("wcsnrtombs returns %lx and src %p\n", count, *src));

		return count;
	}

	size_t result = 0;
	status_t status = backend->WcharStringToMultibyte(dst, len, src, nwc,
		ps, result);

	if (status == B_BAD_DATA) {
		TRACE(("wcsnrtomb(): setting errno to EILSEQ\n"));
		__set_errno(EILSEQ);
		result = (size_t)-1;
	} else if (status != B_OK) {
		TRACE(("wcsnrtomb(): setting errno to EINVAL (status: %lx)\n", status));
		__set_errno(EINVAL);
		result = (size_t)-1;
	}

	TRACE(("wcsnrtombs returns %lx and src %p\n", result, *src));

	return result;
}


B_DEFINE_WEAK_ALIAS(__wcsnrtombs, wcsnrtombs);


extern "C" size_t
__wcsrtombs(char* dst, const wchar_t** src, size_t len, mbstate_t* ps)
{
	if (ps == NULL) {
		static mbstate_t internalMbState;
		ps = &internalMbState;
	}

	return __wcsnrtombs(dst, src, __wcslen(*src) + 1, len, ps);
}


B_DEFINE_WEAK_ALIAS(__wcsrtombs, wcsrtombs);
