/*
 * Copyright 2010-2011, Oliver Tappe, zooey@hirschkaefer.de.
 * Distributed under the terms of the MIT License.
 */


#include "ICUCollateData.h"

#include <assert.h>
#include <string.h>
#include <strings.h>
#include <wchar.h>

#include <unicode/unistr.h>

#include <AutoDeleter.h>


U_NAMESPACE_USE


namespace BPrivate {
namespace Libroot {


ICUCollateData::ICUCollateData(pthread_key_t tlsKey)
	:
	inherited(tlsKey),
	fCollator(NULL)
{
}


ICUCollateData::~ICUCollateData()
{
	delete fCollator;
}


status_t
ICUCollateData::SetTo(const Locale& locale, const char* posixLocaleName)
{
	status_t result = inherited::SetTo(locale, posixLocaleName);

	if (result == B_OK) {
		UErrorCode icuStatus = U_ZERO_ERROR;
		delete fCollator;
		fCollator = Collator::createInstance(fLocale, icuStatus);
		if (!U_SUCCESS(icuStatus))
			return B_NO_MEMORY;
	}

	return result;
}


status_t
ICUCollateData::SetToPosix()
{
	status_t result = inherited::SetToPosix();

	if (result == B_OK) {
		delete fCollator;
		fCollator = NULL;
	}

	return result;
}


status_t
ICUCollateData::Strcoll(const char* a, const char* b, int& result)
{
	if (fCollator == NULL || strcmp(fPosixLocaleName, "POSIX") == 0) {
		// handle POSIX here as the collator ICU uses for that (english) is
		// incompatible in too many ways
		result = strcmp(a, b);
		for (const char* aIter = a; *aIter != 0; ++aIter) {
			if (*aIter < 0)
				return B_BAD_VALUE;
		}
		for (const char* bIter = b; *bIter != 0; ++bIter) {
			if (*bIter < 0)
				return B_BAD_VALUE;
		}
		return B_OK;
	}

	status_t status = B_OK;
	UErrorCode icuStatus = U_ZERO_ERROR;

	if (strcasecmp(fGivenCharset, "utf-8") == 0) {
		UCharIterator aIter, bIter;
		uiter_setUTF8(&aIter, a, -1);
		uiter_setUTF8(&bIter, b, -1);

		result = fCollator->compare(aIter, bIter, icuStatus);
	} else {
		UnicodeString unicodeA;
		UnicodeString unicodeB;

		if (_ToUnicodeString(a, unicodeA) != B_OK
			|| _ToUnicodeString(b, unicodeB) != B_OK) {
			status = B_BAD_VALUE;
		}

		result = fCollator->compare(unicodeA, unicodeB, icuStatus);
	}

	if (!U_SUCCESS(icuStatus))
		status = B_BAD_VALUE;

	return status;
}


status_t
ICUCollateData::Strxfrm(char* out, const char* in,
	size_t outSize, size_t& requiredSize)
{
	if (in == NULL) {
		requiredSize = 0;
		return B_OK;
	}

	if (fCollator == NULL || strcmp(fPosixLocaleName, "POSIX") == 0) {
		// handle POSIX here as the collator ICU uses for that (english) is
		// incompatible in too many ways
		requiredSize = strlcpy(out, in, outSize);
		for (const char* inIter = in; *inIter != 0; ++inIter) {
			if (*inIter < 0)
				return B_BAD_VALUE;
		}
		return B_OK;
	}

	UnicodeString unicodeIn;
	if (_ToUnicodeString(in, unicodeIn) != B_OK)
		return B_BAD_VALUE;

	requiredSize = fCollator->getSortKey(unicodeIn, (uint8_t*)out, outSize);

	// Do not include terminating NULL byte in the required-size.
	if (requiredSize > 0) {
		if (outSize >= requiredSize)
			assert(out[requiredSize - 1] == '\0');
		requiredSize--;
	}

	return B_OK;
}


status_t
ICUCollateData::Wcscoll(const wchar_t* a, const wchar_t* b, int& result)
{
	if (fCollator == NULL || strcmp(fPosixLocaleName, "POSIX") == 0) {
		// handle POSIX here as the collator ICU uses for that (english) is
		// incompatible in too many ways
		result = wcscmp(a, b);
		for (const wchar_t* aIter = a; *aIter != 0; ++aIter) {
			if (*aIter > 127)
				return B_BAD_VALUE;
		}
		for (const wchar_t* bIter = b; *bIter != 0; ++bIter) {
			if (*bIter > 127)
				return B_BAD_VALUE;
		}
		return B_OK;
	}

	UnicodeString unicodeA = UnicodeString::fromUTF32((UChar32*)a, -1);
	UnicodeString unicodeB = UnicodeString::fromUTF32((UChar32*)b, -1);

	UErrorCode icuStatus = U_ZERO_ERROR;
	result = fCollator->compare(unicodeA, unicodeB, icuStatus);

	if (!U_SUCCESS(icuStatus))
		return B_BAD_VALUE;

	return B_OK;
}


status_t
ICUCollateData::Wcsxfrm(wchar_t* out, const wchar_t* in, size_t outSize,
	size_t& requiredSize)
{
	if (in == NULL) {
		requiredSize = 0;
		return B_OK;
	}

	if (fCollator == NULL || strcmp(fPosixLocaleName, "POSIX") == 0) {
		// handle POSIX here as the collator ICU uses for that (english) is
		// incompatible in too many ways
		requiredSize = wcslcpy(out, in, outSize);
		for (const wchar_t* inIter = in; *inIter != 0; ++inIter) {
			if (*inIter > 127)
				return B_BAD_VALUE;
		}
		return B_OK;
	}

	UnicodeString unicodeIn = UnicodeString::fromUTF32((UChar32*)in, -1);
	requiredSize = fCollator->getSortKey(unicodeIn, NULL, 0);

	if (outSize == 0)
		return B_OK;

	uint8_t* buffer = (uint8_t*)out;
	fCollator->getSortKey(unicodeIn, buffer, outSize);

	// convert 1-byte characters to 4-byte wide characters:
	for (size_t i = 0; i < outSize; ++i)
		out[outSize - 1 - i] = buffer[outSize - 1 - i];

	// Do not include terminating NULL character in the required-size.
	if (requiredSize > 0) {
		if (outSize >= requiredSize)
			assert(out[requiredSize - 1] == 0);
		requiredSize--;
	}

	return B_OK;
}


status_t
ICUCollateData::_ToUnicodeString(const char* in, UnicodeString& out)
{
	out.remove();

	if (in == NULL)
		return B_OK;

	size_t inLen = strlen(in);
	if (inLen == 0)
		return B_OK;

	UConverter* converter;
	status_t result = _GetConverter(converter);
	if (result != B_OK)
		return result;

	UErrorCode icuStatus = U_ZERO_ERROR;
	int32_t outLen = ucnv_toUChars(converter, NULL, 0, in, inLen, &icuStatus);
	if (icuStatus != U_BUFFER_OVERFLOW_ERROR)
		return B_BAD_VALUE;
	if (outLen < 0)
		return B_ERROR;
	if (outLen == 0)
		return B_OK;

	UChar* outBuf = out.getBuffer(outLen + 1);
	icuStatus = U_ZERO_ERROR;
	outLen
		= ucnv_toUChars(converter, outBuf, outLen + 1, in, inLen, &icuStatus);
	if (!U_SUCCESS(icuStatus)) {
		out.releaseBuffer(0);
		return B_BAD_VALUE;
	}

	out.releaseBuffer(outLen);

	return B_OK;
}


}	// namespace Libroot
}	// namespace BPrivate
