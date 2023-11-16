/*
 * COPYRIGHT:   See COPYING in the top level directory
 * PROJECT:     ReactOS system libraries
 * FILE:        lib/sdk/crt/wstring/wcsupr.c
 * PURPOSE:     Unknown
 * PROGRAMER:   Unknown
 * UPDATE HISTORY:
 *              25/11/05: Added license header
 */

#include <precomp.h>

/*
 * @implemented
 */
wchar_t * CDECL _wcsupr(wchar_t *x)
{
	wchar_t  *y = x;

	while (*y) {
		*y = towupper(*y);
		y++;
	}
	return x;
}
