/*
FUNCTION
	<<islower>>, <<islower_l>>---lowercase character predicate

INDEX
	islower

INDEX
	islower_l

ANSI_SYNOPSIS
	#include <ctype.h>
	int islower(int <[c]>);

	#include <ctype.h>
	int islower_l(int <[c]>, locale_t <[locale]>);

TRAD_SYNOPSIS
	#include <ctype.h>
	int islower(<[c]>);

DESCRIPTION
<<islower>> is a macro which classifies singlebyte charset values by table
lookup.  It is a predicate returning non-zero for minuscules
(lowercase alphabetic characters), and 0 for other characters.
It is defined only if <[c]> is representable as an unsigned char or if
<[c]> is EOF.

<<islower_l>> is like <<islower>> but performs the check based on the
locale specified by the locale object locale.  If <[locale]> is
LC_GLOBAL_LOCALE or not a valid locale object, the behaviour is undefined.

You can use a compiled subroutine instead of the macro definition by
undefining the macro using `<<#undef islower>>' or `<<#undef islower_l>>'.

RETURNS
<<islower>>, <<islower_l>> return non-zero if <[c]> is a lowercase letter.

PORTABILITY
<<islower>> is ANSI C.
<<islower_l>> is POSIX-1.2008.

No supporting OS subroutines are required.
*/

/* Includes */
#include <ctype.h>

/* Undefine symbol in case it's been
 * inlined as a macro */
#undef islower

/* Checks for a lower-case character. */
int islower(int c)
{
	return ((__CTYPE_PTR[c+1] & (_CTYPE_U|_CTYPE_L)) == _CTYPE_L);
}
