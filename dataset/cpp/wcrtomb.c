#include "../../threads/tss.h"
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "../local.h"

size_t _wcrtomb_r(
	char *s,
	wchar_t wc,
	mbstate_t *ps)
{
  int retval = 0;
  char buf[10];

#ifdef _MB_CAPABLE
  if (ps == NULL) {
      ps = &(tls_current()->mbst);
    }
#endif

  if (s == NULL)
    retval = __WCTOMB (buf, L'\0', ps);
  else
    retval = __WCTOMB (s, wc, ps);

  if (retval == -1)
    {
      ps->__count = 0;
      _set_errno(EILSEQ);
      return (size_t)(-1);
    }
  else
    return (size_t)retval;
}

size_t wcrtomb(
	char *__restrict s,
	wchar_t wc,
	mbstate_t *__restrict ps)
{
#if defined(PREFER_SIZE_OVER_SPEED) || defined(__OPTIMIZE_SIZE__)
  return _wcrtomb_r (s, wc, ps);
#else
  int retval = 0;
  char buf[10];

#ifdef _MB_CAPABLE
  if (ps == NULL)
    {
      ps = &(tls_current()->mbst);
    }
#endif

  if (s == NULL)
    retval = __WCTOMB (buf, L'\0', ps);
  else
    retval = __WCTOMB (s, wc, ps);

  if (retval == -1)
    {
      ps->__count = 0;
      _set_errno(EILSEQ);
      return (size_t)(-1);
    }
  else
    return (size_t)retval;
#endif /* not PREFER_SIZE_OVER_SPEED */
}
