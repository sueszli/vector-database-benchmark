/*
 * CDDL HEADER START
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License, Version 1.0 only
 * (the "License").  You may not use this file except in compliance
 * with the License.
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
 * Copyright (c) 1995-1998 by Sun Microsystems, Inc.
 * All rights reserved.
 */

/* LINTLIBRARY */

/*
 * addnws.c
 *
 * XCurses Library
 *
 * Copyright 1990, 1995 by Mortice Kern Systems Inc.  All rights reserved.
 *
 */

#if M_RCSID
#ifndef lint
static char rcsID[] = "$Header: /rd/src/libc/xcurses/rcs/addnws.c 1.2 "
"1995/05/18 20:55:00 ant Exp $";
#endif
#endif

#include <private.h>

#undef addnwstr

int
addnwstr(const wchar_t *wcs, int n)
{
	int code;

	code = waddnwstr(stdscr, wcs, n);

	return (code);
}

#undef mvaddnwstr

int
mvaddnwstr(int y, int x, const wchar_t *wcs, int n)
{
	int code;

	if ((code = wmove(stdscr, y, x)) == OK)
		code = waddnwstr(stdscr, wcs, n);

	return (code);
}

#undef mvwaddnwstr

int
mvwaddnwstr(WINDOW *w, int y, int x, const wchar_t *wcs, int n)
{
	int code;

	if ((code = wmove(w, y, x)) == OK)
		code = waddnwstr(w, wcs, n);

	return (code);
}

#undef addwstr

int
addwstr(const wchar_t *wcs)
{
	int code;

	code = waddnwstr(stdscr, wcs, -1);

	return (code);
}

#undef mvaddwstr

int
mvaddwstr(int y, int x, const wchar_t *wcs)
{
	int code;

	if ((code = wmove(stdscr, y, x)) == OK)
		code = waddnwstr(stdscr, wcs, -1);

	return (code);
}

#undef mvwaddwstr

int
mvwaddwstr(WINDOW *w, int y, int x, const wchar_t *wcs)
{
	int code;

	if ((code = wmove(w, y, x)) == OK)
		code = waddnwstr(w, wcs, -1);

	return (code);
}

#undef waddwstr

int
waddwstr(WINDOW *w, const wchar_t *wcs)
{
	int code;

	code = waddnwstr(w, wcs, -1);

	return (code);
}
