/*
 * Copyright 2001 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*	Copyright (c) 1983, 1984, 1985, 1986, 1987, 1988, 1989 AT&T	*/
/*	  All Rights Reserved  	*/

/*
 * Copyright (c) 1980 Regents of the University of California.
 * All rights reserved.  The Berkeley software License Agreement
 * specifies the terms and conditions for redistribution.
 */

/*LINTLIBRARY*/

#ifndef lint
static char
sccsid[] = "@(#)standout.c 1.6 88/02/08 SMI"; /* from UCB 5.1 85/06/07 */
#endif /* not lint */

/*
 * routines dealing with entering and exiting standout mode
 *
 */

#include	"curses.ext"

/*
 * enter standout mode
 */

char *
wstandout(WINDOW *win)
{
	if (!SO && !UC)
		return (FALSE);

	win->_flags |= _STANDOUT;
	return (SO ? SO : UC);
}

/*
 * exit standout mode
 */

char *
wstandend(WINDOW *win)
{
	if (!SO && !UC)
		return (FALSE);

	win->_flags &= ~_STANDOUT;
	return (SE ? SE : UC);
}
