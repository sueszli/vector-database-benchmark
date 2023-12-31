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
 * Copyright 2005 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <sys/systm.h>
#include <sys/kobj.h>

#include <kmdb/kctl/kctl.h>

char *
kctl_basename(char *s)
{
	char *p = strrchr(s, '/');

	if (p == NULL)
		return (s);

	return (++p);
}

char *
kctl_strdup(const char *s)
{
	char *s1 = kobj_alloc(strlen(s) + 1, KM_SLEEP);

	(void) strcpy(s1, s);
	return (s1);
}

void
kctl_strfree(char *s)
{
	kobj_free(s, strlen(s) + 1);
}
