/*
 * Copyright (C) 2002 by Darren Reed.
 *
 * See the IPFILTER.LICENCE file for details on licencing.
 *
 * Copyright 2007 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include "ipf.h"

#define	PRINTF	(void)printf
#define	FPRINTF	(void)fprintf


void printhashdata(hp, opts)
iphtable_t *hp;
int opts;
{

	if ((opts & OPT_DEBUG) == 0) {
		if ((hp->iph_type & IPHASH_ANON) == IPHASH_ANON)
			PRINTF("# 'anonymous' table\n");
		switch (hp->iph_type & ~IPHASH_ANON)
		{
		case IPHASH_LOOKUP :
			PRINTF("table");
			break;
		case IPHASH_GROUPMAP :
			PRINTF("group-map");
			if (hp->iph_flags & FR_INQUE)
				PRINTF(" in");
			else if (hp->iph_flags & FR_OUTQUE)
				PRINTF(" out");
			else
				PRINTF(" ???");
			break;
		default :
			PRINTF("%#x", hp->iph_type);
			break;
		}
		PRINTF(" role = ");
	} else {
		PRINTF("Hash Table Number: %s", hp->iph_name);
		if ((hp->iph_type & IPHASH_ANON) == IPHASH_ANON)
			PRINTF("(anon)");
		putchar(' ');
		PRINTF("Role: ");
	}

	switch (hp->iph_unit)
	{
	case IPL_LOGNAT :
		PRINTF("nat");
		break;
	case IPL_LOGIPF :
		PRINTF("ipf");
		break;
	case IPL_LOGAUTH :
		PRINTF("auth");
		break;
	case IPL_LOGCOUNT :
		PRINTF("count");
		break;
	default :
		PRINTF("#%d", hp->iph_unit);
		break;
	}

	if ((opts & OPT_DEBUG) == 0) {
		if ((hp->iph_type & ~IPHASH_ANON) == IPHASH_LOOKUP)
			PRINTF(" type = hash");
		PRINTF(" number = %s size = %lu",
			hp->iph_name, (u_long)hp->iph_size);
		if (hp->iph_seed != 0)
			PRINTF(" seed = %lu", hp->iph_seed);
		putchar('\n');
	} else {
		PRINTF(" Type: ");
		switch (hp->iph_type & ~IPHASH_ANON)
		{
		case IPHASH_LOOKUP :
			PRINTF("lookup");
			break;
		case IPHASH_GROUPMAP :
			PRINTF("groupmap Group. %s", hp->iph_name);
			break;
		default :
			break;
		}

		putchar('\n');
		PRINTF("\t\tSize: %lu\tSeed: %lu",
			(u_long)hp->iph_size, hp->iph_seed);
		PRINTF("\tRef. Count: %d\tMasks: %#x\n", hp->iph_ref,
			hp->iph_masks[0]);
	}

	if ((opts & OPT_DEBUG) != 0) {
		struct in_addr m;
		int i;

		for (i = 0; i < 32; i++) {
			if ((1 << i) & hp->iph_masks[0]) {
				ntomask(4, i, &m.s_addr);
				PRINTF("\t\tMask: %s\n", inet_ntoa(m));
			}
		}
	}
}
