/*
 * Copyright 2012 Haiku, Inc. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Authors:
 *		Alexander von Gluck, kallisti5@unixzen.com
 */

/*
 * Pass a standard CPUID in hex, and get out a CPUID for cpu_type.h
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define EXT_FAMILY_MASK 0xF00000
#define EXT_MODEL_MASK	0x0F0000
#define FAMILY_MASK		0x000F00
#define MODEL_MASK		0x0000F0
#define STEPPING_MASK	0x00000F


// Converts a hexadecimal string to integer
static int
xtoi(const char* xs, unsigned int* result)
{
	size_t szlen = strlen(xs);
	int i;
	int xv;
	int fact;

	if (szlen > 0) {
		// Converting more than 32bit hexadecimal value?
		if (szlen > 8)
			return 2;

		// Begin conversion here
		*result = 0;
		fact = 1;

		// Run until no more character to convert
		for (i = szlen - 1; i>=0; i--) {
			if (isxdigit(*(xs + i))) {
				if (*(xs + i) >= 97)
					xv = (*(xs + i) - 97) + 10;
				else if (*(xs + i) >= 65)
					xv = (*(xs + i) - 65) + 10;
				else
					xv = *(xs + i) - 48;

				*result += (xv * fact);
				fact *= 16;
			} else {
				// Conversion was abnormally terminated
				// by non hexadecimal digit, hence
				// returning only the converted with
				// an error value 4 (illegal hex character)
				return 4;
			}
		}
	}

	// Nothing to convert
	return 1;
}


int
main(int argc, char *argv[])
{
	if (argc != 2) {
		printf("Provide the cpuid in hex, and you will get how we id it\n");
		printf("usage: cpuidhaiku <cpuid_hex>\n");
		return 1;
	}

	unsigned int cpuid = 0;
	xtoi(argv[1], &cpuid);

	printf("cpuid: 0x%X\n", cpuid);

    int family = ((cpuid >> 8) & 0xf) | ((cpuid >> 16) & 0xff0);
    int model = ((cpuid >> 4) & 0xf) | ((cpuid >> 12) & 0xf0);
    int stepping = cpuid & 0xf;

	printf("Haiku CPUID: Family: 0x%x, Model: 0x%x, Stepping: 0x%x\n", family,
		model, stepping);

	return 0;
}
