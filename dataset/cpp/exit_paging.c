/*
 * This file and its contents are supplied under the terms of the
 * Common Development and Distribution License ("CDDL"), version 1.0.
 * You may only use this file in accordance with the terms of version
 * 1.0 of the CDDL.
 *
 * A full copy of the text of the CDDL should have accompanied this
 * source.  A copy of the CDDL is also available via the Internet at
 * http://www.illumos.org/license/CDDL.
 */

/*
 * Copyright 2022 Oxide Computer Company
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <strings.h>
#include <libgen.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/sysmacros.h>
#include <sys/debug.h>
#include <sys/mman.h>
#include <sys/vmm.h>
#include <sys/vmm_dev.h>
#include <vmmapi.h>

#include "in_guest.h"

int
main(int argc, char *argv[])
{
	const char *test_suite_name = basename(argv[0]);
	struct vmctx *ctx = NULL;
	int err;

	ctx = test_initialize(test_suite_name);

	err = test_setup_vcpu(ctx, 0, MEM_LOC_PAYLOAD, MEM_LOC_STACK);
	if (err != 0) {
		test_fail_errno(err, "Could not initialize vcpu0");
	}

	struct vm_entry ventry = { 0 };
	struct vm_exit vexit = { 0 };

	const uintptr_t expected_gpa = MEM_LOC_ROM;
	const int expected_ftype = PROT_WRITE;

	do {
		const enum vm_exit_kind kind =
		    test_run_vcpu(ctx, 0, &ventry, &vexit);
		switch (kind) {
		case VEK_REENTR:
			break;
		case VEK_UNHANDLED:
			if (vexit.exitcode != VM_EXITCODE_PAGING) {
				test_fail_vmexit(&vexit);
			}
			if (vexit.u.paging.gpa != expected_gpa) {
				test_fail_msg("gpa %08x != %08x\n",
				    vexit.u.paging.gpa, expected_gpa);
			}
			if (vexit.u.paging.fault_type != expected_ftype) {
				test_fail_msg("fault_type %x != %x\n",
				    vexit.u.paging.fault_type, expected_ftype);
			}
			test_pass();
			break;

		default:
			test_fail_vmexit(&vexit);
			break;
		}
	} while (true);
}
