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
	const uint32_t expected_code = 0x01020304;
	const uint64_t expected_wval = 0x05060708090a0b0c;

	do {
		const enum vm_exit_kind kind =
		    test_run_vcpu(ctx, 0, &ventry, &vexit);
		switch (kind) {
		case VEK_REENTR:
			break;
		case VEK_UNHANDLED:
			/* Look for wrmsr of test value */
			if (vexit.exitcode != VM_EXITCODE_WRMSR) {
				test_fail_vmexit(&vexit);
			}
			if (vexit.u.msr.code != expected_code) {
				test_fail_msg("code %08x != %08x\n",
				    vexit.u.msr.code, expected_code);
			}
			if (vexit.u.msr.wval != expected_wval) {
				test_fail_msg("wval %lx != %lx\n",
				    vexit.u.msr.wval, expected_wval);
			}
			test_pass();
			break;

		case VEK_TEST_PASS:
		case VEK_TEST_FAIL:
		default:
			test_fail_vmexit(&vexit);
			break;
		}
	} while (true);
}
