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
 * Copyright 2023 Oxide Computer Company
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
#include "test_defs.h"


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

	timespec_t ts = { 0 };
	err = vm_rtc_settime(ctx, &ts);
	if (err != 0) {
		test_fail_errno(err, "Could zero out RTC time");
	}

	struct vm_entry ventry = { 0 };
	struct vm_exit vexit = { 0 };

	do {
		const enum vm_exit_kind kind =
		    test_run_vcpu(ctx, 0, &ventry, &vexit);
		switch (kind) {
		case VEK_REENTR:
			break;
		case VEK_TEST_PASS:
			test_pass();
			break;
		case VEK_TEST_FAIL:
			test_fail();
			break;
		case VEK_TEST_MSG:
			test_msg_print(ctx);
			break;
		default:
			test_fail_vmexit(&vexit);
			break;
		}
	} while (true);
}
