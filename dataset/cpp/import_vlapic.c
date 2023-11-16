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
#include <fcntl.h>
#include <libgen.h>
#include <sys/stat.h>
#include <errno.h>
#include <err.h>
#include <assert.h>
#include <sys/sysmacros.h>
#include <stdbool.h>

#include <sys/vmm.h>
#include <sys/vmm_dev.h>
#include <sys/vmm_data.h>
#include <vmmapi.h>

#include "common.h"

#define	APIC_ADDR_CCR	0xfee00390

#define	TIMER_TEST_VAL	0x10000

int
main(int argc, char *argv[])
{
	const char *suite_name = basename(argv[0]);
	struct vmctx *ctx;

	ctx = create_test_vm(suite_name);
	if (ctx == NULL) {
		errx(EXIT_FAILURE, "could not open test VM");
	}

	if (vm_activate_cpu(ctx, 0) != 0) {
		err(EXIT_FAILURE, "could not activate vcpu0");
	}

	const int vmfd = vm_get_device_fd(ctx);
	int error;

	/* Pause the instance before attempting to manipulate vlapic data */
	if (ioctl(vmfd, VM_PAUSE, 0) != 0) {
		err(EXIT_FAILURE, "VM_PAUSE failed");
	}

	struct vdi_lapic_v1 lapic_data;
	struct vm_data_xfer xfer = {
		.vdx_vcpuid = 0,
		.vdx_class = VDC_LAPIC,
		.vdx_version = 1,
		.vdx_len = sizeof (lapic_data),
		.vdx_data = &lapic_data,
	};

	/* Read the existing lapic data to get a baseline */
	if (ioctl(vmfd, VM_DATA_READ, &xfer) != 0) {
		err(EXIT_FAILURE, "VM_DATA_READ of lapic failed");
	}

	/* Writing that exact same data back should be fine */
	if (ioctl(vmfd, VM_DATA_WRITE, &xfer) != 0) {
		err(EXIT_FAILURE, "VM_DATA_WRITE of lapic failed");
	}

	/* Simulate ICR being loaded with a meaningful (but short) value */
	lapic_data.vl_lapic.vlp_icr_timer = TIMER_TEST_VAL;
	/*
	 * Pretend as if timer is scheduled to fire 100s (in the future) after
	 * VM boot time.  With the ICR value, this should trigger the overage
	 * detection and clamping.
	 */
	lapic_data.vl_timer_target = 1000000000UL * 100;

	/* Try to write the outlandish timer result */
	if (ioctl(vmfd, VM_DATA_WRITE, &xfer) != 0) {
		err(EXIT_FAILURE, "VM_DATA_WRITE of lapic failed");
	}

	/*
	 * The timer will not actually be scheduled (and thus observable via
	 * CCR) until the instance is resumed...
	 */
	if (ioctl(vmfd, VM_RESUME, 0) != 0) {
		err(EXIT_FAILURE, "VM_RESUME failed");
	}

	/* Now simulate a read of CCR from that LAPIC */
	uint64_t ccr_value = 0;
	error = vm_readwrite_kernemu_device(ctx, 0, APIC_ADDR_CCR,
	    false, 4, &ccr_value);
	if (error != 0) {
		err(EXIT_FAILURE, "could not emulate MMIO of LAPIC CCR");
	}
	if (ccr_value != TIMER_TEST_VAL) {
		errx(EXIT_FAILURE, "CCR not clamped: %lx != %lx",
		    ccr_value, TIMER_TEST_VAL);
	}

	vm_destroy(ctx);
	(void) printf("%s\tPASS\n", suite_name);
	return (EXIT_SUCCESS);
}
