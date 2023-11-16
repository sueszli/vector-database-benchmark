// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) International Business Machines  Corp., 2004
 *  Written by Robbie Williamson
 * Copyright (c) 2023 SUSE LLC Avinesh Kumar <avinesh.kumar@suse.com>
 */

/*\
 * [Description]
 *
 * Verify that, a normal page cannot be mapped into a high memory region,
 * and mmap() call fails with either ENOMEM or EINVAL errno.
 */

#include "tst_test.h"

#ifdef __ia64__
# define HIGH_ADDR ((void *)(0xa000000000000000UL))
#else
# define HIGH_ADDR ((void *)(-page_size))
#endif

#define TEMPFILE "mmapfile"
static long page_size;
static int fd;

static void run(void)
{
#ifdef TST_ABI32
	tst_brk(TCONF, "Test is not applicable for 32-bit systems.");
#endif

	fd = SAFE_OPEN(TEMPFILE, O_RDWR | O_CREAT, 0666);

	TESTPTR(mmap(HIGH_ADDR, page_size, PROT_READ, MAP_SHARED | MAP_FIXED, fd, 0));

	if (TST_RET_PTR != MAP_FAILED) {
		tst_res(TFAIL, "mmap() into high mem region succeeded unexpectedly");
		SAFE_MUNMAP(TST_RET_PTR, page_size);
		return;
	} else if (TST_RET_PTR == MAP_FAILED && (TST_ERR == ENOMEM || TST_ERR == EINVAL)) {
		tst_res(TPASS | TERRNO, "mmap() failed with expected errno");
	} else {
		tst_res(TFAIL | TERRNO, "mmap() failed with unexpected errno");
	}

	SAFE_CLOSE(fd);
}

static void setup(void)
{
	page_size = getpagesize();
}

static void cleanup(void)
{
	if (fd > 0)
		SAFE_CLOSE(fd);
}

static struct tst_test test = {
	.setup = setup,
	.cleanup = cleanup,
	.test_all = run,
	.needs_root = 1,
	.needs_tmpdir = 1
};
