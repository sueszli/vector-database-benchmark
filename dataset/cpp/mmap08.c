// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) International Business Machines  Corp., 2001
 *  07/2001 Ported by Wayne Boyer
 * Copyright (c) 2023 SUSE LLC Avinesh Kumar <avinesh.kumar@suse.com>
 */

/*\
 * [Description]
 *
 * verify that, mmap() calls fails with errno EBADF when a file mapping
 * is requested but the fd is not a valid file descriptor.
 */

#include <stdlib.h>
#include "tst_test.h"

#define TEMPFILE "mmapfile"
static size_t page_sz;
static int fd;

static void setup(void)
{
	fd = SAFE_OPEN(TEMPFILE, O_RDWR | O_CREAT, 0666);
	SAFE_CLOSE(fd);
}

static void run(void)
{
	TESTPTR(mmap(NULL, page_sz, PROT_WRITE, MAP_FILE | MAP_SHARED, fd, 0));

	if (TST_RET_PTR != MAP_FAILED) {
		tst_res(TFAIL, "mmap() passed unexpectedly");
		SAFE_MUNMAP(TST_RET_PTR, page_sz);
	} else if (TST_ERR == EBADF) {
		tst_res(TPASS, "mmap() failed with EBADF");
	} else {
		tst_res(TFAIL | TERRNO, "mmap() failed with an invalid errno");
	}
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
	.needs_tmpdir = 1
};
