// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) International Business Machines  Corp., 2001
 */

/*\
 * [Description]
 *
 * Test whether the inode number are the same for both file descriptors.
 */

#include <unistd.h>
#include "tst_test.h"
#include "tst_safe_macros.h"

static int fd[2] = {-1, -1};
static int nfd[2] = {10, 20};

static void setup(void)
{
	SAFE_PIPE(fd);
}

static void cleanup(void)
{
	unsigned int i;

	for (i = 0; i < ARRAY_SIZE(fd); i++) {
		close(fd[i]);
		close(nfd[i]);
	}
}

static void run(unsigned int i)
{
	struct stat oldbuf, newbuf;

	TST_EXP_VAL(dup2(fd[i], nfd[i]), nfd[i]);
	if (TST_RET == -1)
		return;

	SAFE_FSTAT(fd[i], &oldbuf);
	SAFE_FSTAT(nfd[i], &newbuf);

	TST_EXP_EQ_LU(oldbuf.st_ino, newbuf.st_ino);

	SAFE_CLOSE(TST_RET);
}

static struct tst_test test = {
	.tcnt = ARRAY_SIZE(fd),
	.test = run,
	.setup = setup,
	.cleanup = cleanup,
};
