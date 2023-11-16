// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) International Business Machines  Corp., 2001
 * 07/2001 Ported by Wayne Boyer
 * Copyright (c) 2021 Xie Ziyao <xieziyao@huawei.com>
 */

/*\
 * [Description]
 *
 * Verify that, chown(2) succeeds to change the owner and group of a file
 * specified by path to any numeric owner(uid)/group(gid) values when invoked
 * by super-user.
 */

#include "tst_test.h"
#include "compat_tst_16.h"
#include "tst_safe_macros.h"

#define FILE_MODE (S_IFREG|S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)
#define TESTFILE "testfile"

struct test_case_t {
	char *desc;
	uid_t uid;
	gid_t gid;
} tc[] = {
	{"change owner/group ids", 700, 701},
	{"change owner id only", 702, -1},
	{"change owner id only", 703, 701},
	{"change group id only", -1, 704},
	{"change group id only", 703, 705},
	{"no change", -1, -1}
};

static void run(unsigned int i)
{
	struct stat stat_buf;
	uid_t expect_uid = tc[i].uid == (uid_t)-1 ? tc[i - 1].uid : tc[i].uid;
	gid_t expect_gid = tc[i].gid == (uid_t)-1 ? tc[i - 1].gid : tc[i].gid;

	TST_EXP_PASS(CHOWN(TESTFILE, tc[i].uid, tc[i].gid), "chown(%s, %d, %d), %s",
		     TESTFILE, tc[i].uid, tc[i].gid, tc[i].desc);

	SAFE_STAT(TESTFILE, &stat_buf);
	if (stat_buf.st_uid != expect_uid || stat_buf.st_gid != expect_gid) {
		tst_res(TFAIL, "%s: incorrect ownership set, expected %d %d",
			TESTFILE, expect_uid, expect_gid);
	}
}

static void setup(void)
{
	SAFE_TOUCH(TESTFILE, FILE_MODE, NULL);
}

static struct tst_test test = {
	.tcnt = ARRAY_SIZE(tc),
	.needs_root = 1,
	.needs_tmpdir = 1,
	.setup = setup,
	.test = run,
};
