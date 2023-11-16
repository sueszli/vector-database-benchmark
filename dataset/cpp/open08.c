// SPDX-License-Identifier: GPL-2.0-or-later
/*
 *   Copyright (c) 2013 Wanlong Gao <gaowanlong@cn.fujitsu.com>
 *   Copyright (c) 2018 Linux Test Project
 */

/*\
 * [Description]
 *
 * Verify that open() fails with:
 *
 * - EEXIST when pathname already exists and O_CREAT and O_EXCL were used
 * - EISDIR when pathname refers to a directory and the access requested
 * involved writing
 * - ENOTDIR when O_DIRECTORY was specified and pathname was not a directory
 * - ENAMETOOLONG when pathname was too long
 * - EACCES when requested access to the file is not allowed
 * - EFAULT when pathname points outside the accessible address space
 */

#define _GNU_SOURCE		/* for O_DIRECTORY */

#include <pwd.h>
#include "tst_test.h"

#define FLAGS_DESC(x) .flags = x, .desc = #x

static char *existing_fname = "open08_testfile";
static char *toolong_fname = "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyzabcdefghijklmnopqrstmnopqrstuvwxyz";
static char *dir_fname = "/tmp";
static char *user2_fname = "user2_0600";
static char *unmapped_fname;

struct test_case_t;

static struct test_case_t {
	char **fname;
	int flags;
	const char *desc;
	int error;
} tcases[] = {
	{&existing_fname, FLAGS_DESC(O_CREAT | O_EXCL), EEXIST},
	{&dir_fname, FLAGS_DESC(O_RDWR), EISDIR},
	{&existing_fname, FLAGS_DESC(O_DIRECTORY), ENOTDIR},
	{&toolong_fname, FLAGS_DESC(O_RDWR), ENAMETOOLONG},
	{&user2_fname, FLAGS_DESC(O_WRONLY), EACCES},
	{&unmapped_fname, FLAGS_DESC(O_CREAT), EFAULT},
};

static void verify_open(unsigned int i)
{
	TST_EXP_FAIL2(open(*tcases[i].fname, tcases[i].flags, 0644),
				tcases[i].error, "%s", tcases[i].desc);
}

static void setup(void)
{
	int fildes;
	char nobody_uid[] = "nobody";
	struct passwd *ltpuser;

	umask(0);

	SAFE_CREAT(user2_fname, 0600);

	/* Switch to nobody user for correct error code collection */
	ltpuser = getpwnam(nobody_uid);
	SAFE_SETGID(ltpuser->pw_gid);
	SAFE_SETUID(ltpuser->pw_uid);

	fildes = SAFE_CREAT(existing_fname, 0600);
	SAFE_CLOSE(fildes);

	unmapped_fname = tst_get_bad_addr(NULL);
}

static struct tst_test test = {
	.tcnt = ARRAY_SIZE(tcases),
	.needs_tmpdir = 1,
	.needs_root = 1,
	.setup = setup,
	.test = verify_open,
};
