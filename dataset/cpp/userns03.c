// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) Huawei Technologies Co., Ltd., 2015
 * Copyright (C) 2022 SUSE LLC Andrea Cervesato <andrea.cervesato@suse.com>
 */

/*\
 * [Description]
 *
 * Verify that /proc/PID/uid_map and /proc/PID/gid_map contains three values
 * separated by white space:
 *
 * ID-inside-ns   ID-outside-ns   length
 *
 * ID-outside-ns is interpreted according to which process is opening the file.
 *
 * If the process opening the file is in the same user namespace as the process
 * PID, then ID-outside-ns is defined with respect to the parent user namespace.
 *
 * If the process opening the file is in a different user namespace, then
 * ID-outside-ns is defined with respect to the user namespace of the process
 * opening the file.
 *
 * The string "deny" would be written to /proc/self/setgroups before GID
 * check if setgroups is allowed, see kernel commits:
 *
 * - 9cc46516ddf4 ("userns: Add a knob to disable setgroups on a per user namespace basis")
 * - 66d2f338ee4c ("userns: Allow setting gid_maps without privilege when setgroups is disabled")
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdbool.h>
#include "tst_test.h"
#include "lapi/sched.h"
#include "common.h"

#define CHILD1UID 0
#define CHILD1GID 0
#define CHILD2UID 200
#define CHILD2GID 200
#define UID_MAP 0
#define GID_MAP 1

static void child_fn1(void)
{
	TST_CHECKPOINT_WAIT(0);
}

static void child_fn2(int cpid1, int parentuid, int parentgid)
{
	int uid, gid;
	char cpid1uidpath[BUFSIZ];
	char cpid1gidpath[BUFSIZ];
	int idinsidens, idoutsidens, length;

	TST_CHECKPOINT_WAIT(1);

	uid = geteuid();
	gid = getegid();

	TST_EXP_EQ_LI(uid, CHILD2UID);
	TST_EXP_EQ_LI(gid, CHILD2GID);

	/* Get the uid parameters of the child_fn2 process */
	SAFE_FILE_SCANF("/proc/self/uid_map", "%d %d %d", &idinsidens, &idoutsidens, &length);

	/* map file format:ID-inside-ns   ID-outside-ns   length
	 * If the process opening the file is in the same user namespace as
	 * the process PID, then ID-outside-ns is defined with respect to the
	 * parent user namespace
	 */
	tst_res(TINFO, "child2 checks /proc/cpid2/uid_map");

	if (idinsidens != CHILD2UID || idoutsidens != parentuid)
		tst_res(TFAIL, "unexpected: namespace ID inside=%d outside=%d", idinsidens, idoutsidens);
	else
		tst_res(TPASS, "expected namespaces IDs");

	sprintf(cpid1uidpath, "/proc/%d/uid_map", cpid1);
	SAFE_FILE_SCANF(cpid1uidpath, "%d %d %d", &idinsidens, &idoutsidens, &length);

	/* If the process opening the file is in a different user namespace,
	 * then ID-outside-ns is defined with respect to the user namespace
	 * of the process opening the file
	 */
	tst_res(TINFO, "child2 checks /proc/cpid1/uid_map");

	if (idinsidens != CHILD1UID || idoutsidens != CHILD2UID)
		tst_res(TFAIL, "unexpected: namespace ID inside=%d outside=%d", idinsidens, idoutsidens);
	else
		tst_res(TPASS, "expected namespaces IDs");

	sprintf(cpid1gidpath, "/proc/%d/gid_map", cpid1);
	SAFE_FILE_SCANF("/proc/self/gid_map", "%d %d %d", &idinsidens, &idoutsidens, &length);

	tst_res(TINFO, "child2 checks /proc/cpid2/gid_map");

	if (idinsidens != CHILD2GID || idoutsidens != parentgid)
		tst_res(TFAIL, "unexpected: namespace ID inside=%d outside=%d", idinsidens, idoutsidens);
	else
		tst_res(TPASS, "expected namespaces IDs");

	SAFE_FILE_SCANF(cpid1gidpath, "%d %d %d", &idinsidens, &idoutsidens, &length);

	tst_res(TINFO, "child1 checks /proc/cpid1/gid_map");

	if (idinsidens != CHILD1GID || idoutsidens != CHILD2GID)
		tst_res(TFAIL, "unexpected: namespace ID inside=%d outside=%d", idinsidens, idoutsidens);
	else
		tst_res(TPASS, "expected namespaces IDs");

	TST_CHECKPOINT_WAKE(0);
	TST_CHECKPOINT_WAKE(1);
}

static void run(void)
{
	const struct tst_clone_args args = {
		.flags = CLONE_NEWUSER,
		.exit_signal = SIGCHLD,
	};
	pid_t cpid1, cpid2;
	uid_t parentuid;
	gid_t parentgid;
	char path[BUFSIZ];
	int fd;

	parentuid = geteuid();
	parentgid = getegid();

	cpid1 = SAFE_CLONE(&args);
	if (!cpid1) {
		child_fn1();
		return;
	}

	cpid2 = SAFE_CLONE(&args);
	if (!cpid2) {
		child_fn2(cpid1, parentuid, parentgid);
		return;
	}

	if (access("/proc/self/setgroups", F_OK) == 0) {
		sprintf(path, "/proc/%d/setgroups", cpid1);

		fd = SAFE_OPEN(path, O_WRONLY, 0644);
		SAFE_WRITE(SAFE_WRITE_ALL, fd, "deny", 4);
		SAFE_CLOSE(fd);

		/* If the setgroups file has the value "deny",
		 * then the setgroups(2) system call can't
		 * subsequently be reenabled (by writing "allow" to
		 * the file) in this user namespace.  (Attempts to
		 * do so will fail with the error EPERM.)
		 */

		tst_res(TINFO, "Check if setgroups can be re-enabled");

		fd = SAFE_OPEN(path, O_WRONLY, 0644);
		TST_EXP_FAIL2(write(fd, "allow", 5), EPERM);
		SAFE_CLOSE(fd);

		sprintf(path, "/proc/%d/setgroups", cpid2);

		fd = SAFE_OPEN(path, O_WRONLY, 0644);
		SAFE_WRITE(SAFE_WRITE_ALL, fd, "deny", 4);
		SAFE_CLOSE(fd);
	}

	updatemap(cpid1, UID_MAP, CHILD1UID, parentuid);
	updatemap(cpid2, UID_MAP, CHILD2UID, parentuid);

	updatemap(cpid1, GID_MAP, CHILD1GID, parentgid);
	updatemap(cpid2, GID_MAP, CHILD2GID, parentgid);

	TST_CHECKPOINT_WAKE_AND_WAIT(1);
}

static struct tst_test test = {
	.test_all = run,
	.needs_root = 1,
	.forks_child = 1,
	.needs_checkpoints = 1,
	.needs_kconfigs = (const char *[]) {
		"CONFIG_USER_NS",
		NULL,
	},
};
