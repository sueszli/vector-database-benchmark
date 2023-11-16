// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) International Business Machines Corp., 2007
 * Copyright (C) 2023 SUSE LLC Andrea Cervesato <andrea.cervesato@suse.com>
 */

/*\
 * [Description]
 *
 * Clone two processes using CLONE_NEWUTS, change hostname from the first
 * container and check if hostname didn't change inside the second one.
 */

#define _GNU_SOURCE

#include "tst_test.h"
#include "lapi/sched.h"

#define HOSTNAME "LTP_HOSTNAME"

static char *str_op;
static char *hostname1;
static char *hostname2;
static char originalhost[HOST_NAME_MAX];

static void reset_hostname(void)
{
	SAFE_SETHOSTNAME(originalhost, strlen(originalhost));
}

static void child1_run(void)
{
	SAFE_SETHOSTNAME(HOSTNAME, strlen(HOSTNAME));
	SAFE_GETHOSTNAME(hostname1, HOST_NAME_MAX);

	TST_CHECKPOINT_WAKE(0);
}

static void child2_run(void)
{
	TST_CHECKPOINT_WAIT(0);

	SAFE_GETHOSTNAME(hostname2, HOST_NAME_MAX);
}

static void run(void)
{
	const struct tst_clone_args cargs = {
		.flags = CLONE_NEWUTS,
		.exit_signal = SIGCHLD,
	};

	memset(hostname1, 0, HOST_NAME_MAX);
	memset(hostname2, 0, HOST_NAME_MAX);

	if (!str_op || !strcmp(str_op, "clone")) {
		tst_res(TINFO, "clone() with CLONE_NEWUTS");

		if (!SAFE_CLONE(&cargs)) {
			child1_run();
			return;
		}

		if (!SAFE_CLONE(&cargs)) {
			child2_run();
			return;
		}
	} else {
		tst_res(TINFO, "unshare() with CLONE_NEWUTS");

		if (!SAFE_FORK()) {
			SAFE_UNSHARE(CLONE_NEWUTS);
			child1_run();
			return;
		}

		if (!SAFE_FORK()) {
			SAFE_UNSHARE(CLONE_NEWUTS);
			child2_run();
			return;
		}
	}

	tst_reap_children();

	TST_EXP_PASS(strcmp(hostname1, HOSTNAME));
	TST_EXP_PASS(strcmp(hostname2, originalhost));

	reset_hostname();
}

static void setup(void)
{
	hostname1 = SAFE_MMAP(NULL, HOST_NAME_MAX, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	hostname2 = SAFE_MMAP(NULL, HOST_NAME_MAX, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	memset(originalhost, 0, HOST_NAME_MAX);

	SAFE_GETHOSTNAME(originalhost, HOST_NAME_MAX);
}

static void cleanup(void)
{
	SAFE_MUNMAP(hostname1, HOST_NAME_MAX);
	SAFE_MUNMAP(hostname2, HOST_NAME_MAX);

	reset_hostname();
}

static struct tst_test test = {
	.test_all = run,
	.setup = setup,
	.cleanup = cleanup,
	.needs_root = 1,
	.forks_child = 1,
	.needs_checkpoints = 1,
	.options = (struct tst_option[]) {
		{ "m:", &str_op, "Test execution mode <clone|unshare>" },
		{},
	},
};
