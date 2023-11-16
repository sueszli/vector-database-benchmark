// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Author: Xie Ziyao <xieziyao@huawei.com>
 */

/*\
 * [Description]
 *
 * Verify that, epoll_pwait() and epoll_pwait2() return -1 and set errno to
 * EFAULT with a sigmask points outside user's accessible address space.
 */

#include <sys/epoll.h>

#include "tst_test.h"
#include "epoll_pwait_var.h"

static int efd, sfd[2];
static struct epoll_event e;
static void *bad_addr;

static void run(void)
{
	TST_EXP_FAIL(do_epoll_pwait(efd, &e, 1, -1, bad_addr),
		     EFAULT, "with an invalid sigmask pointer");
}

static void setup(void)
{
	epoll_pwait_init();

	SAFE_SOCKETPAIR(AF_UNIX, SOCK_STREAM, 0, sfd);

	efd = epoll_create(1);
	if (efd == -1)
		tst_brk(TBROK | TERRNO, "epoll_create()");

	e.events = EPOLLIN;
	if (epoll_ctl(efd, EPOLL_CTL_ADD, sfd[0], &e))
		tst_brk(TBROK | TERRNO, "epoll_ctl(..., EPOLL_CTL_ADD, ...)");
	SAFE_WRITE(SAFE_WRITE_ALL, sfd[1], "w", 1);

	bad_addr = tst_get_bad_addr(NULL);
}

static void cleanup(void)
{
	if (efd > 0)
		SAFE_CLOSE(efd);

	if (sfd[0] > 0)
		SAFE_CLOSE(sfd[0]);

	if (sfd[1] > 0)
		SAFE_CLOSE(sfd[1]);
}

static struct tst_test test = {
	.test_all = run,
	.setup = setup,
	.cleanup = cleanup,
	.test_variants = TEST_VARIANTS,
};
