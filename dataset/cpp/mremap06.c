// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (c) 2023 SUSE LLC
 * Author: Vlastimil Babka <vbabka@suse.cz>
 * https://bugzilla.suse.com/attachment.cgi?id=867254
 * LTP port: Petr Vorel <pvorel@suse.cz>
 */

/*\
 * [Description]
 *
 * Bug reproducer for 7e7757876f25 ("mm/mremap: fix vm_pgoff in vma_merge() case 3")
 */

#define _GNU_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "tst_test.h"
#include "tst_safe_macros.h"

#define NUM_PAGES 3

static int fd;
static char *buf, *buf2;
static int page_size, mmap_size, mremap_size;

static struct tcase {
	size_t incompatible;
	const char *desc;
} tcases[] = {
	{
		.desc = "all pages with compatible mapping",
	},
	{
		.incompatible = 3,
		.desc = "third page's mapping incompatible",
	},
	{
		.incompatible = 1,
		.desc = "first page's mapping incompatible",
	},
};

static int check_pages(void)
{
	int fail = 0, i;
	char val;

	for (i = 0; i < (int)ARRAY_SIZE(tcases); i++) {
		val = buf[i * page_size];
		if (val != 0x30 + i) {
			tst_res(TFAIL, "page %d wrong value %d (0x%x)", i, val - 0x30, val);
			fail = 1;
		}
	}

	return fail;
}

static void do_test(unsigned int n)
{
	struct tcase *tc = &tcases[n];
	int ret;

	tst_res(TINFO, "%s", tc->desc);

	buf = SAFE_MMAP(0, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

	buf2 = mremap(buf + page_size, page_size, page_size,
			MREMAP_MAYMOVE|MREMAP_FIXED, buf + mremap_size);
	if (buf2 == MAP_FAILED)
		tst_brk(TBROK, "mremap() failed");

	if (tc->incompatible) {
		ret = mprotect(buf + (tc->incompatible-1)*page_size, page_size, PROT_READ);
		if (ret == -1)
			tst_brk(TBROK, "mprotect() failed");
	}

	buf2 = mremap(buf + mremap_size, page_size, page_size,
			MREMAP_MAYMOVE|MREMAP_FIXED, buf + page_size);
	if (buf2 == MAP_FAILED)
		tst_brk(TBROK, "mremap() failed");

	if (!check_pages())
		tst_res(TPASS, "mmap/mremap work properly");

	SAFE_MUNMAP(buf, mremap_size);
}

static void setup(void)
{
	int ret, i;

	page_size = getpagesize();
	mmap_size = (NUM_PAGES+1) * page_size;
	mremap_size = NUM_PAGES * page_size;

	fd = SAFE_OPEN("testfile", O_CREAT | O_RDWR | O_TRUNC, 0600);

	ret = fallocate(fd, 0, 0, mmap_size);
	if (ret == -1)
		tst_brk(TBROK, "fallocate() failed");

	buf = SAFE_MMAP(0, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

	for (i = 0; i < (int)ARRAY_SIZE(tcases)+1; i++)
		buf[i*page_size] = 0x30 + i;

	/* clear the page tables */
	SAFE_MUNMAP(buf, mmap_size);
}

static void cleanup(void)
{
	if (fd > 0)
		SAFE_CLOSE(fd);
}

static struct tst_test test = {
	.setup = setup,
	.cleanup = cleanup,
	.test = do_test,
	.needs_tmpdir = 1,
	.tcnt = ARRAY_SIZE(tcases),
	.tags = (struct tst_tag[]) {
		{"linux-git", "7e7757876f25"},
		{}
	},
};
