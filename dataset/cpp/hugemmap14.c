// SPDX-License-Identifier: LGPL-2.1-or-later
/*
 * Copyright (C) 2005-2006 David Gibson & Adam Litke, IBM Corporation.
 * Author: David Gibson & Adam Litke
 */

/*\
 * [Description]
 * On some old ppc64 kernel, when huge page is mapped at below touching
 * 32 bit boundary (4GB - hpage_size), and normal page is mmaped
 * at just above it, it triggers a bug caused by off-by-one error.
 *
 * WARNING: The offsets and addresses used within are specifically
 * calculated to trigger the bug as it existed.  Don't mess with them
 * unless you *really* know what you're doing.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <sys/mount.h>
#include <limits.h>
#include <sys/param.h>
#include <sys/types.h>

#include "hugetlb.h"

#define FOURGB (1ULL << 32)
#define MNTPOINT "hugetlbfs/"
static int  fd = -1;
static unsigned long long hpage_size;
static int page_size;

static void run_test(void)
{
	void *p, *q = NULL, *r = NULL;
	unsigned long long lowaddr, highaddr;
	unsigned long long below_start;
	unsigned long long above_end;

	/*
	 * We use a low address right below 4GB so we can test for
	 * off-by-one errors
	 */
	lowaddr = FOURGB - hpage_size;
	tst_res(TINFO, "Mapping hugepage at %llx...", lowaddr);
	p = mmap((void *)lowaddr, hpage_size, PROT_READ|PROT_WRITE,
		 MAP_SHARED|MAP_FIXED, fd, 0);
	if (p == MAP_FAILED) {
		/* This is last low slice - 256M just before 4G */
		below_start = FOURGB - 256ULL*1024*1024;
		above_end = FOURGB;

		if (range_is_mapped(below_start, above_end) == 1) {
			tst_res(TINFO|TERRNO, "region (4G-256M)-4G is not free & "
					"mmap() failed expected");
			tst_res(TPASS, "Successful but inconclusive");
		} else
			tst_res(TFAIL|TERRNO, "mmap() huge failed unexpected");
		goto cleanup;
	}
	if (p != (void *)lowaddr) {
		tst_res(TFAIL, "Wrong address with MAP_FIXED huge");
		goto cleanup;
	}
	memset(p, 0, hpage_size);

	/* Test for off by one errors */
	highaddr = FOURGB;
	tst_res(TINFO, "Mapping normal page at %llx...", highaddr);
	q = mmap((void *)highaddr, page_size, PROT_READ|PROT_WRITE,
		 MAP_SHARED|MAP_FIXED|MAP_ANONYMOUS, 0, 0);
	if (q == MAP_FAILED) {
		below_start = FOURGB;
		above_end = FOURGB + page_size;

		if (range_is_mapped(below_start, above_end) == 1) {
			tst_res(TINFO|TERRNO, "region 4G-(4G+page) is not free & "
					"mmap() failed expected");
			tst_res(TPASS, "Successful but inconclusive");
		} else
			tst_res(TFAIL|TERRNO, "mmap() normal 1 failed unexpected");
		goto cleanup;
	}
	if (q != (void *)highaddr) {
		tst_res(TFAIL, "Wrong address with MAP_FIXED normal 1");
		goto cleanup;
	}
	memset(q, 0, page_size);

	/*
	 * Why this address?  Well on ppc64, we're working with 256MB
	 * segment numbers, hence >>28.  In practice the shift
	 * instructions only start wrapping around with shifts 128 or
	 * greater.
	 */
	highaddr = ((lowaddr >> 28) + 128) << 28;
	tst_res(TINFO, "Mapping normal page at %llx...", highaddr);
	r = mmap((void *)highaddr, page_size, PROT_READ|PROT_WRITE,
		 MAP_SHARED|MAP_FIXED|MAP_ANONYMOUS, 0, 0);
	if (r == MAP_FAILED) {
		below_start = highaddr;
		above_end = highaddr + page_size;

		if (range_is_mapped(below_start, above_end) == 1) {
			tst_res(TINFO|TERRNO, "region haddr-(haddr+page) not free & "
					"mmap() failed unexpected");
			tst_res(TPASS, "Successful but inconclusive");
		}
		tst_res(TFAIL|TERRNO, "mmap() normal 2 failed unexpected");
		goto cleanup;
	}
	if (r != (void *)highaddr) {
		tst_res(TFAIL, "Wrong address with MAP_FIXED normal 2");
		goto cleanup;
	}
	memset(r, 0, page_size);
	tst_res(TPASS, "Successful");

cleanup:
	if (p && p != MAP_FAILED)
		SAFE_MUNMAP(p, hpage_size);
	if (q && q != MAP_FAILED)
		SAFE_MUNMAP(q, page_size);
	if (r && r != MAP_FAILED)
		SAFE_MUNMAP(r, page_size);
}

static void setup(void)
{
	page_size = getpagesize();
	hpage_size = SAFE_READ_MEMINFO("Hugepagesize:")*1024;

	if (sizeof(void *) <= 4)
		tst_brk(TCONF, "Machine must be >32 bit");
	if (hpage_size > FOURGB)
		tst_brk(TCONF, "Huge page size is too large");
	fd = tst_creat_unlinked(MNTPOINT, 0);
}

static void cleanup(void)
{
	if (fd > 0)
		SAFE_CLOSE(fd);
}

static struct tst_test test = {
	.tags = (struct tst_tag[]) {
		{"linux-git", "9a94c5793a7b"},
		{}
	},
	.needs_root = 1,
	.mntpoint = MNTPOINT,
	.needs_hugetlbfs = 1,
	.needs_tmpdir = 1,
	.setup = setup,
	.cleanup = cleanup,
	.test_all = run_test,
	.hugepages = {2, TST_NEEDS},
};
