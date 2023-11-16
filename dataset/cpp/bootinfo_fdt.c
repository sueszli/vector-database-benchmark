/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2023, Unikraft GmbH and The Unikraft Authors.
 * Licensed under the BSD-3-Clause License (the "License").
 * You may not use this file except in compliance with the License.
 */

#include <uk/essentials.h>
#include <uk/plat/common/bootinfo.h>
#include <uk/plat/common/sections.h>
#include <libfdt.h>

#define ukplat_bootinfo_crash(s)		ukplat_crash()

static void fdt_bootinfo_mem_mrd(struct ukplat_bootinfo *bi, void *fdtp)
{
	struct ukplat_memregion_desc mrd = {0};
	int prop_len, prop_min_len;
	__u64 mem_base, mem_sz;
	int naddr, nsz, nmem;
	const __u64 *regs;
	int rc;

	nmem = fdt_node_offset_by_prop_value(fdtp, -1, "device_type",
					     "memory", sizeof("memory"));
	if (unlikely(nmem < 0))
		ukplat_bootinfo_crash("No memory found in DTB");

	naddr = fdt_address_cells(fdtp, nmem);
	if (unlikely(naddr < 0 || naddr >= FDT_MAX_NCELLS))
		ukplat_bootinfo_crash("Could not find proper address cells!");

	nsz = fdt_size_cells(fdtp, nmem);
	if (unlikely(nsz < 0 || nsz >= FDT_MAX_NCELLS))
		ukplat_bootinfo_crash("Could not find proper size cells!");

	/*
	 * The property must contain at least the start address
	 * and size, each of which is 8-bytes.
	 * For now, we only support one memory bank.
	 * TODO: Support more than one memory@ node/regs/ranges properties.
	 */
	prop_len = 0;
	prop_min_len = (int)sizeof(fdt32_t) * (naddr + nsz);
	regs = fdt_getprop(fdtp, nmem, "reg", &prop_len);
	if (unlikely(!regs || prop_len < prop_min_len))
		ukplat_bootinfo_crash("Bad 'reg' property or more than one memory bank.");

	mem_sz = fdt64_to_cpu(regs[1]);
	mem_base = fdt64_to_cpu(regs[0]);
	if (unlikely(!RANGE_CONTAIN(mem_base, mem_sz,
				    __BASE_ADDR, __END - __BASE_ADDR)))
		ukplat_bootinfo_crash("Image outside of RAM");

	/* Check that we are not placed at the top of the memory region */
	mrd.len   = __BASE_ADDR - mem_base;
	if (!mrd.len)
		goto end_mrd;

	mrd.vbase = (__vaddr_t)mem_base;
	mrd.pbase = (__paddr_t)mem_base;
	mrd.type  = UKPLAT_MEMRT_FREE;
	mrd.flags = UKPLAT_MEMRF_READ | UKPLAT_MEMRF_WRITE;

	rc = ukplat_memregion_list_insert(&bi->mrds, &mrd);
	if (unlikely(rc < 0))
		ukplat_bootinfo_crash("Could not add free memory descriptor");

end_mrd:
	/* Check that we are not placed at the end of the memory region */
	mrd.len   = mem_base + mem_sz - __END;
	if (!mrd.len)
		return;

	mrd.vbase = (__vaddr_t)__END;
	mrd.pbase = (__paddr_t)__END;
	mrd.type  = UKPLAT_MEMRT_FREE;
	mrd.flags = UKPLAT_MEMRF_READ | UKPLAT_MEMRF_WRITE;

	rc = ukplat_memregion_list_insert(&bi->mrds, &mrd);
	if (unlikely(rc < 0))
		ukplat_bootinfo_crash("Could not add free memory descriptor");
}

static void fdt_bootinfo_cmdl_mrd(struct ukplat_bootinfo *bi, void *fdtp)
{
	const void *fdt_cmdl;
	int fdt_cmdl_len;
	__sz cmdl_len;
	int nchosen;
	char *cmdl;

	nchosen = fdt_path_offset(fdtp, "/chosen");
	if (unlikely(!nchosen))
		return;

	fdt_cmdl = fdt_getprop(fdtp, nchosen, "bootargs", &fdt_cmdl_len);
	if (unlikely(!fdt_cmdl || fdt_cmdl_len <= 0))
		return;

	cmdl = ukplat_memregion_alloc(fdt_cmdl_len + sizeof(CONFIG_UK_NAME) + 1,
				      UKPLAT_MEMRT_CMDLINE,
				      UKPLAT_MEMRF_READ |
				      UKPLAT_MEMRF_MAP);
	if (unlikely(!cmdl))
		ukplat_bootinfo_crash("Command-line alloc failed\n");

	cmdl_len = sizeof(CONFIG_UK_NAME);
	strncpy(cmdl, CONFIG_UK_NAME, cmdl_len);
	cmdl[cmdl_len - 1] = ' ';
	strncpy(cmdl + cmdl_len, fdt_cmdl, fdt_cmdl_len);
	cmdl_len += fdt_cmdl_len;
	cmdl[cmdl_len] = '\0';

	bi->cmdline = (__u64)cmdl;
	bi->cmdline_len = (__u64)cmdl_len;
}

/* Ideally the initrd nodes would use #address-cells, yet these nodes are not
 * defined by the device-tree spec, and as such there is no formal requirement
 * that they do so. In fact, QEMU virt uses a 32-bit address here, despite
 * defining 2 address cells. To handle such cases, use the property length to
 * determine the correct address.
 */
#define initrd_addr(val, len)                          \
	(len == 4 ? fdt32_to_cpu(val) : fdt64_to_cpu(val))

static void fdt_bootinfo_initrd_mrd(struct ukplat_bootinfo *bi, void *fdtp)
{
	struct ukplat_memregion_desc mrd = {0};
	const __u64 *fdt_initrd_start;
	const __u64 *fdt_initrd_end;
	int start_len, end_len;
	int nchosen;
	int rc;

	nchosen = fdt_path_offset(fdtp, "/chosen");
	if (unlikely(!nchosen))
		return;

	fdt_initrd_start = fdt_getprop(fdtp, nchosen, "linux,initrd-start",
				       &start_len);
	if (unlikely(!fdt_initrd_start || start_len <= 0))
		return;

	fdt_initrd_end = fdt_getprop(fdtp, nchosen, "linux,initrd-end",
				     &end_len);
	if (unlikely(!fdt_initrd_end || end_len <= 0))
		return;

	mrd.vbase = initrd_addr(fdt_initrd_start[0], start_len);
	mrd.pbase = initrd_addr(fdt_initrd_start[0], start_len);
	mrd.len = initrd_addr(fdt_initrd_end[0], end_len) -
		  initrd_addr(fdt_initrd_start[0], start_len);
	mrd.type = UKPLAT_MEMRT_INITRD;
	mrd.flags = UKPLAT_MEMRF_READ | UKPLAT_MEMRF_MAP;

	rc = ukplat_memregion_list_insert(&bi->mrds, &mrd);
	if (unlikely(rc < 0))
		ukplat_bootinfo_crash("Could not add initrd memory descriptor");
}

static void fdt_bootinfo_fdt_mrd(struct ukplat_bootinfo *bi, void *fdtp)
{
	struct ukplat_memregion_desc mrd = {0};
	int rc;

	mrd.vbase = (__vaddr_t)fdtp;
	mrd.pbase = (__paddr_t)fdtp;
	mrd.len   = fdt_totalsize(fdtp);
	mrd.type  = UKPLAT_MEMRT_DEVICETREE;
	mrd.flags = UKPLAT_MEMRF_READ | UKPLAT_MEMRF_MAP;

	rc = ukplat_memregion_list_insert(&bi->mrds, &mrd);
	if (unlikely(rc < 0))
		ukplat_bootinfo_crash("Could not insert DT memory descriptor");
}

void ukplat_bootinfo_fdt_setup(void *fdtp)
{
	struct ukplat_bootinfo *bi;

	bi = ukplat_bootinfo_get();
	if (unlikely(!bi))
		ukplat_bootinfo_crash("Invalid bootinfo");

	if (unlikely(fdt_check_header(fdtp)))
		ukplat_bootinfo_crash("Invalid DTB");

	fdt_bootinfo_fdt_mrd(bi, fdtp);
	fdt_bootinfo_mem_mrd(bi, fdtp);
	fdt_bootinfo_initrd_mrd(bi, fdtp);
	ukplat_memregion_list_coalesce(&bi->mrds);

	/* We use this after coalescing/sorted because this calls
	 * `ukplat_memregion_alloc()` which would be unsafe to do so before
	 * knowing that the memory region descriptor list has been coalesced
	 * and sorted.
	 */
	fdt_bootinfo_cmdl_mrd(bi, fdtp);

	bi->dtb = (__u64)fdtp;
}
