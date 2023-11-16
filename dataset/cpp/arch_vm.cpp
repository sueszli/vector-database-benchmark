/*
 * Copyright 2001, Travis Geiselbrecht. All rights reserved.
 * Copyright 2003-2005, Axel Dörfler, axeld@pinc-software.de.
 * Copyright 2019, Adrien Destugues, pulkomandy@pulkomandy.tk
 * Distributed under the terms of the MIT License.
 */

#include <vm/vm.h>
#include <vm/VMAddressSpace.h>
#include <arch/vm.h>
#include <boot/kernel_args.h>

#include "RISCV64VMTranslationMap.h"


#define TRACE_ARCH_VM
#ifdef TRACE_ARCH_VM
#	define TRACE(x) dprintf x
#else
#	define TRACE(x) ;
#endif


static uint64_t
SignExtendVirtAdr(uint64_t virtAdr)
{
	if (((uint64_t)1 << 38) & virtAdr)
		return virtAdr | 0xFFFFFF8000000000;
	return virtAdr;
}


static Pte*
LookupPte(phys_addr_t pageTable, addr_t virtAdr)
{
	Pte *pte = (Pte*)VirtFromPhys(pageTable);
	for (int level = 2; level > 0; level --) {
		pte += VirtAdrPte(virtAdr, level);
		if (!pte->isValid) {
			return NULL;
		}
		// TODO: Handle superpages (RWX=0 when not at lowest level)
		pte = (Pte*)VirtFromPhys(B_PAGE_SIZE * pte->ppn);
	}
	pte += VirtAdrPte(virtAdr, 0);
	return pte;
}



static void
WritePteFlags(uint32 flags)
{
	bool first = true;
	dprintf("{");
	for (uint32 i = 0; i < 32; i++) {
		if ((1 << i) & flags) {
			if (first)
				first = false;
			else
				dprintf(", ");

			switch (i) {
				case 0:
					dprintf("valid");
					break;
				case 1:
					dprintf("read");
					break;
				case 2:
					dprintf("write");
					break;
				case 3:
					dprintf("exec");
					break;
				case 4:
					dprintf("user");
					break;
				case 5:
					dprintf("global");
					break;
				case 6:
					dprintf("accessed");
					break;
				case 7:
					dprintf("dirty");
					break;
				default:
					dprintf("%" B_PRIu32, i);
			}
		}
	}
	dprintf("}");
}


class PageTableDumper
{
private:
	uint64 firstVirt;
	uint64 firstPhys;
	uint64 firstFlags;
	uint64 len;

public:
	PageTableDumper()
		:
		firstVirt(0),
		firstPhys(0),
		firstFlags(0),
		len(0)
	{}

	~PageTableDumper()
	{
		Write(0, 0, 0, 0);
	}

	void Write(uint64_t virtAdr, uint64_t physAdr, size_t size, uint64 flags) {
		if (virtAdr == firstVirt + len && physAdr == firstPhys + len && flags == firstFlags) {
			len += size;
		} else {
			if (len != 0) {
				dprintf("  0x%08" B_PRIxADDR " - 0x%08" B_PRIxADDR,
				firstVirt, firstVirt + (len - 1));
				dprintf(": 0x%08" B_PRIxADDR " - 0x%08" B_PRIxADDR ", %#" B_PRIxADDR ", ",
					firstPhys, firstPhys + (len - 1), len);
				WritePteFlags(firstFlags); dprintf("\n");
			}
			firstVirt = virtAdr;
			firstPhys = physAdr;
			firstFlags = flags;
			len = size;
		}
	}
};


static void
DumpPageTableInt(Pte* pte, uint64_t virtAdr, uint32_t level, PageTableDumper& dumper)
{
	for (uint32 i = 0; i < pteCount; i++) {
		if (pte[i].isValid) {
			if (!pte[i].isRead && !pte[i].isWrite && !pte[i].isExec) {

				if (level == 0)
					kprintf("  internal page table on level 0\n");

				DumpPageTableInt((Pte*)VirtFromPhys(B_PAGE_SIZE*pte[i].ppn),
					virtAdr + ((uint64_t)i << (pageBits + pteIdxBits * level)),
					level - 1, dumper);
			} else {
				dumper.Write(SignExtendVirtAdr(virtAdr
						+ ((uint64_t)i << (pageBits + pteIdxBits*level))),
					pte[i].ppn * B_PAGE_SIZE, 1 << (pageBits + pteIdxBits * level),
					pte[i].val & 0xff);
			}
		}
	}
}


static int
DumpPageTable(int argc, char** argv)
{
	int curArg = 1;
	SatpReg satp;
	bool isArea = false;
	addr_t base = 0;
	size_t size = 0;

	satp.val = Satp();
	while (curArg < argc && argv[curArg][0] == '-') {
		if (strcmp(argv[curArg], "-team") == 0) {
			curArg++;
			team_id id = strtoul(argv[curArg++], NULL, 0);
			VMAddressSpace* addrSpace = VMAddressSpace::DebugGet(id);
			if (addrSpace == NULL) {
				kprintf("could not find team %" B_PRId32 "\n", id);
				return 0;
			}
			satp.val = ((RISCV64VMTranslationMap*)
				addrSpace->TranslationMap())->Satp();
			isArea = false;
		} else if (strcmp(argv[curArg], "-area") == 0) {
			curArg++;
			uint64 areaId;
			if (!evaluate_debug_expression(argv[curArg++], &areaId, false))
				return 0;
			VMArea* area = VMAreas::Lookup((area_id)areaId);
			if (area == NULL) {
				kprintf("could not find area %" B_PRId32 "\n", (area_id)areaId);
				return 0;
			}
			satp.val = ((RISCV64VMTranslationMap*)
				area->address_space->TranslationMap())->Satp();
			base = area->Base();
			size = area->Size();
			kprintf("area %" B_PRId32 "(%s)\n", area->id, area->name);
				isArea = true;
		} else {
			kprintf("unknown flag \"%s\"\n", argv[curArg]);
			return 0;
		}
	}

	kprintf("satp: %#" B_PRIx64 "\n", satp.val);

	PageTableDumper dumper;

	if (!isArea) {
		Pte* root = (Pte*)VirtFromPhys(satp.ppn * B_PAGE_SIZE);
		DumpPageTableInt(root, 0, 2, dumper);
	} else {
		for (; size > 0; base += B_PAGE_SIZE, size -= B_PAGE_SIZE) {
			Pte* pte = LookupPte(satp.ppn * B_PAGE_SIZE, base);
			if (pte == NULL || !pte->isValid)
				continue;

			dumper.Write(base, pte->ppn * B_PAGE_SIZE, B_PAGE_SIZE, pte->val & 0xff);
		}
	}

	return 0;
}


static int
DumpVirtPage(int argc, char** argv)
{
	int curArg = 1;
	SatpReg satp;

	satp.val = Satp();
	while (curArg < argc && argv[curArg][0] == '-') {
		if (strcmp(argv[curArg], "-team") == 0) {
			curArg++;
			team_id id = strtoul(argv[curArg++], NULL, 0);
			VMAddressSpace* addrSpace = VMAddressSpace::DebugGet(id);
			if (addrSpace == NULL) {
				kprintf("could not find team %" B_PRId32 "\n", id);
				return 0;
			}
			satp.val = ((RISCV64VMTranslationMap*)
				addrSpace->TranslationMap())->Satp();
		} else {
			kprintf("unknown flag \"%s\"\n", argv[curArg]);
			return 0;
		}
	}

	kprintf("satp: %#" B_PRIx64 "\n", satp.val);

	uint64 virt = 0;
	if (!evaluate_debug_expression(argv[curArg++], &virt, false))
		return 0;

	virt = ROUNDDOWN(virt, B_PAGE_SIZE);

	Pte* pte = LookupPte(satp.ppn * B_PAGE_SIZE, virt);
	if (pte == NULL) {
		dprintf("not mapped\n");
		return 0;
	}

	PageTableDumper dumper;
	dumper.Write(virt, pte->ppn * B_PAGE_SIZE, B_PAGE_SIZE, pte->val & 0xff);

	return 0;
}


status_t
arch_vm_init(kernel_args *args)
{
	return B_OK;
}


status_t
arch_vm_init_post_area(kernel_args *args)
{
	void* address = (void*)args->arch_args.physMap.start;
	area_id area = vm_create_null_area(VMAddressSpace::KernelID(),
		"physical map area", &address, B_EXACT_ADDRESS,
		args->arch_args.physMap.size, 0);
	if (area < B_OK)
		return area;

	add_debugger_command("dump_page_table", &DumpPageTable, "Dump page table");
	add_debugger_command("dump_virt_page", &DumpVirtPage, "Dump virtual page mapping");

	return B_OK;
}


status_t
arch_vm_init_post_modules(kernel_args *args)
{
	return B_OK;
}


status_t
arch_vm_init_end(kernel_args *args)
{
	TRACE(("arch_vm_init_end(): %" B_PRIu32 " virtual ranges to keep:\n",
		args->arch_args.num_virtual_ranges_to_keep));

	for (int i = 0; i < (int)args->arch_args.num_virtual_ranges_to_keep; i++) {
		addr_range &range = args->arch_args.virtual_ranges_to_keep[i];

		TRACE(("  start: %p, size: %#" B_PRIxSIZE "\n", (void*)range.start, range.size));

#if 1
		// skip ranges outside the kernel address space
		if (!IS_KERNEL_ADDRESS(range.start)) {
			TRACE(("    no kernel address, skipping...\n"));
			continue;
		}

		phys_addr_t physicalAddress;
		void *address = (void*)range.start;
		if (vm_get_page_mapping(VMAddressSpace::KernelID(), range.start,
				&physicalAddress) != B_OK)
			panic("arch_vm_init_end(): No page mapping for %p\n", address);
		area_id area = vm_map_physical_memory(VMAddressSpace::KernelID(),
			"boot loader reserved area", &address,
			B_EXACT_ADDRESS, range.size,
			B_KERNEL_READ_AREA | B_KERNEL_WRITE_AREA,
			physicalAddress, true);
		if (area < 0) {
			panic("arch_vm_init_end(): Failed to create area for boot loader "
				"reserved area: %p - %p\n", (void*)range.start,
				(void*)(range.start + range.size));
		}
#endif
	}

#if 0
	// Throw away any address space mappings we've inherited from the boot
	// loader and have not yet turned into an area.
	vm_free_unused_boot_loader_range(0, 0xffffffff - B_PAGE_SIZE + 1);
#endif

	return B_OK;
}


void
arch_vm_aspace_swap(struct VMAddressSpace *from, struct VMAddressSpace *to)
{
	// This functions is only invoked when a userland thread is in the process
	// of dying. It switches to the kernel team and does whatever cleanup is
	// necessary (in case it is the team's main thread, it will delete the
	// team).
	// It is however not necessary to change the page directory. Userland team's
	// page directories include all kernel mappings as well. Furthermore our
	// arch specific translation map data objects are ref-counted, so they won't
	// go away as long as they are still used on any CPU.

	SetSatp(((RISCV64VMTranslationMap*)to->TranslationMap())->Satp());
	FlushTlbAllAsid(0);
}


bool
arch_vm_supports_protection(uint32 protection)
{
	return true;
}


void
arch_vm_unset_memory_type(VMArea *area)
{
}


status_t
arch_vm_set_memory_type(VMArea *area, phys_addr_t physicalBase, uint32 type)
{
	return B_OK;
}
