//
// Created by XingfengYang on 2020/6/15.
//

#include "kernel/assert.h"
#include "arm/vmm.h"
#include "arm/kernel_vmm.h"
#include "arm/mmu.h"
#include "arm/page.h"
#include "kernel/log.h"
#include "kernel/scheduler.h"
#include "kernel/type.h"
#include "libc/stdlib.h"

extern Scheduler cfsScheduler;

void virtual_memory_default_mapping_page(VirtualMemory *virtualMemory, uint32_t virtualAddress, uint32_t physicalPage){
    uint32_t l1Offset = (virtualAddress >> 30) & 0b11;
    uint32_t l2Offset = (virtualAddress >> 21) & 0b111111111;
    uint32_t l3Offset = (virtualAddress >> 12) & 0b111111111;
    uint32_t pageOffset = virtualAddress & 0xFFF;

    LogWarn("[vmm]: map %x to %x.\n", physicalPage, virtualAddress);

    PageTableEntry *level1PageTable = virtualMemory->pageTable;
    PageTableEntry level1PageTableEntry = level1PageTable[l1Offset];
    if (level1PageTableEntry.valid == 0) {
        // level 1 page table entry not set.
        uint64_t l2ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(
                virtualMemory->physicalPageAllocator, USAGE_PAGE_TABLE);

        level1PageTableEntry.valid = 1;
        level1PageTableEntry.table = 1;
        level1PageTableEntry.af = 1;
        level1PageTableEntry.base = l2ptPage;

        uint64_t ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(
                virtualMemory->physicalPageAllocator,
                USAGE_PAGE_TABLE);

        PageTableEntry *l2pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + l2ptPage * PAGE_SIZE;

        l2pt[0].valid = 1;
        l2pt[0].table = 1;
        l2pt[0].af = 1;
        l2pt[0].base = ptPage;

        PageTableEntry *pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + ptPage * PAGE_SIZE;

        pt[0].valid = 1;
        pt[0].table = 1;
        pt[0].af = 1;
        pt[0].base = (uint64_t) (virtualMemory->physicalPageAllocator->operations.allocPage4K(
                virtualMemory->physicalPageAllocator, USAGE_NORMAL));

    } else {
        PageTableEntry *level2PageTable = (PageTableEntry *) (level1PageTableEntry.base >> VA_OFFSET);
        PageTableEntry level2PageTableEntry = level2PageTable[l2Offset];
        if (level2PageTableEntry.valid == 0) {
            //   level 2 page table entry not set.
            uint64_t ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(
                    virtualMemory->physicalPageAllocator, USAGE_PAGE_TABLE);

            level2PageTableEntry.valid = 1;
            level2PageTableEntry.table = 1;
            level2PageTableEntry.af = 1;
            level2PageTableEntry.base = ptPage;

            PageTableEntry *pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + ptPage * PAGE_SIZE;

            pt[0].valid = 1;
            pt[0].table = 1;
            pt[0].af = 1;
            pt[0].base = (uint64_t) (virtualMemory->physicalPageAllocator->operations.allocPage4K(
                    virtualMemory->physicalPageAllocator, USAGE_NORMAL));
        } else {
            PageTableEntry *pageTable = (PageTableEntry *) (level2PageTable->base >> VA_OFFSET);
            PageTableEntry pageTableEntry = pageTable[l3Offset];
            if (pageTableEntry.valid == 0) {
                // page table entry not set
                pageTableEntry.valid = 1;
                pageTableEntry.table = 1;
                pageTableEntry.af = 1;
                pageTableEntry.base = (uint64_t) (physicalPage);

            } else {
                // should not be there, if goto there, means it not a page fault exception
            }
        }
    }
}

void virtual_memory_default_allocate_page(VirtualMemory *virtualMemory, uint32_t virtualAddress) {
    virtual_memory_default_mapping_page(virtualMemory, virtualAddress, (uint64_t) (virtualMemory->physicalPageAllocator->operations.allocPage4K(
                        virtualMemory->physicalPageAllocator, USAGE_NORMAL)));
}

void virtual_memory_default_mapping_pages(VirtualMemory *virtualMemory, uint32_t virtualAddress,
                                         uint32_t physicalAddress, uint32_t size) {
    uint32_t pages = size / (4 * KB);
    LogWarn("pages:%d \n",pages);
    for(uint32_t i = 0; i < pages; i++){
        virtual_memory_default_mapping_page(virtualMemory, virtualAddress + i * 4 * KB,  (physicalAddress + i * 4 * KB) << VA_OFFSET);
    }    
}

void virtual_memory_default_enable(VirtualMemory *virtualMemory) {
    write_ttbcr(CONFIG_ARM_LPAE << 31);
    LogInfo("[vmm]: ttbcr writed\n");

    write_ttbr0((uint32_t) virtualMemory->pageTable);
    LogInfo("[vmm]: ttbr0 writed\n");

    write_dacr(0x55555555);
    LogInfo("[vmm]: dacr writed\n");

    mmu_enable();
    LogInfo("[vmm]: vmm enabled\n");
}

void virtual_memory_default_disable(VirtualMemory *virtualMemory) { mmu_disable(); }

void virtual_memory_default_release(VirtualMemory *virtualMemory) {
    // TODO: release physical pages for page table when thread was fucking killed.
}

void virtual_memory_default_context_switch(VirtualMemory *old, VirtualMemory *new) {
    // TODO: switch page table when thread switch
}

uint32_t virtual_memory_default_translate_to_physical(struct VirtualMemory *virtualMemory, uint32_t address) {
    // calculate the physical address of buffer
    uint32_t l1Offset = address >> 30 & 0b11;
    uint32_t l2Offset = address >> 21 & 0b111111111;
    uint32_t l3Offset = address >> 12 & 0b111111111;
    uint32_t pageOffset = address & 0xFFF;

    PageTableEntry l1pte = virtualMemory->pageTable[l1Offset];
    PageTableEntry *level2PageTable = (PageTableEntry *) (l1pte.base << VA_OFFSET);
    PageTableEntry l2pte = level2PageTable[l2Offset];
    PageTableEntry *pageTable = (PageTableEntry *) (l2pte.base << VA_OFFSET);
    PageTableEntry pageTableEntry = pageTable[l3Offset];

    uint32_t physicalPageAddress = pageTableEntry.base;

    return physicalPageAddress + pageOffset;
}

uint32_t virtual_memory_default_get_user_str_len(struct VirtualMemory *virtualMemory, void *str) {
    uint32_t len = 0;
    uint32_t userAddress = (uint32_t) str;
    while (*(char *) (virtualMemory->operations.translateToPhysical(virtualMemory, userAddress)) != '\0') {
        len++;
        userAddress++;
    }
    return len;
}

void *virtual_memory_default_copy_to_kernel(struct VirtualMemory *virtualMemory, char *src, char *dest, uint32_t size) {

    // TODO: copy buffer from user space vmm to kernel space
    for (uint32_t i = 0; i < size; i++) {
        dest[i] = *(char *) (virtualMemory->operations.translateToPhysical(virtualMemory, (uint32_t) src + i));
    }
}

KernelStatus vmm_create(VirtualMemory *virtualMemory, PhysicalPageAllocator *physicalPageAllocator) {
    virtualMemory->operations.mappingPages = (VirtualMemoryOperationMappingPages) virtual_memory_default_mapping_pages;
    virtualMemory->operations.contextSwitch = (VirtualMemoryOperationContextSwitch) virtual_memory_default_context_switch;
    virtualMemory->operations.allocatePage = (VirtualMemoryOperationAllocatePage) virtual_memory_default_allocate_page;
    virtualMemory->operations.release = (VirtualMemoryOperationRelease) virtual_memory_default_release;
    virtualMemory->operations.enable = (VirtualMemoryOperationEnable) virtual_memory_default_enable;
    virtualMemory->operations.disable = (VirtualMemoryOperationDisable) virtual_memory_default_disable;
    virtualMemory->operations.copyToKernel = (VirtualMemoryOperationCopyToKernel) virtual_memory_default_copy_to_kernel;
    virtualMemory->operations.translateToPhysical = (VirtualMemoryOperationTranslateToPhysical) virtual_memory_default_translate_to_physical;
    virtualMemory->operations.getUserStrLen = (VirtualMemoryOperationGetUserStrLen) virtual_memory_default_get_user_str_len;

    virtualMemory->physicalPageAllocator = physicalPageAllocator;

    uint64_t ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(virtualMemory->physicalPageAllocator,
                                                                                   USAGE_PAGE_TABLE);

    if (ptPage == -1) {
        LogError("[VMM]: physical page allocate pt, no free page.\n")
    }

    PageTableEntry *pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + ptPage * PAGE_SIZE;

    DEBUG_ASSERT(pt != nullptr);
    if (pt == nullptr) {
        LogError("[VMM]: pt is null.\n");
        return ERROR;
    }

    pt[0].valid = 1;
    pt[0].table = 1;
    pt[0].af = 1;
    pt[0].base = (uint64_t) (
            virtualMemory->physicalPageAllocator->operations.allocPage4K(virtualMemory->physicalPageAllocator,
                                                                         USAGE_NORMAL));


    uint64_t l2ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(
            virtualMemory->physicalPageAllocator,
            USAGE_PAGE_TABLE);

    if (l2ptPage == -1) {
        LogError("[VMM]: physical page allocate l2pt, no free page.\n")
    }
    PageTableEntry *l2pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + l2ptPage * PAGE_SIZE;

    DEBUG_ASSERT(l2pt != nullptr);
    if (l2pt == nullptr) {
        LogError("[VMM]: l2pt is null.\n");
        return ERROR;
    }

    l2pt[0].valid = 1;
    l2pt[0].table = 1;
    l2pt[0].af = 1;
    l2pt[0].base = ptPage;


    uint64_t l1ptPage = virtualMemory->physicalPageAllocator->operations.allocPage4K(
            virtualMemory->physicalPageAllocator,
            USAGE_PAGE_TABLE);

    if (l1ptPage == -1) {
        LogError("[VMM]: physical page allocate l1pt, no free page.\n")
    }

    PageTableEntry *l1pt = (PageTableEntry *) virtualMemory->physicalPageAllocator->base + l1ptPage * PAGE_SIZE;

    DEBUG_ASSERT(l1pt != nullptr);
    if (l1pt == nullptr) {
        LogError("[VMM]: l1pt is null.\n");
        return ERROR;
    }


    l1pt[0].valid = 1;
    l1pt[0].table = 1;
    l1pt[0].af = 1;
    l1pt[0].base = l2ptPage;

    virtualMemory->pageTable = l1pt;

    return OK;
}

void do_page_fault(uint32_t address) {
    LogError("[vmm]: page fault at: %d .\n", address);
    // check is there is a thread running, if it was, then map for thread's vmm:
    // TODO: it not good, may be make some mistake when thread is running and kernel triggered this.

    Thread *currThread = cfsScheduler.operation.getCurrentThread(&cfsScheduler);
    if (currThread != nullptr) {
        // may be user triggered this
        VirtualMemory virtualMemory = currThread->memoryStruct.virtualMemory;
        virtualMemory.operations.allocatePage(&virtualMemory, address);
    } else {
        // kernel triggered this
        kernel_vmm_map(address);
    }
}
