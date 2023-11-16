// Copyright 2017 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/errors.h"
#include "core/hle/kernel/process.h"
#include "core/hle/kernel/vm_manager.h"
#include "core/memory.h"

TEST_CASE("Memory Basics", "[kernel][memory]") {
    auto mem = std::make_shared<BufferMem>(Memory::CITRA_PAGE_SIZE);
    MemoryRef block{mem};
    Core::Timing timing(1, 100);
    Core::System system;
    Memory::MemorySystem memory{system};
    Kernel::KernelSystem kernel(
        memory, timing, [] {}, Kernel::MemoryMode::Prod, 1,
        Kernel::New3dsHwCapabilities{false, false, Kernel::New3dsMemoryMode::Legacy});
    Kernel::Process process(kernel);
    SECTION("mapping memory") {
        // Because of the PageTable, Kernel::VMManager is too big to be created on the stack.
        auto manager = std::make_unique<Kernel::VMManager>(memory, process);
        auto result =
            manager->MapBackingMemory(Memory::HEAP_VADDR, block, static_cast<u32>(block.GetSize()),
                                      Kernel::MemoryState::Private);
        REQUIRE(result.Code() == RESULT_SUCCESS);

        auto vma = manager->FindVMA(Memory::HEAP_VADDR);
        CHECK(vma != manager->vma_map.end());
        CHECK(vma->second.size == static_cast<u32>(block.GetSize()));
        CHECK(vma->second.type == Kernel::VMAType::BackingMemory);
        CHECK(vma->second.backing_memory.GetPtr() == block.GetPtr());
        CHECK(vma->second.meminfo_state == Kernel::MemoryState::Private);
    }

    SECTION("unmapping memory") {
        // Because of the PageTable, Kernel::VMManager is too big to be created on the stack.
        auto manager = std::make_unique<Kernel::VMManager>(memory, process);
        auto result =
            manager->MapBackingMemory(Memory::HEAP_VADDR, block, static_cast<u32>(block.GetSize()),
                                      Kernel::MemoryState::Private);
        REQUIRE(result.Code() == RESULT_SUCCESS);

        ResultCode code =
            manager->UnmapRange(Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()));
        REQUIRE(code == RESULT_SUCCESS);

        auto vma = manager->FindVMA(Memory::HEAP_VADDR);
        CHECK(vma != manager->vma_map.end());
        CHECK(vma->second.type == Kernel::VMAType::Free);
        CHECK(vma->second.backing_memory.GetPtr() == nullptr);
    }

    SECTION("changing memory permissions") {
        // Because of the PageTable, Kernel::VMManager is too big to be created on the stack.
        auto manager = std::make_unique<Kernel::VMManager>(memory, process);
        auto result =
            manager->MapBackingMemory(Memory::HEAP_VADDR, block, static_cast<u32>(block.GetSize()),
                                      Kernel::MemoryState::Private);
        REQUIRE(result.Code() == RESULT_SUCCESS);

        ResultCode code = manager->ReprotectRange(
            Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()), Kernel::VMAPermission::Execute);
        CHECK(code == RESULT_SUCCESS);

        auto vma = manager->FindVMA(Memory::HEAP_VADDR);
        CHECK(vma != manager->vma_map.end());
        CHECK(vma->second.permissions == Kernel::VMAPermission::Execute);

        code = manager->UnmapRange(Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()));
        REQUIRE(code == RESULT_SUCCESS);
    }

    SECTION("changing memory state") {
        // Because of the PageTable, Kernel::VMManager is too big to be created on the stack.
        auto manager = std::make_unique<Kernel::VMManager>(memory, process);
        auto result =
            manager->MapBackingMemory(Memory::HEAP_VADDR, block, static_cast<u32>(block.GetSize()),
                                      Kernel::MemoryState::Private);
        REQUIRE(result.Code() == RESULT_SUCCESS);

        SECTION("reprotect memory range") {
            ResultCode code =
                manager->ReprotectRange(Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()),
                                        Kernel::VMAPermission::ReadWrite);
            REQUIRE(code == RESULT_SUCCESS);
        }

        SECTION("with invalid address") {
            ResultCode code = manager->ChangeMemoryState(
                0xFFFFFFFF, static_cast<u32>(block.GetSize()), Kernel::MemoryState::Locked,
                Kernel::VMAPermission::ReadWrite, Kernel::MemoryState::Aliased,
                Kernel::VMAPermission::Execute);
            CHECK(code == Kernel::ERR_INVALID_ADDRESS);
        }

        SECTION("ignoring the original permissions") {
            ResultCode code = manager->ChangeMemoryState(
                Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()), Kernel::MemoryState::Private,
                Kernel::VMAPermission::None, Kernel::MemoryState::Locked,
                Kernel::VMAPermission::Write);
            CHECK(code == RESULT_SUCCESS);

            auto vma = manager->FindVMA(Memory::HEAP_VADDR);
            CHECK(vma != manager->vma_map.end());
            CHECK(vma->second.permissions == Kernel::VMAPermission::Write);
            CHECK(vma->second.meminfo_state == Kernel::MemoryState::Locked);
        }

        SECTION("enforcing the original permissions with correct expectations") {
            ResultCode code = manager->ChangeMemoryState(
                Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()), Kernel::MemoryState::Private,
                Kernel::VMAPermission::ReadWrite, Kernel::MemoryState::Aliased,
                Kernel::VMAPermission::Execute);
            CHECK(code == RESULT_SUCCESS);

            auto vma = manager->FindVMA(Memory::HEAP_VADDR);
            CHECK(vma != manager->vma_map.end());
            CHECK(vma->second.permissions == Kernel::VMAPermission::Execute);
            CHECK(vma->second.meminfo_state == Kernel::MemoryState::Aliased);
        }

        SECTION("with incorrect permission expectations") {
            ResultCode code = manager->ChangeMemoryState(
                Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()), Kernel::MemoryState::Private,
                Kernel::VMAPermission::Execute, Kernel::MemoryState::Aliased,
                Kernel::VMAPermission::Execute);
            CHECK(code == Kernel::ERR_INVALID_ADDRESS_STATE);

            auto vma = manager->FindVMA(Memory::HEAP_VADDR);
            CHECK(vma != manager->vma_map.end());
            CHECK(vma->second.permissions == Kernel::VMAPermission::ReadWrite);
            CHECK(vma->second.meminfo_state == Kernel::MemoryState::Private);
        }

        SECTION("with incorrect state expectations") {
            ResultCode code = manager->ChangeMemoryState(
                Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()), Kernel::MemoryState::Locked,
                Kernel::VMAPermission::ReadWrite, Kernel::MemoryState::Aliased,
                Kernel::VMAPermission::Execute);
            CHECK(code == Kernel::ERR_INVALID_ADDRESS_STATE);

            auto vma = manager->FindVMA(Memory::HEAP_VADDR);
            CHECK(vma != manager->vma_map.end());
            CHECK(vma->second.permissions == Kernel::VMAPermission::ReadWrite);
            CHECK(vma->second.meminfo_state == Kernel::MemoryState::Private);
        }

        ResultCode code =
            manager->UnmapRange(Memory::HEAP_VADDR, static_cast<u32>(block.GetSize()));
        REQUIRE(code == RESULT_SUCCESS);
    }
}
