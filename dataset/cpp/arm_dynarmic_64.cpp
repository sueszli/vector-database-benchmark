// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cinttypes>
#include <memory>
#include <dynarmic/interface/A64/a64.h>
#include <dynarmic/interface/A64/config.h>
#include "common/assert.h"
#include "common/literals.h"
#include "common/logging/log.h"
#include "common/page_table.h"
#include "common/settings.h"
#include "core/arm/dynarmic/arm_dynarmic.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/arm/dynarmic/dynarmic_exclusive_monitor.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/debugger/debugger.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/svc.h"
#include "core/memory.h"

namespace Core {

using Vector = Dynarmic::A64::Vector;
using namespace Common::Literals;

class DynarmicCallbacks64 : public Dynarmic::A64::UserCallbacks {
public:
    explicit DynarmicCallbacks64(ARM_Dynarmic_64& parent_)
        : parent{parent_}, memory(parent.system.ApplicationMemory()),
          debugger_enabled{parent.system.DebuggerEnabled()},
          check_memory_access{debugger_enabled ||
                              !Settings::values.cpuopt_ignore_memory_aborts.GetValue()} {}

    u8 MemoryRead8(u64 vaddr) override {
        CheckMemoryAccess(vaddr, 1, Kernel::DebugWatchpointType::Read);
        return memory.Read8(vaddr);
    }
    u16 MemoryRead16(u64 vaddr) override {
        CheckMemoryAccess(vaddr, 2, Kernel::DebugWatchpointType::Read);
        return memory.Read16(vaddr);
    }
    u32 MemoryRead32(u64 vaddr) override {
        CheckMemoryAccess(vaddr, 4, Kernel::DebugWatchpointType::Read);
        return memory.Read32(vaddr);
    }
    u64 MemoryRead64(u64 vaddr) override {
        CheckMemoryAccess(vaddr, 8, Kernel::DebugWatchpointType::Read);
        return memory.Read64(vaddr);
    }
    Vector MemoryRead128(u64 vaddr) override {
        CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Read);
        return {memory.Read64(vaddr), memory.Read64(vaddr + 8)};
    }
    std::optional<u32> MemoryReadCode(u64 vaddr) override {
        if (!memory.IsValidVirtualAddressRange(vaddr, sizeof(u32))) {
            return std::nullopt;
        }
        return memory.Read32(vaddr);
    }

    void MemoryWrite8(u64 vaddr, u8 value) override {
        if (CheckMemoryAccess(vaddr, 1, Kernel::DebugWatchpointType::Write)) {
            memory.Write8(vaddr, value);
        }
    }
    void MemoryWrite16(u64 vaddr, u16 value) override {
        if (CheckMemoryAccess(vaddr, 2, Kernel::DebugWatchpointType::Write)) {
            memory.Write16(vaddr, value);
        }
    }
    void MemoryWrite32(u64 vaddr, u32 value) override {
        if (CheckMemoryAccess(vaddr, 4, Kernel::DebugWatchpointType::Write)) {
            memory.Write32(vaddr, value);
        }
    }
    void MemoryWrite64(u64 vaddr, u64 value) override {
        if (CheckMemoryAccess(vaddr, 8, Kernel::DebugWatchpointType::Write)) {
            memory.Write64(vaddr, value);
        }
    }
    void MemoryWrite128(u64 vaddr, Vector value) override {
        if (CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Write)) {
            memory.Write64(vaddr, value[0]);
            memory.Write64(vaddr + 8, value[1]);
        }
    }

    bool MemoryWriteExclusive8(u64 vaddr, std::uint8_t value, std::uint8_t expected) override {
        return CheckMemoryAccess(vaddr, 1, Kernel::DebugWatchpointType::Write) &&
               memory.WriteExclusive8(vaddr, value, expected);
    }
    bool MemoryWriteExclusive16(u64 vaddr, std::uint16_t value, std::uint16_t expected) override {
        return CheckMemoryAccess(vaddr, 2, Kernel::DebugWatchpointType::Write) &&
               memory.WriteExclusive16(vaddr, value, expected);
    }
    bool MemoryWriteExclusive32(u64 vaddr, std::uint32_t value, std::uint32_t expected) override {
        return CheckMemoryAccess(vaddr, 4, Kernel::DebugWatchpointType::Write) &&
               memory.WriteExclusive32(vaddr, value, expected);
    }
    bool MemoryWriteExclusive64(u64 vaddr, std::uint64_t value, std::uint64_t expected) override {
        return CheckMemoryAccess(vaddr, 8, Kernel::DebugWatchpointType::Write) &&
               memory.WriteExclusive64(vaddr, value, expected);
    }
    bool MemoryWriteExclusive128(u64 vaddr, Vector value, Vector expected) override {
        return CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Write) &&
               memory.WriteExclusive128(vaddr, value, expected);
    }

    void InterpreterFallback(u64 pc, std::size_t num_instructions) override {
        parent.LogBacktrace();
        LOG_ERROR(Core_ARM,
                  "Unimplemented instruction @ 0x{:X} for {} instructions (instr = {:08X})", pc,
                  num_instructions, memory.Read32(pc));
        ReturnException(pc, PrefetchAbort);
    }

    void InstructionCacheOperationRaised(Dynarmic::A64::InstructionCacheOperation op,
                                         u64 value) override {
        switch (op) {
        case Dynarmic::A64::InstructionCacheOperation::InvalidateByVAToPoU: {
            static constexpr u64 ICACHE_LINE_SIZE = 64;

            const u64 cache_line_start = value & ~(ICACHE_LINE_SIZE - 1);
            parent.system.InvalidateCpuInstructionCacheRange(cache_line_start, ICACHE_LINE_SIZE);
            break;
        }
        case Dynarmic::A64::InstructionCacheOperation::InvalidateAllToPoU:
            parent.system.InvalidateCpuInstructionCaches();
            break;
        case Dynarmic::A64::InstructionCacheOperation::InvalidateAllToPoUInnerSharable:
        default:
            LOG_DEBUG(Core_ARM, "Unprocesseed instruction cache operation: {}", op);
            break;
        }

        parent.jit.load()->HaltExecution(Dynarmic::HaltReason::CacheInvalidation);
    }

    void ExceptionRaised(u64 pc, Dynarmic::A64::Exception exception) override {
        switch (exception) {
        case Dynarmic::A64::Exception::WaitForInterrupt:
        case Dynarmic::A64::Exception::WaitForEvent:
        case Dynarmic::A64::Exception::SendEvent:
        case Dynarmic::A64::Exception::SendEventLocal:
        case Dynarmic::A64::Exception::Yield:
            return;
        case Dynarmic::A64::Exception::NoExecuteFault:
            LOG_CRITICAL(Core_ARM, "Cannot execute instruction at unmapped address {:#016x}", pc);
            ReturnException(pc, PrefetchAbort);
            return;
        default:
            if (debugger_enabled) {
                ReturnException(pc, InstructionBreakpoint);
                return;
            }

            parent.LogBacktrace();
            LOG_CRITICAL(Core_ARM, "ExceptionRaised(exception = {}, pc = {:08X}, code = {:08X})",
                         static_cast<std::size_t>(exception), pc, memory.Read32(pc));
        }
    }

    void CallSVC(u32 swi) override {
        parent.svc_swi = swi;
        parent.jit.load()->HaltExecution(SupervisorCall);
    }

    void AddTicks(u64 ticks) override {
        if (parent.uses_wall_clock) {
            return;
        }

        // Divide the number of ticks by the amount of CPU cores. TODO(Subv): This yields only a
        // rough approximation of the amount of executed ticks in the system, it may be thrown off
        // if not all cores are doing a similar amount of work. Instead of doing this, we should
        // device a way so that timing is consistent across all cores without increasing the ticks 4
        // times.
        u64 amortized_ticks = ticks / Core::Hardware::NUM_CPU_CORES;
        // Always execute at least one tick.
        amortized_ticks = std::max<u64>(amortized_ticks, 1);

        parent.system.CoreTiming().AddTicks(amortized_ticks);
    }

    u64 GetTicksRemaining() override {
        if (parent.uses_wall_clock) {
            if (!IsInterrupted()) {
                return minimum_run_cycles;
            }
            return 0U;
        }

        return std::max<s64>(parent.system.CoreTiming().GetDowncount(), 0);
    }

    u64 GetCNTPCT() override {
        return parent.system.CoreTiming().GetClockTicks();
    }

    bool CheckMemoryAccess(u64 addr, u64 size, Kernel::DebugWatchpointType type) {
        if (!check_memory_access) {
            return true;
        }

        if (!memory.IsValidVirtualAddressRange(addr, size)) {
            LOG_CRITICAL(Core_ARM, "Stopping execution due to unmapped memory access at {:#x}",
                         addr);
            parent.jit.load()->HaltExecution(PrefetchAbort);
            return false;
        }

        if (!debugger_enabled) {
            return true;
        }

        const auto match{parent.MatchingWatchpoint(addr, size, type)};
        if (match) {
            parent.halted_watchpoint = match;
            parent.jit.load()->HaltExecution(DataAbort);
            return false;
        }

        return true;
    }

    void ReturnException(u64 pc, Dynarmic::HaltReason hr) {
        parent.SaveContext(parent.breakpoint_context);
        parent.breakpoint_context.pc = pc;
        parent.jit.load()->HaltExecution(hr);
    }

    bool IsInterrupted() {
        return parent.system.Kernel().PhysicalCore(parent.core_index).IsInterrupted();
    }

    ARM_Dynarmic_64& parent;
    Core::Memory::Memory& memory;
    u64 tpidrro_el0 = 0;
    u64 tpidr_el0 = 0;
    const bool debugger_enabled{};
    const bool check_memory_access{};
    static constexpr u64 minimum_run_cycles = 10000U;
};

std::shared_ptr<Dynarmic::A64::Jit> ARM_Dynarmic_64::MakeJit(Common::PageTable* page_table,
                                                             std::size_t address_space_bits) const {
    Dynarmic::A64::UserConfig config;

    // Callbacks
    config.callbacks = cb.get();

    // Memory
    if (page_table) {
        config.page_table = reinterpret_cast<void**>(page_table->pointers.data());
        config.page_table_address_space_bits = address_space_bits;
        config.page_table_pointer_mask_bits = Common::PageTable::ATTRIBUTE_BITS;
        config.silently_mirror_page_table = false;
        config.absolute_offset_page_table = true;
        config.detect_misaligned_access_via_page_table = 16 | 32 | 64 | 128;
        config.only_detect_misalignment_via_page_table_on_page_boundary = true;

        config.fastmem_pointer = page_table->fastmem_arena;
        config.fastmem_address_space_bits = address_space_bits;
        config.silently_mirror_fastmem = false;

        config.fastmem_exclusive_access = config.fastmem_pointer != nullptr;
        config.recompile_on_exclusive_fastmem_failure = true;
    }

    // Multi-process state
    config.processor_id = core_index;
    config.global_monitor = &exclusive_monitor.monitor;

    // System registers
    config.tpidrro_el0 = &cb->tpidrro_el0;
    config.tpidr_el0 = &cb->tpidr_el0;
    config.dczid_el0 = 4;
    config.ctr_el0 = 0x8444c004;
    config.cntfrq_el0 = Hardware::CNTFREQ;

    // Unpredictable instructions
    config.define_unpredictable_behaviour = true;

    // Timing
    config.wall_clock_cntpct = uses_wall_clock;
    config.enable_cycle_counting = true;

    // Code cache size
#ifdef ARCHITECTURE_arm64
    config.code_cache_size = 128_MiB;
#else
    config.code_cache_size = 512_MiB;
#endif

    // Allow memory fault handling to work
    if (system.DebuggerEnabled()) {
        config.check_halt_on_memory_access = true;
    }

    // null_jit
    if (!page_table) {
        // Don't waste too much memory on null_jit
        config.code_cache_size = 8_MiB;
    }

    // Safe optimizations
    if (Settings::values.cpu_debug_mode) {
        if (!Settings::values.cpuopt_page_tables) {
            config.page_table = nullptr;
        }
        if (!Settings::values.cpuopt_block_linking) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::BlockLinking;
        }
        if (!Settings::values.cpuopt_return_stack_buffer) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::ReturnStackBuffer;
        }
        if (!Settings::values.cpuopt_fast_dispatcher) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::FastDispatch;
        }
        if (!Settings::values.cpuopt_context_elimination) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::GetSetElimination;
        }
        if (!Settings::values.cpuopt_const_prop) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::ConstProp;
        }
        if (!Settings::values.cpuopt_misc_ir) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::MiscIROpt;
        }
        if (!Settings::values.cpuopt_reduce_misalign_checks) {
            config.only_detect_misalignment_via_page_table_on_page_boundary = false;
        }
        if (!Settings::values.cpuopt_fastmem) {
            config.fastmem_pointer = nullptr;
            config.fastmem_exclusive_access = false;
        }
        if (!Settings::values.cpuopt_fastmem_exclusives) {
            config.fastmem_exclusive_access = false;
        }
        if (!Settings::values.cpuopt_recompile_exclusives) {
            config.recompile_on_exclusive_fastmem_failure = false;
        }
        if (!Settings::values.cpuopt_ignore_memory_aborts) {
            config.check_halt_on_memory_access = true;
        }
    } else {
        // Unsafe optimizations
        if (Settings::values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Unsafe) {
            config.unsafe_optimizations = true;
            if (Settings::values.cpuopt_unsafe_unfuse_fma) {
                config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_UnfuseFMA;
            }
            if (Settings::values.cpuopt_unsafe_reduce_fp_error) {
                config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_ReducedErrorFP;
            }
            if (Settings::values.cpuopt_unsafe_inaccurate_nan) {
                config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_InaccurateNaN;
            }
            if (Settings::values.cpuopt_unsafe_fastmem_check) {
                config.fastmem_address_space_bits = 64;
            }
            if (Settings::values.cpuopt_unsafe_ignore_global_monitor) {
                config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
            }
        }

        // Curated optimizations
        if (Settings::values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Auto) {
            config.unsafe_optimizations = true;
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_UnfuseFMA;
            config.fastmem_address_space_bits = 64;
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
        }

        // Paranoia mode for debugging optimizations
        if (Settings::values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Paranoid) {
            config.unsafe_optimizations = false;
            config.optimizations = Dynarmic::no_optimizations;
        }
    }

    return std::make_shared<Dynarmic::A64::Jit>(config);
}

HaltReason ARM_Dynarmic_64::RunJit() {
    return TranslateHaltReason(jit.load()->Run());
}

HaltReason ARM_Dynarmic_64::StepJit() {
    return TranslateHaltReason(jit.load()->Step());
}

u32 ARM_Dynarmic_64::GetSvcNumber() const {
    return svc_swi;
}

const Kernel::DebugWatchpoint* ARM_Dynarmic_64::HaltedWatchpoint() const {
    return halted_watchpoint;
}

void ARM_Dynarmic_64::RewindBreakpointInstruction() {
    LoadContext(breakpoint_context);
}

ARM_Dynarmic_64::ARM_Dynarmic_64(System& system_, bool uses_wall_clock_,
                                 DynarmicExclusiveMonitor& exclusive_monitor_,
                                 std::size_t core_index_)
    : ARM_Interface{system_, uses_wall_clock_},
      cb(std::make_unique<DynarmicCallbacks64>(*this)), core_index{core_index_},
      exclusive_monitor{exclusive_monitor_}, null_jit{MakeJit(nullptr, 48)}, jit{null_jit.get()} {}

ARM_Dynarmic_64::~ARM_Dynarmic_64() = default;

void ARM_Dynarmic_64::SetPC(u64 pc) {
    jit.load()->SetPC(pc);
}

u64 ARM_Dynarmic_64::GetPC() const {
    return jit.load()->GetPC();
}

u64 ARM_Dynarmic_64::GetSP() const {
    return jit.load()->GetSP();
}

u64 ARM_Dynarmic_64::GetReg(int index) const {
    return jit.load()->GetRegister(index);
}

void ARM_Dynarmic_64::SetReg(int index, u64 value) {
    jit.load()->SetRegister(index, value);
}

u128 ARM_Dynarmic_64::GetVectorReg(int index) const {
    return jit.load()->GetVector(index);
}

void ARM_Dynarmic_64::SetVectorReg(int index, u128 value) {
    jit.load()->SetVector(index, value);
}

u32 ARM_Dynarmic_64::GetPSTATE() const {
    return jit.load()->GetPstate();
}

void ARM_Dynarmic_64::SetPSTATE(u32 pstate) {
    jit.load()->SetPstate(pstate);
}

u64 ARM_Dynarmic_64::GetTlsAddress() const {
    return cb->tpidrro_el0;
}

void ARM_Dynarmic_64::SetTlsAddress(u64 address) {
    cb->tpidrro_el0 = address;
}

u64 ARM_Dynarmic_64::GetTPIDR_EL0() const {
    return cb->tpidr_el0;
}

void ARM_Dynarmic_64::SetTPIDR_EL0(u64 value) {
    cb->tpidr_el0 = value;
}

void ARM_Dynarmic_64::SaveContext(ThreadContext64& ctx) const {
    Dynarmic::A64::Jit* j = jit.load();
    ctx.cpu_registers = j->GetRegisters();
    ctx.sp = j->GetSP();
    ctx.pc = j->GetPC();
    ctx.pstate = j->GetPstate();
    ctx.vector_registers = j->GetVectors();
    ctx.fpcr = j->GetFpcr();
    ctx.fpsr = j->GetFpsr();
    ctx.tpidr = cb->tpidr_el0;
}

void ARM_Dynarmic_64::LoadContext(const ThreadContext64& ctx) {
    Dynarmic::A64::Jit* j = jit.load();
    j->SetRegisters(ctx.cpu_registers);
    j->SetSP(ctx.sp);
    j->SetPC(ctx.pc);
    j->SetPstate(ctx.pstate);
    j->SetVectors(ctx.vector_registers);
    j->SetFpcr(ctx.fpcr);
    j->SetFpsr(ctx.fpsr);
    SetTPIDR_EL0(ctx.tpidr);
}

void ARM_Dynarmic_64::SignalInterrupt() {
    jit.load()->HaltExecution(BreakLoop);
}

void ARM_Dynarmic_64::ClearInterrupt() {
    jit.load()->ClearHalt(BreakLoop);
}

void ARM_Dynarmic_64::ClearInstructionCache() {
    jit.load()->ClearCache();
}

void ARM_Dynarmic_64::InvalidateCacheRange(u64 addr, std::size_t size) {
    jit.load()->InvalidateCacheRange(addr, size);
}

void ARM_Dynarmic_64::ClearExclusiveState() {
    jit.load()->ClearExclusiveState();
}

void ARM_Dynarmic_64::PageTableChanged(Common::PageTable& page_table,
                                       std::size_t new_address_space_size_in_bits) {
    ThreadContext64 ctx{};
    SaveContext(ctx);

    auto key = std::make_pair(&page_table, new_address_space_size_in_bits);
    auto iter = jit_cache.find(key);
    if (iter != jit_cache.end()) {
        jit.store(iter->second.get());
        LoadContext(ctx);
        return;
    }
    std::shared_ptr new_jit = MakeJit(&page_table, new_address_space_size_in_bits);
    jit.store(new_jit.get());
    LoadContext(ctx);
    jit_cache.emplace(key, std::move(new_jit));
}

} // namespace Core
