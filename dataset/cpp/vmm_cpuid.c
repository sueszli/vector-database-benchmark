/*-
 * SPDX-License-Identifier: BSD-2-Clause-FreeBSD
 *
 * Copyright (c) 2011 NetApp, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY NETAPP, INC ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL NETAPP, INC OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $FreeBSD$
 */
/*
 * This file and its contents are supplied under the terms of the
 * Common Development and Distribution License ("CDDL"), version 1.0.
 * You may only use this file in accordance with the terms of version
 * 1.0 of the CDDL.
 *
 * A full copy of the text of the CDDL should have accompanied this
 * source.  A copy of the CDDL is also available via the Internet at
 * http://www.illumos.org/license/CDDL.
 *
 * Copyright 2014 Pluribus Networks Inc.
 * Copyright 2018 Joyent, Inc.
 * Copyright 2022 Oxide Computer Company
 */

#include <sys/types.h>
#include <sys/stdbool.h>
#include <sys/errno.h>

#include <machine/md_var.h>
#include <machine/specialreg.h>

#include <machine/vmm.h>
#include <sys/vmm_kernel.h>

#include "vmm_host.h"
#include "vmm_util.h"

/*
 * CPUID Emulation
 *
 * All CPUID instruction exits are handled by the in-kernel emulation.
 *
 * ----------------
 * Legacy Emulation
 * ----------------
 *
 * Originally, the kernel vmm portion of bhyve relied on fixed logic to filter
 * and/or generate CPUID results based on what was reported by the host CPU, as
 * well as attributes of the VM (such as CPU topology, and enabled features).
 * This is largely adequate to expose CPU capabilities to the guest in manner
 * which allows it to operate properly.
 *
 * ------------------------------
 * Userspace-Controlled Emulation
 * ------------------------------
 *
 * In certain situations, more control over the CPUID emulation results present
 * to the guest is desired.  Live migration between physical hosts is one such
 * example, where the underlying CPUs, or at least their microcode, may differ
 * between the source and destination.  In such cases, where changes to the
 * CPUID results cannot be tolerated, the userspace portion of the VMM can be in
 * complete control over the leaves which are presented to the guest.  It may
 * still consult the "legacy" CPUID data for guidance about which CPU features
 * are safe to expose (due to hypervisor limitations, etc).  This leaf
 * information is configured on a per-vCPU basis.
 *
 * The emulation entries provided by userspace are expected to be in sorted
 * order, running from lowest function and index to highest.
 *
 * For example:
 * (func: 00h idx: 00h) ->
 *     (flags: 0, eax: highest std leaf, ebx-edx: vendor id)
 * (func: 0Dh idx: 00h) ->
 *     (flags: VCE_FLAG_MATCH_INDEX, eax - edx: XCR0/XSAVE info)
 * (func: 0Dh idx: 01h) ->
 *     (flags: VCE_FLAG_MATCH_INDEX, eax - edx: XSAVE/XSAVEOPT details)
 *     ...
 * (func: 0Dh idx: 07H) ->
 *     (flags: VCE_FLAG_MATCH_INDEX, eax - edx: AVX-512 details)
 * (func: 8000000h idx: 0h) ->
 *     (flags: 0, eax: highest extd leaf ...)
 *     ...
 */


#define	CPUID_TYPE_MASK	0xf0000000
#define	CPUID_TYPE_STD	0x00000000
#define	CPUID_TYPE_EXTD	0x80000000

static const struct vcpu_cpuid_entry cpuid_empty_entry = { 0 };

/*
 * Given the CPUID configuration for a vCPU, locate the entry which matches the
 * provided function/index tuple.  The entries list is walked in order, and the
 * first valid match based on the function/index and flags will be emitted.
 *
 * If no match is found, but Intel-style fallback is configured, then the
 * highest standard leaf encountered will be emitted.
 */
static const struct vcpu_cpuid_entry *
cpuid_find_entry(const vcpu_cpuid_config_t *cfg, uint32_t func, uint32_t idx)
{
	const struct vcpu_cpuid_entry *last_std = NULL;
	const bool intel_fallback =
	    (cfg->vcc_flags & VCC_FLAG_INTEL_FALLBACK) != 0;
	bool matched_leaf = false;

	ASSERT0(cfg->vcc_flags & VCC_FLAG_LEGACY_HANDLING);

	for (uint_t i = 0; i < cfg->vcc_nent; i++) {
		const struct vcpu_cpuid_entry *ent = &cfg->vcc_entries[i];
		const bool ent_is_std =
		    (ent->vce_function & CPUID_TYPE_MASK) == CPUID_TYPE_STD;
		const bool ent_must_match_idx =
		    (ent->vce_flags & VCE_FLAG_MATCH_INDEX) != 0;

		if (ent_is_std) {
			/*
			 * Keep track of the last "standard" leaf for
			 * Intel-style fallback behavior.
			 *
			 * This does currently not account for the sub-leaf
			 * index matching behavior for fallback described in the
			 * SDM.  It is not clear if any consumers rely on such
			 * matching when encountering fallback.
			 */
			last_std = ent;
		}
		if (ent->vce_function == func) {
			if (ent->vce_index == idx || !ent_must_match_idx) {
				return (ent);
			}
			/*
			 * Make note of when the top-level leaf matches, even
			 * when the index does not.
			 */
			matched_leaf = true;
		} else if (ent->vce_function > func) {
			if ((ent->vce_function & CPUID_TYPE_MASK) ==
			    (func & CPUID_TYPE_MASK)) {
				/*
				 * We are beyond a valid leaf to match, but have
				 * not exceeded the maximum leaf for this "type"
				 * (standard, extended, hvm, etc), so return an
				 * empty entry.
				 */
				return (&cpuid_empty_entry);
			} else {
				/*
				 * Otherwise, we can stop now, having gone
				 * beyond the last entry which could match the
				 * target function in a sorted list.
				 */
				break;
			}
		}
	}

	if (matched_leaf || !intel_fallback) {
		return (&cpuid_empty_entry);
	} else {
		return (last_std);
	}
}

void
vcpu_emulate_cpuid(struct vm *vm, int vcpuid, uint64_t *rax, uint64_t *rbx,
    uint64_t *rcx, uint64_t *rdx)
{
	const vcpu_cpuid_config_t *cfg = vm_cpuid_config(vm, vcpuid);

	ASSERT3P(rax, !=, NULL);
	ASSERT3P(rbx, !=, NULL);
	ASSERT3P(rcx, !=, NULL);
	ASSERT3P(rdx, !=, NULL);

	/* Fall back to legacy handling if specified */
	if ((cfg->vcc_flags & VCC_FLAG_LEGACY_HANDLING) != 0) {
		uint32_t regs[4] = { *rax, 0, *rcx, 0 };

		legacy_emulate_cpuid(vm, vcpuid, &regs[0], &regs[1], &regs[2],
		    &regs[3]);
		/* CPUID clears the upper 32-bits of the long-mode registers. */
		*rax = regs[0];
		*rbx = regs[1];
		*rcx = regs[2];
		*rdx = regs[3];
		return;
	}

	const struct vcpu_cpuid_entry *ent = cpuid_find_entry(cfg, *rax, *rcx);
	ASSERT(ent != NULL);
	/* CPUID clears the upper 32-bits of the long-mode registers. */
	*rax = ent->vce_eax;
	*rbx = ent->vce_ebx;
	*rcx = ent->vce_ecx;
	*rdx = ent->vce_edx;
}

/*
 * Get the current CPUID emulation configuration for this vCPU.
 *
 * Only the existing flags will be emitted if the vCPU is configured for legacy
 * operation via the VCC_FLAG_LEGACY_HANDLING flag.  If in userspace-controlled
 * mode, then we will attempt to copy the existing entries into vcc_entries,
 * its side specified by vcc_nent.
 *
 * Regardless of whether vcc_entries is adequately sized (or even present),
 * vcc_nent will be set to the number of existing entries.
 */
int
vm_get_cpuid(struct vm *vm, int vcpuid, vcpu_cpuid_config_t *res)
{
	if (vcpuid < 0 || vcpuid > VM_MAXCPU) {
		return (EINVAL);
	}

	const vcpu_cpuid_config_t *src = vm_cpuid_config(vm, vcpuid);
	if (src->vcc_nent > res->vcc_nent) {
		res->vcc_nent = src->vcc_nent;
		return (E2BIG);
	} else if (src->vcc_nent != 0) {
		bcopy(src->vcc_entries, res->vcc_entries,
		    src->vcc_nent * sizeof (struct vcpu_cpuid_entry));
	}
	res->vcc_flags = src->vcc_flags;
	res->vcc_nent = src->vcc_nent;
	return (0);
}

/*
 * Set the CPUID emulation configuration for this vCPU.
 *
 * If VCC_FLAG_LEGACY_HANDLING is set in vcc_flags, then vcc_nent is expected to
 * be set to 0, as configuring a list of entries would be useless when using the
 * legacy handling.
 *
 * Any existing entries which are configured are freed, and the newly provided
 * ones will be copied into their place.
 */
int
vm_set_cpuid(struct vm *vm, int vcpuid, const vcpu_cpuid_config_t *src)
{
	if (vcpuid < 0 || vcpuid > VM_MAXCPU) {
		return (EINVAL);
	}
	if (src->vcc_nent > VMM_MAX_CPUID_ENTRIES) {
		return (EINVAL);
	}
	if ((src->vcc_flags & ~VCC_FLAGS_VALID) != 0) {
		return (EINVAL);
	}
	if ((src->vcc_flags & VCC_FLAG_LEGACY_HANDLING) != 0 &&
	    src->vcc_nent != 0) {
		/* No entries should be provided if using legacy handling */
		return (EINVAL);
	}
	for (uint_t i = 0; i < src->vcc_nent; i++) {
		/* Ensure all entries carry valid flags */
		if ((src->vcc_entries[i].vce_flags & ~VCE_FLAGS_VALID) != 0) {
			return (EINVAL);
		}
	}

	vcpu_cpuid_config_t *cfg = vm_cpuid_config(vm, vcpuid);

	/* Free any existing entries first */
	vcpu_cpuid_cleanup(cfg);

	/* Copy supplied entries into freshly allocated space */
	if (src->vcc_nent != 0) {
		const size_t entries_sz =
		    src->vcc_nent * sizeof (struct vcpu_cpuid_entry);

		cfg->vcc_nent = src->vcc_nent;
		cfg->vcc_entries = kmem_alloc(entries_sz, KM_SLEEP);
		bcopy(src->vcc_entries, cfg->vcc_entries, entries_sz);
	}
	cfg->vcc_flags = src->vcc_flags;

	return (0);
}

void
vcpu_cpuid_init(vcpu_cpuid_config_t *cfg)
{
	/* Default to legacy-style handling */
	cfg->vcc_flags = VCC_FLAG_LEGACY_HANDLING;
	cfg->vcc_nent = 0;
	cfg->vcc_entries = NULL;
}

void
vcpu_cpuid_cleanup(vcpu_cpuid_config_t *cfg)
{
	if (cfg->vcc_nent != 0) {
		ASSERT3P(cfg->vcc_entries, !=, NULL);

		kmem_free(cfg->vcc_entries,
		    cfg->vcc_nent * sizeof (struct vcpu_cpuid_entry));

		cfg->vcc_nent = 0;
		cfg->vcc_entries = NULL;
	}
}

static const char bhyve_id[12] = "bhyve bhyve ";

/*
 * Force exposition of the invariant TSC capability, regardless of whether the
 * host CPU reports having it.
 */
static int vmm_force_invariant_tsc = 0;

#define	CPUID_0000_0000	(0x0)
#define	CPUID_0000_0001	(0x1)
#define	CPUID_0000_0002	(0x2)
#define	CPUID_0000_0003	(0x3)
#define	CPUID_0000_0004	(0x4)
#define	CPUID_0000_0006	(0x6)
#define	CPUID_0000_0007	(0x7)
#define	CPUID_0000_000A	(0xA)
#define	CPUID_0000_000B	(0xB)
#define	CPUID_0000_000D	(0xD)
#define	CPUID_0000_000F	(0xF)
#define	CPUID_0000_0010	(0x10)
#define	CPUID_0000_0015	(0x15)
#define	CPUID_8000_0000	(0x80000000)
#define	CPUID_8000_0001	(0x80000001)
#define	CPUID_8000_0002	(0x80000002)
#define	CPUID_8000_0003	(0x80000003)
#define	CPUID_8000_0004	(0x80000004)
#define	CPUID_8000_0006	(0x80000006)
#define	CPUID_8000_0007	(0x80000007)
#define	CPUID_8000_0008	(0x80000008)
#define	CPUID_8000_001D	(0x8000001D)
#define	CPUID_8000_001E	(0x8000001E)

#define	CPUID_VM_HIGH	0x40000000

/*
 * CPUID instruction Fn0000_0001:
 */
#define	CPUID_0000_0001_APICID_SHIFT	24


/*
 * Round up to the next power of two, if necessary, and then take log2.
 * Returns -1 if argument is zero.
 */
static __inline int
log2(uint_t x)
{
	return (fls(x << (1 - powerof2(x))) - 1);
}

/*
 * The "legacy" bhyve cpuid emulation, which largly applies statically defined
 * masks to the data provided by the host CPU.
 */
void
legacy_emulate_cpuid(struct vm *vm, int vcpu_id, uint32_t *eax, uint32_t *ebx,
    uint32_t *ecx, uint32_t *edx)
{
	const struct xsave_limits *limits;
	uint64_t cr4;
	int error, enable_invpcid, level, width = 0, x2apic_id = 0;
	unsigned int func, regs[4], logical_cpus = 0, param;
	enum x2apic_state x2apic_state;
	uint16_t cores, maxcpus, sockets, threads;

	/*
	 * The function of CPUID is controlled through the provided value of
	 * %eax (and secondarily %ecx, for certain leaf data).
	 */
	func = (uint32_t)*eax;
	param = (uint32_t)*ecx;

	/*
	 * Requests for invalid CPUID levels should map to the highest
	 * available level instead.
	 */
	if (cpu_exthigh != 0 && func >= 0x80000000) {
		if (func > cpu_exthigh)
			func = cpu_exthigh;
	} else if (func >= 0x40000000) {
		if (func > CPUID_VM_HIGH)
			func = CPUID_VM_HIGH;
	} else if (func > cpu_high) {
		func = cpu_high;
	}

	/*
	 * In general the approach used for CPU topology is to
	 * advertise a flat topology where all CPUs are packages with
	 * no multi-core or SMT.
	 */
	switch (func) {
		/*
		 * Pass these through to the guest
		 */
		case CPUID_0000_0000:
		case CPUID_0000_0002:
		case CPUID_0000_0003:
		case CPUID_8000_0000:
		case CPUID_8000_0002:
		case CPUID_8000_0003:
		case CPUID_8000_0004:
		case CPUID_8000_0006:
			cpuid_count(func, param, regs);
			break;
		case CPUID_8000_0008:
			cpuid_count(func, param, regs);
			if (vmm_is_svm()) {
				/*
				 * As on Intel (0000_0007:0, EDX), mask out
				 * unsupported or unsafe AMD extended features
				 * (8000_0008 EBX).
				 */
				regs[1] &= (AMDFEID_CLZERO | AMDFEID_IRPERF |
				    AMDFEID_XSAVEERPTR);

				vm_get_topology(vm, &sockets, &cores, &threads,
				    &maxcpus);
				/*
				 * Here, width is ApicIdCoreIdSize, present on
				 * at least Family 15h and newer.  It
				 * represents the "number of bits in the
				 * initial apicid that indicate thread id
				 * within a package."
				 *
				 * Our topo_probe_amd() uses it for
				 * pkg_id_shift and other OSes may rely on it.
				 */
				width = MIN(0xF, log2(threads * cores));
				if (width < 0x4)
					width = 0;
				logical_cpus = MIN(0xFF, threads * cores - 1);
				regs[2] = (width << AMDID_COREID_SIZE_SHIFT) |
				    logical_cpus;
			}
			break;

		case CPUID_8000_0001:
			cpuid_count(func, param, regs);

			/*
			 * Hide SVM from guest.
			 */
			regs[2] &= ~AMDID2_SVM;

			/*
			 * Don't advertise extended performance counter MSRs
			 * to the guest.
			 */
			regs[2] &= ~AMDID2_PCXC;
			regs[2] &= ~AMDID2_PNXC;
			regs[2] &= ~AMDID2_PTSCEL2I;

			/*
			 * Don't advertise Instruction Based Sampling feature.
			 */
			regs[2] &= ~AMDID2_IBS;

			/* NodeID MSR not available */
			regs[2] &= ~AMDID2_NODE_ID;

			/* Don't advertise the OS visible workaround feature */
			regs[2] &= ~AMDID2_OSVW;

			/* Hide mwaitx/monitorx capability from the guest */
			regs[2] &= ~AMDID2_MWAITX;

#ifndef __FreeBSD__
			/*
			 * Detection routines for TCE and FFXSR are missing
			 * from our vm_cpuid_capability() detection logic
			 * today.  Mask them out until that is remedied.
			 * They do not appear to be in common usage, so their
			 * absence should not cause undue trouble.
			 */
			regs[2] &= ~AMDID2_TCE;
			regs[3] &= ~AMDID_FFXSR;
#endif

			/*
			 * Hide rdtscp/ia32_tsc_aux until we know how
			 * to deal with them.
			 */
			regs[3] &= ~AMDID_RDTSCP;
			break;

		case CPUID_8000_0007:
			cpuid_count(func, param, regs);
			/*
			 * AMD uses this leaf to advertise the processor's
			 * power monitoring and RAS capabilities. These
			 * features are hardware-specific and exposing
			 * them to a guest doesn't make a lot of sense.
			 *
			 * Intel uses this leaf only to advertise the
			 * "Invariant TSC" feature with all other bits
			 * being reserved (set to zero).
			 */
			regs[0] = 0;
			regs[1] = 0;
			regs[2] = 0;

			/*
			 * If the host system possesses an invariant TSC, then
			 * it is safe to expose to the guest.
			 *
			 * If there is measured skew between host TSCs, it will
			 * be properly offset so guests do not observe any
			 * change between CPU migrations.
			 */
			regs[3] &= AMDPM_TSC_INVARIANT;

			/*
			 * Since illumos avoids deep C-states on CPUs which do
			 * not support an invariant TSC, it may be safe (and
			 * desired) to unconditionally expose that capability to
			 * the guest.
			 */
			if (vmm_force_invariant_tsc != 0) {
				regs[3] |= AMDPM_TSC_INVARIANT;
			}
			break;

		case CPUID_8000_001D:
			/* AMD Cache topology, like 0000_0004 for Intel. */
			if (!vmm_is_svm())
				goto default_leaf;

			/*
			 * Similar to Intel, generate a ficticious cache
			 * topology for the guest with L3 shared by the
			 * package, and L1 and L2 local to a core.
			 */
			vm_get_topology(vm, &sockets, &cores, &threads,
			    &maxcpus);
			switch (param) {
			case 0:
				logical_cpus = threads;
				level = 1;
				func = 1;	/* data cache */
				break;
			case 1:
				logical_cpus = threads;
				level = 2;
				func = 3;	/* unified cache */
				break;
			case 2:
				logical_cpus = threads * cores;
				level = 3;
				func = 3;	/* unified cache */
				break;
			default:
				logical_cpus = 0;
				level = 0;
				func = 0;
				break;
			}

			logical_cpus = MIN(0xfff, logical_cpus - 1);
			regs[0] = (logical_cpus << 14) | (1 << 8) |
			    (level << 5) | func;
			regs[1] = (func > 0) ? (CACHE_LINE_SIZE - 1) : 0;
			regs[2] = 0;
			regs[3] = 0;
			break;

		case CPUID_8000_001E:
			/*
			 * AMD Family 16h+ and Hygon Family 18h additional
			 * identifiers.
			 */
			if (!vmm_is_svm() || CPUID_TO_FAMILY(cpu_id) < 0x16)
				goto default_leaf;

			vm_get_topology(vm, &sockets, &cores, &threads,
			    &maxcpus);
			regs[0] = vcpu_id;
			threads = MIN(0xFF, threads - 1);
			regs[1] = (threads << 8) |
			    (vcpu_id >> log2(threads + 1));
			/*
			 * XXX Bhyve topology cannot yet represent >1 node per
			 * processor.
			 */
			regs[2] = 0;
			regs[3] = 0;
			break;

		case CPUID_0000_0001:
			do_cpuid(1, regs);

			error = vm_get_x2apic_state(vm, vcpu_id, &x2apic_state);
			VERIFY0(error);

			/*
			 * Override the APIC ID only in ebx
			 */
			regs[1] &= ~(CPUID_LOCAL_APIC_ID);
			regs[1] |= (vcpu_id << CPUID_0000_0001_APICID_SHIFT);

			/*
			 * Don't expose VMX, SpeedStep, TME or SMX capability.
			 * Advertise x2APIC capability and Hypervisor guest.
			 */
			regs[2] &= ~(CPUID2_VMX | CPUID2_EST | CPUID2_TM2);
			regs[2] &= ~(CPUID2_SMX);

			regs[2] |= CPUID2_HV;

			if (x2apic_state != X2APIC_DISABLED)
				regs[2] |= CPUID2_X2APIC;
			else
				regs[2] &= ~CPUID2_X2APIC;

			/*
			 * Only advertise CPUID2_XSAVE in the guest if
			 * the host is using XSAVE.
			 */
			if (!(regs[2] & CPUID2_OSXSAVE))
				regs[2] &= ~CPUID2_XSAVE;

			/*
			 * If CPUID2_XSAVE is being advertised and the
			 * guest has set CR4_XSAVE, set
			 * CPUID2_OSXSAVE.
			 */
			regs[2] &= ~CPUID2_OSXSAVE;
			if (regs[2] & CPUID2_XSAVE) {
				error = vm_get_register(vm, vcpu_id,
				    VM_REG_GUEST_CR4, &cr4);
				VERIFY0(error);
				if (cr4 & CR4_XSAVE)
					regs[2] |= CPUID2_OSXSAVE;
			}

			/*
			 * Hide monitor/mwait until we know how to deal with
			 * these instructions.
			 */
			regs[2] &= ~CPUID2_MON;

			/*
			 * Hide the performance and debug features.
			 */
			regs[2] &= ~CPUID2_PDCM;

			/*
			 * No TSC deadline support in the APIC yet
			 */
			regs[2] &= ~CPUID2_TSCDLT;

			/*
			 * Hide thermal monitoring
			 */
			regs[3] &= ~(CPUID_ACPI | CPUID_TM);

			/*
			 * Hide the debug store capability.
			 */
			regs[3] &= ~CPUID_DS;

			/*
			 * Advertise the Machine Check and MTRR capability.
			 *
			 * Some guest OSes (e.g. Windows) will not boot if
			 * these features are absent.
			 */
			regs[3] |= (CPUID_MCA | CPUID_MCE | CPUID_MTRR);

			vm_get_topology(vm, &sockets, &cores, &threads,
			    &maxcpus);
			logical_cpus = threads * cores;
			regs[1] &= ~CPUID_HTT_CORES;
			regs[1] |= (logical_cpus & 0xff) << 16;
			regs[3] |= CPUID_HTT;
			break;

		case CPUID_0000_0004:
			cpuid_count(func, param, regs);

			if (regs[0] || regs[1] || regs[2] || regs[3]) {
				vm_get_topology(vm, &sockets, &cores, &threads,
				    &maxcpus);
				regs[0] &= 0x3ff;
				regs[0] |= (cores - 1) << 26;
				/*
				 * Cache topology:
				 * - L1 and L2 are shared only by the logical
				 *   processors in a single core.
				 * - L3 and above are shared by all logical
				 *   processors in the package.
				 */
				logical_cpus = threads;
				level = (regs[0] >> 5) & 0x7;
				if (level >= 3)
					logical_cpus *= cores;
				regs[0] |= (logical_cpus - 1) << 14;
			}
			break;

		case CPUID_0000_0007:
			regs[0] = 0;
			regs[1] = 0;
			regs[2] = 0;
			regs[3] = 0;

			/* leaf 0 */
			if (param == 0) {
				cpuid_count(func, param, regs);

				/* Only leaf 0 is supported */
				regs[0] = 0;

				/*
				 * Expose known-safe features.
				 */
				regs[1] &= (CPUID_STDEXT_FSGSBASE |
				    CPUID_STDEXT_BMI1 | CPUID_STDEXT_HLE |
				    CPUID_STDEXT_AVX2 | CPUID_STDEXT_SMEP |
				    CPUID_STDEXT_BMI2 |
				    CPUID_STDEXT_ERMS | CPUID_STDEXT_RTM |
				    CPUID_STDEXT_AVX512F |
				    CPUID_STDEXT_RDSEED |
				    CPUID_STDEXT_SMAP |
				    CPUID_STDEXT_AVX512PF |
				    CPUID_STDEXT_AVX512ER |
				    CPUID_STDEXT_AVX512CD | CPUID_STDEXT_SHA);
				regs[2] = 0;
				regs[3] &= CPUID_STDEXT3_MD_CLEAR;

				/* Advertise INVPCID if it is enabled. */
				error = vm_get_capability(vm, vcpu_id,
				    VM_CAP_ENABLE_INVPCID, &enable_invpcid);
				if (error == 0 && enable_invpcid)
					regs[1] |= CPUID_STDEXT_INVPCID;
			}
			break;

		case CPUID_0000_0006:
			regs[0] = CPUTPM1_ARAT;
			regs[1] = 0;
			regs[2] = 0;
			regs[3] = 0;
			break;

		case CPUID_0000_000A:
			/*
			 * Handle the access, but report 0 for
			 * all options
			 */
			regs[0] = 0;
			regs[1] = 0;
			regs[2] = 0;
			regs[3] = 0;
			break;

		case CPUID_0000_000B:
			/*
			 * Intel processor topology enumeration
			 */
			if (vmm_is_intel()) {
				vm_get_topology(vm, &sockets, &cores, &threads,
				    &maxcpus);
				if (param == 0) {
					logical_cpus = threads;
					width = log2(logical_cpus);
					level = CPUID_TYPE_SMT;
					x2apic_id = vcpu_id;
				}

				if (param == 1) {
					logical_cpus = threads * cores;
					width = log2(logical_cpus);
					level = CPUID_TYPE_CORE;
					x2apic_id = vcpu_id;
				}

				if (param >= 2) {
					width = 0;
					logical_cpus = 0;
					level = 0;
					x2apic_id = 0;
				}

				regs[0] = width & 0x1f;
				regs[1] = logical_cpus & 0xffff;
				regs[2] = (level << 8) | (param & 0xff);
				regs[3] = x2apic_id;
			} else {
				regs[0] = 0;
				regs[1] = 0;
				regs[2] = 0;
				regs[3] = 0;
			}
			break;

		case CPUID_0000_000D:
			limits = vmm_get_xsave_limits();
			if (!limits->xsave_enabled) {
				regs[0] = 0;
				regs[1] = 0;
				regs[2] = 0;
				regs[3] = 0;
				break;
			}

			cpuid_count(func, param, regs);
			switch (param) {
			case 0:
				/*
				 * Only permit the guest to use bits
				 * that are active in the host in
				 * %xcr0.  Also, claim that the
				 * maximum save area size is
				 * equivalent to the host's current
				 * save area size.  Since this runs
				 * "inside" of vmrun(), it runs with
				 * the guest's xcr0, so the current
				 * save area size is correct as-is.
				 */
				regs[0] &= limits->xcr0_allowed;
				regs[2] = limits->xsave_max_size;
				regs[3] &= (limits->xcr0_allowed >> 32);
				break;
			case 1:
				/* Only permit XSAVEOPT. */
				regs[0] &= CPUID_EXTSTATE_XSAVEOPT;
				regs[1] = 0;
				regs[2] = 0;
				regs[3] = 0;
				break;
			default:
				/*
				 * If the leaf is for a permitted feature,
				 * pass through as-is, otherwise return
				 * all zeroes.
				 */
				if (!(limits->xcr0_allowed & (1ul << param))) {
					regs[0] = 0;
					regs[1] = 0;
					regs[2] = 0;
					regs[3] = 0;
				}
				break;
			}
			break;

		case CPUID_0000_000F:
		case CPUID_0000_0010:
			/*
			 * Do not report any Resource Director Technology
			 * capabilities.  Exposing control of cache or memory
			 * controller resource partitioning to the guest is not
			 * at all sensible.
			 *
			 * This is already hidden at a high level by masking of
			 * leaf 0x7.  Even still, a guest may look here for
			 * detailed capability information.
			 */
			regs[0] = 0;
			regs[1] = 0;
			regs[2] = 0;
			regs[3] = 0;
			break;

		case CPUID_0000_0015:
			/*
			 * Don't report CPU TSC/Crystal ratio and clock
			 * values since guests may use these to derive the
			 * local APIC frequency..
			 */
			regs[0] = 0;
			regs[1] = 0;
			regs[2] = 0;
			regs[3] = 0;
			break;

		case 0x40000000:
			regs[0] = CPUID_VM_HIGH;
			bcopy(bhyve_id, &regs[1], 4);
			bcopy(bhyve_id + 4, &regs[2], 4);
			bcopy(bhyve_id + 8, &regs[3], 4);
			break;

		default:
default_leaf:
			/*
			 * The leaf value has already been clamped so
			 * simply pass this through.
			 */
			cpuid_count(func, param, regs);
			break;
	}

	*eax = regs[0];
	*ebx = regs[1];
	*ecx = regs[2];
	*edx = regs[3];
}
