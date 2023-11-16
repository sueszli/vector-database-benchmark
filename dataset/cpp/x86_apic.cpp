/*
 * Copyright 2008, Dustin Howett, dustin.howett@gmail.com. All rights reserved.
 * Copyright 2002-2005, Axel Dörfler, axeld@pinc-software.de. All rights reserved.
 * Distributed under the terms of the MIT License.
 *
 * Copyright 2001-2002, Travis Geiselbrecht. All rights reserved.
 * Distributed under the terms of the NewOS License.
 */

#include <timer.h>
#include <arch/x86/timer.h>

#include <int.h>
#include <arch/x86/apic.h>

#include <arch/cpu.h>

#include "apic_timer.h"


//#define TRACE_APIC
#ifdef TRACE_APIC
#	define TRACE(x...) dprintf("apic: " x)
#else
#	define TRACE(x...) ;
#endif


/* Method Prototypes */
static int apic_timer_get_priority();
static status_t apic_timer_set_hardware_timer(bigtime_t relativeTimeout);
static status_t apic_timer_clear_hardware_timer();
static status_t apic_timer_init(struct kernel_args *args);

static uint32 sApicTicsPerSec = 0;

struct timer_info gAPICTimer = {
	"APIC",
	&apic_timer_get_priority,
	&apic_timer_set_hardware_timer,
	&apic_timer_clear_hardware_timer,
	&apic_timer_init
};


static int
apic_timer_get_priority()
{
	return 3;
}


static int32
apic_timer_interrupt(void *data)
{
	return timer_interrupt();
}


#define MIN_TIMEOUT 1

static status_t
apic_timer_set_hardware_timer(bigtime_t relativeTimeout)
{
	if (relativeTimeout < MIN_TIMEOUT)
		relativeTimeout = MIN_TIMEOUT;

	// calculation should be ok, since it's going to be 64-bit
	uint32 ticks = ((relativeTimeout * sApicTicsPerSec) / 1000000);

	cpu_status state = disable_interrupts();

	uint32 config = apic_lvt_timer() | APIC_LVT_MASKED; // mask the timer
	apic_set_lvt_timer(config);

	apic_set_lvt_initial_timer_count(0); // zero out the timer

	config = apic_lvt_timer() & ~APIC_LVT_MASKED; // unmask the timer
	apic_set_lvt_timer(config);

	TRACE("arch_smp_set_apic_timer: config 0x%" B_PRIx32 ", timeout %" B_PRIdBIGTIME
		", tics/sec %" B_PRIu32 ", tics %" B_PRId32 "\n", config, relativeTimeout,
		sApicTicsPerSec, ticks);

	apic_set_lvt_initial_timer_count(ticks); // start it up

	restore_interrupts(state);

	return B_OK;
}


static status_t
apic_timer_clear_hardware_timer()
{
	cpu_status state = disable_interrupts();

	uint32 config = apic_lvt_timer() | APIC_LVT_MASKED;
		// mask the timer
	apic_set_lvt_timer(config);

	apic_set_lvt_initial_timer_count(0); // zero out the timer

	restore_interrupts(state);
	return B_OK;
}


static status_t
apic_timer_init(struct kernel_args *args)
{
	if (!apic_available())
		return B_ERROR;

	sApicTicsPerSec = args->arch_args.apic_time_cv_factor;

	reserve_io_interrupt_vectors(1, 0xfb - ARCH_INTERRUPT_BASE,
		INTERRUPT_TYPE_LOCAL_IRQ);
	install_io_interrupt_handler(0xfb - ARCH_INTERRUPT_BASE,
		&apic_timer_interrupt, NULL, B_NO_LOCK_VECTOR);

	return B_OK;
}


status_t
apic_timer_per_cpu_init(struct kernel_args *args, int32 cpu)
{
	/* setup timer */
	uint32 config = apic_lvt_timer() & APIC_LVT_TIMER_MASK;
	config |= 0xfb | APIC_LVT_MASKED; // vector 0xfb, timer masked
	apic_set_lvt_timer(config);

	apic_set_lvt_initial_timer_count(0); // zero out the clock

	config = apic_lvt_timer_divide_config() & 0xfffffff0;
	config |= APIC_TIMER_DIVIDE_CONFIG_1; // clock division by 1
	apic_set_lvt_timer_divide_config(config);
	return B_OK;
}
