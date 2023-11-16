/*
 * CDDL HEADER START
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can obtain a copy of the license at usr/src/OPENSOLARIS.LICENSE
 * or http://www.opensolaris.org/os/licensing.
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL HEADER in each
 * file and include the License file at usr/src/OPENSOLARIS.LICENSE.
 * If applicable, add the following below this CDDL HEADER, with the
 * fields enclosed by brackets "[]" replaced with your own identifying
 * information: Portions Copyright [yyyy] [name of copyright owner]
 *
 * CDDL HEADER END
 */

/*
 * Copyright 2021 Oxide Computer Company
 * Copyright 2021 OmniOS Community Edition (OmniOSce) Association.
 */

/*
 * Copyright 2009 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/* Copyright (c) 1984, 1986, 1987, 1988, 1989 AT&T */
/* All Rights Reserved */

/*
 * University Copyright- Copyright (c) 1982, 1986, 1988
 * The Regents of the University of California
 * All Rights Reserved
 *
 * University Acknowledgment- Portions of this document are derived from
 * software developed by the University of California, Berkeley, and its
 * contributors.
 */

#include <sys/types.h>
#include <sys/t_lock.h>
#include <sys/param.h>
#include <sys/buf.h>
#include <sys/uio.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/mman.h>
#include <sys/cred.h>
#include <sys/vnode.h>
#include <sys/vm.h>
#include <sys/vmparam.h>
#include <sys/vtrace.h>
#include <sys/cmn_err.h>
#include <sys/cpuvar.h>
#include <sys/user.h>
#include <sys/kmem.h>
#include <sys/debug.h>
#include <sys/callb.h>
#include <sys/mem_cage.h>
#include <sys/time.h>
#include <sys/stdbool.h>

#include <vm/hat.h>
#include <vm/as.h>
#include <vm/seg.h>
#include <vm/page.h>
#include <vm/pvn.h>
#include <vm/seg_kmem.h>

/*
 * FREE MEMORY MANAGEMENT
 *
 * Management of the pool of free pages is a tricky business.  There are
 * several critical threshold values which constrain our allocation of new
 * pages and inform the rate of paging out of memory to swap.  These threshold
 * values, and the behaviour they induce, are described below in descending
 * order of size -- and thus increasing order of severity!
 *
 *   +---------------------------------------------------- physmem (all memory)
 *   |
 *   | Ordinarily there are no particular constraints placed on page
 *   v allocation.  The page scanner is not running and page_create_va()
 *   | will effectively grant all page requests (whether from the kernel
 *   | or from user processes) without artificial delay.
 *   |
 *   +------------------------ lotsfree (1.56% of physmem, min. 16MB, max. 2GB)
 *   |
 *   | When we have less than "lotsfree" pages, pageout_scanner() is
 *   v signalled by schedpaging() to begin looking for pages that can
 *   | be evicted to disk to bring us back above lotsfree.  At this
 *   | stage there is still no constraint on allocation of free pages.
 *   |
 *   | For small systems, we set a lower bound of 16MB for lotsfree;
 *   v this is the natural value for a system with 1GB memory.  This is
 *   | to ensure that the pageout reserve pool contains at least 4MB
 *   | for use by ZFS.
 *   |
 *   | For systems with a large amount of memory, we constrain lotsfree
 *   | to be at most 2GB (with a pageout reserve of around 0.5GB), as
 *   v at some point the required slack relates more closely to the
 *   | rate at which paging can occur than to the total amount of memory.
 *   |
 *   +------------------- desfree (1/2 of lotsfree, 0.78% of physmem, min. 8MB)
 *   |
 *   | When we drop below desfree, a number of kernel facilities will
 *   v wait before allocating more memory, under the assumption that
 *   | pageout or reaping will make progress and free up some memory.
 *   | This behaviour is not especially coordinated; look for comparisons
 *   | of desfree and freemem.
 *   |
 *   | In addition to various attempts at advisory caution, clock()
 *   | will wake up the thread that is ordinarily parked in sched().
 *   | This routine is responsible for the heavy-handed swapping out
 *   v of entire processes in an attempt to arrest the slide of free
 *   | memory.  See comments in sched.c for more details.
 *   |
 *   +----- minfree & throttlefree (3/4 of desfree, 0.59% of physmem, min. 6MB)
 *   |
 *   | These two separate tunables have, by default, the same value.
 *   v Various parts of the kernel use minfree to signal the need for
 *   | more aggressive reclamation of memory, and sched() is more
 *   | aggressive at swapping processes out.
 *   |
 *   | If free memory falls below throttlefree, page_create_va() will
 *   | use page_create_throttle() to begin holding most requests for
 *   | new pages while pageout and reaping free up memory.  Sleeping
 *   v allocations (e.g., KM_SLEEP) are held here while we wait for
 *   | more memory.  Non-sleeping allocations are generally allowed to
 *   | proceed, unless their priority is explicitly lowered with
 *   | KM_NORMALPRI (Note: KM_NOSLEEP_LAZY == (KM_NOSLEEP | KM_NORMALPRI).).
 *   |
 *   +------- pageout_reserve (3/4 of throttlefree, 0.44% of physmem, min. 4MB)
 *   |
 *   | When we hit throttlefree, the situation is already dire.  The
 *   v system is generally paging out memory and swapping out entire
 *   | processes in order to free up memory for continued operation.
 *   |
 *   | Unfortunately, evicting memory to disk generally requires short
 *   | term use of additional memory; e.g., allocation of buffers for
 *   | storage drivers, updating maps of free and used blocks, etc.
 *   | As such, pageout_reserve is the number of pages that we keep in
 *   | special reserve for use by pageout() and sched() and by any
 *   v other parts of the kernel that need to be working for those to
 *   | make forward progress such as the ZFS I/O pipeline.
 *   |
 *   | When we are below pageout_reserve, we fail or hold any allocation
 *   | that has not explicitly requested access to the reserve pool.
 *   | Access to the reserve is generally granted via the KM_PUSHPAGE
 *   | flag, or by marking a thread T_PUSHPAGE such that all allocations
 *   | can implicitly tap the reserve.  For more details, see the
 *   v NOMEMWAIT() macro, the T_PUSHPAGE thread flag, the KM_PUSHPAGE
 *   | and VM_PUSHPAGE allocation flags, and page_create_throttle().
 *   |
 *   +---------------------------------------------------------- no free memory
 *   |
 *   | If we have arrived here, things are very bad indeed.  It is
 *   v surprisingly difficult to tell if this condition is even fatal,
 *   | as enough memory may have been granted to pageout() and to the
 *   | ZFS I/O pipeline that requests for eviction that have already been
 *   | made will complete and free up memory some time soon.
 *   |
 *   | If free memory does not materialise, the system generally remains
 *   | deadlocked.  The pageout_deadman() below is run once per second
 *   | from clock(), seeking to limit the amount of time a single request
 *   v to page out can be blocked before the system panics to get a crash
 *   | dump and return to service.
 *   |
 *   +-------------------------------------------------------------------------
 */

/*
 * The following parameters control operation of the page replacement
 * algorithm.  They are initialized to 0, and then computed at boot time based
 * on the size of the system; see setupclock().  If they are patched non-zero
 * in a loaded vmunix they are left alone and may thus be changed per system
 * using "mdb -kw" on the loaded system.
 */
pgcnt_t		slowscan = 0;
pgcnt_t		fastscan = 0;

static pgcnt_t	handspreadpages = 0;

/*
 * looppages:
 *     Cached copy of the total number of pages in the system (total_pages).
 *
 * loopfraction:
 *     Divisor used to relate fastscan to looppages in setupclock().
 */
static uint_t	loopfraction = 2;
static pgcnt_t	looppages;

static uint_t	min_percent_cpu = 4;
static uint_t	max_percent_cpu = 80;
static pgcnt_t	maxfastscan = 0;
static pgcnt_t	maxslowscan = 100;

#define		MEGABYTES		(1024ULL * 1024ULL)

/*
 * pageout_threshold_style:
 *     set to 1 to use the previous default threshold size calculation;
 *     i.e., each threshold is half of the next largest value.
 */
uint_t		pageout_threshold_style = 0;

/*
 * The operator may override these tunables to request a different minimum or
 * maximum lotsfree value, or to change the divisor we use for automatic
 * sizing.
 *
 * By default, we make lotsfree 1/64th of the total memory in the machine.  The
 * minimum and maximum are specified in bytes, rather than pages; a zero value
 * means the default values (below) are used.
 */
uint_t		lotsfree_fraction = 64;
pgcnt_t		lotsfree_min = 0;
pgcnt_t		lotsfree_max = 0;

#define		LOTSFREE_MIN_DEFAULT	(16 * MEGABYTES)
#define		LOTSFREE_MAX_DEFAULT	(2048 * MEGABYTES)

/*
 * If these tunables are set to non-zero values in /etc/system, and provided
 * the value is not larger than the threshold above, the specified value will
 * be used directly without any additional calculation or adjustment.  The boot
 * time value of these overrides is preserved in the "clockinit" struct.  More
 * detail is available in the comment at the top of the file.
 */
pgcnt_t		maxpgio = 0;
pgcnt_t		minfree = 0;
pgcnt_t		desfree = 0;
pgcnt_t		lotsfree = 0;
pgcnt_t		needfree = 0;
pgcnt_t		throttlefree = 0;
pgcnt_t		pageout_reserve = 0;

pgcnt_t		deficit;
pgcnt_t		nscan;
pgcnt_t		desscan;

/*
 * Values for min_pageout_nsec, max_pageout_nsec and pageout_nsec are the
 * number of nanoseconds in each wakeup cycle that gives the equivalent of some
 * underlying %CPU duty cycle.
 *
 * min_pageout_nsec:
 *     nanoseconds/wakeup equivalent of min_percent_cpu.
 *
 * max_pageout_nsec:
 *     nanoseconds/wakeup equivalent of max_percent_cpu.
 *
 * pageout_nsec:
 *     Number of nanoseconds budgeted for each wakeup cycle.
 *     Computed each time around by schedpaging().
 *     Varies between min_pageout_nsec and max_pageout_nsec,
 *     depending on memory pressure.
 */
static hrtime_t	min_pageout_nsec;
static hrtime_t	max_pageout_nsec;
static hrtime_t	pageout_nsec;

static uint_t	reset_hands;

#define	PAGES_POLL_MASK	1023

/*
 * pageout_sample_lim:
 *     The limit on the number of samples needed to establish a value for new
 *     pageout parameters: fastscan, slowscan, pageout_new_spread, and
 *     handspreadpages.
 *
 * pageout_sample_cnt:
 *     Current sample number.  Once the sample gets large enough, set new
 *     values for handspreadpages, pageout_new_spread, fastscan and slowscan.
 *
 * pageout_sample_pages:
 *     The accumulated number of pages scanned during sampling.
 *
 * pageout_sample_etime:
 *     The accumulated nanoseconds for the sample.
 *
 * pageout_rate:
 *     Rate in pages/nanosecond, computed at the end of sampling.
 *
 * pageout_new_spread:
 *     Initially zero while the system scan rate is measured by
 *     pageout_scanner(), which then sets this value once per system boot after
 *     enough samples have been recorded (pageout_sample_cnt).  Once set, this
 *     new value is used for fastscan and handspreadpages.
 *
 * sample_start, sample_end:
 *     The hrtime at which the last pageout_scanner() sample began and ended.
 */
typedef hrtime_t hrrate_t;

static uint64_t	pageout_sample_lim = 4;
static uint64_t	pageout_sample_cnt = 0;
static pgcnt_t	pageout_sample_pages = 0;
static hrrate_t	pageout_rate = 0;
static pgcnt_t	pageout_new_spread = 0;

static hrtime_t	pageout_cycle_nsec;
static hrtime_t	sample_start, sample_end;
static hrtime_t	pageout_sample_etime = 0;

/*
 * Record number of times a pageout_scanner() wakeup cycle finished because it
 * timed out (exceeded its CPU budget), rather than because it visited
 * its budgeted number of pages.
 */
uint64_t	pageout_timeouts = 0;

#ifdef VM_STATS
static struct pageoutvmstats_str {
	ulong_t	checkpage[3];
} pageoutvmstats;
#endif /* VM_STATS */

/*
 * Threads waiting for free memory use this condition variable and lock until
 * memory becomes available.
 */
kmutex_t	memavail_lock;
kcondvar_t	memavail_cv;

typedef enum pageout_hand {
	POH_FRONT = 1,
	POH_BACK,
} pageout_hand_t;

typedef enum {
	CKP_INELIGIBLE,
	CKP_NOT_FREED,
	CKP_FREED,
} checkpage_result_t;

static checkpage_result_t checkpage(page_t *, pageout_hand_t);

static struct clockinit {
	bool ci_init;
	pgcnt_t ci_lotsfree_min;
	pgcnt_t ci_lotsfree_max;
	pgcnt_t ci_lotsfree;
	pgcnt_t ci_desfree;
	pgcnt_t ci_minfree;
	pgcnt_t ci_throttlefree;
	pgcnt_t ci_pageout_reserve;
	pgcnt_t ci_maxpgio;
	pgcnt_t ci_maxfastscan;
	pgcnt_t ci_fastscan;
	pgcnt_t ci_slowscan;
	pgcnt_t ci_handspreadpages;
} clockinit = { .ci_init = false };

static pgcnt_t
clamp(pgcnt_t value, pgcnt_t minimum, pgcnt_t maximum)
{
	if (value < minimum) {
		return (minimum);
	} else if (value > maximum) {
		return (maximum);
	} else {
		return (value);
	}
}

static pgcnt_t
tune(pgcnt_t initval, pgcnt_t initval_ceiling, pgcnt_t defval)
{
	if (initval == 0 || initval >= initval_ceiling) {
		return (defval);
	} else {
		return (initval);
	}
}

/*
 * Set up the paging constants for the clock algorithm used by
 * pageout_scanner(), and by the virtual memory system overall.  See the
 * comments at the top of this file for more information about the threshold
 * values and system responses to memory pressure.
 *
 * This routine is called once by main() at startup, after the initial size of
 * physical memory is determined.  It may be called again later if memory is
 * added to or removed from the system, or if new measurements of the page scan
 * rate become available.
 */
void
setupclock(void)
{
	pgcnt_t defval;
	bool half = (pageout_threshold_style == 1);
	bool recalc = true;

	looppages = total_pages;

	/*
	 * The operator may have provided specific values for some of the
	 * tunables via /etc/system.  On our first call, we preserve those
	 * values so that they can be used for subsequent recalculations.
	 *
	 * A value of zero for any tunable means we will use the default
	 * sizing.
	 */
	if (!clockinit.ci_init) {
		clockinit.ci_init = true;

		clockinit.ci_lotsfree_min = lotsfree_min;
		clockinit.ci_lotsfree_max = lotsfree_max;
		clockinit.ci_lotsfree = lotsfree;
		clockinit.ci_desfree = desfree;
		clockinit.ci_minfree = minfree;
		clockinit.ci_throttlefree = throttlefree;
		clockinit.ci_pageout_reserve = pageout_reserve;
		clockinit.ci_maxpgio = maxpgio;
		clockinit.ci_maxfastscan = maxfastscan;
		clockinit.ci_fastscan = fastscan;
		clockinit.ci_slowscan = slowscan;
		clockinit.ci_handspreadpages = handspreadpages;

		/*
		 * The first call does not trigger a recalculation, only
		 * subsequent calls.
		 */
		recalc = false;
	}

	/*
	 * Configure paging threshold values.  For more details on what each
	 * threshold signifies, see the comments at the top of this file.
	 */
	lotsfree_max = tune(clockinit.ci_lotsfree_max, looppages,
	    btop(LOTSFREE_MAX_DEFAULT));
	lotsfree_min = tune(clockinit.ci_lotsfree_min, lotsfree_max,
	    btop(LOTSFREE_MIN_DEFAULT));

	lotsfree = tune(clockinit.ci_lotsfree, looppages,
	    clamp(looppages / lotsfree_fraction, lotsfree_min, lotsfree_max));

	desfree = tune(clockinit.ci_desfree, lotsfree,
	    lotsfree / 2);

	minfree = tune(clockinit.ci_minfree, desfree,
	    half ? desfree / 2 : 3 * desfree / 4);

	throttlefree = tune(clockinit.ci_throttlefree, desfree,
	    minfree);

	pageout_reserve = tune(clockinit.ci_pageout_reserve, throttlefree,
	    half ? throttlefree / 2 : 3 * throttlefree / 4);

	/*
	 * Maxpgio thresholds how much paging is acceptable.
	 * This figures that 2/3 busy on an arm is all that is
	 * tolerable for paging.  We assume one operation per disk rev.
	 *
	 * XXX - Does not account for multiple swap devices.
	 */
	if (clockinit.ci_maxpgio == 0) {
		maxpgio = (DISKRPM * 2) / 3;
	} else {
		maxpgio = clockinit.ci_maxpgio;
	}

	/*
	 * The clock scan rate varies between fastscan and slowscan
	 * based on the amount of free memory available.  Fastscan
	 * rate should be set based on the number pages that can be
	 * scanned per sec using ~10% of processor time.  Since this
	 * value depends on the processor, MMU, Mhz etc., it is
	 * difficult to determine it in a generic manner for all
	 * architectures.
	 *
	 * Instead of trying to determine the number of pages scanned
	 * per sec for every processor, fastscan is set to be the smaller
	 * of 1/2 of memory or MAXHANDSPREADPAGES and the sampling
	 * time is limited to ~4% of processor time.
	 *
	 * Setting fastscan to be 1/2 of memory allows pageout to scan
	 * all of memory in ~2 secs.  This implies that user pages not
	 * accessed within 1 sec (assuming, handspreadpages == fastscan)
	 * can be reclaimed when free memory is very low.  Stealing pages
	 * not accessed within 1 sec seems reasonable and ensures that
	 * active user processes don't thrash.
	 *
	 * Smaller values of fastscan result in scanning fewer pages
	 * every second and consequently pageout may not be able to free
	 * sufficient memory to maintain the minimum threshold.  Larger
	 * values of fastscan result in scanning a lot more pages which
	 * could lead to thrashing and higher CPU usage.
	 *
	 * Fastscan needs to be limited to a maximum value and should not
	 * scale with memory to prevent pageout from consuming too much
	 * time for scanning on slow CPU's and avoid thrashing, as a
	 * result of scanning too many pages, on faster CPU's.
	 * The value of 64 Meg was chosen for MAXHANDSPREADPAGES
	 * (the upper bound for fastscan) based on the average number
	 * of pages that can potentially be scanned in ~1 sec (using ~4%
	 * of the CPU) on some of the following machines that currently
	 * run Solaris 2.x:
	 *
	 *			average memory scanned in ~1 sec
	 *
	 *	25 Mhz SS1+:		23 Meg
	 *	LX:			37 Meg
	 *	50 Mhz SC2000:		68 Meg
	 *
	 *	40 Mhz 486:		26 Meg
	 *	66 Mhz 486:		42 Meg
	 *
	 * When free memory falls just below lotsfree, the scan rate
	 * goes from 0 to slowscan (i.e., pageout starts running).  This
	 * transition needs to be smooth and is achieved by ensuring that
	 * pageout scans a small number of pages to satisfy the transient
	 * memory demand.  This is set to not exceed 100 pages/sec (25 per
	 * wakeup) since scanning that many pages has no noticible impact
	 * on system performance.
	 *
	 * In addition to setting fastscan and slowscan, pageout is
	 * limited to using ~4% of the CPU.  This results in increasing
	 * the time taken to scan all of memory, which in turn means that
	 * user processes have a better opportunity of preventing their
	 * pages from being stolen.  This has a positive effect on
	 * interactive and overall system performance when memory demand
	 * is high.
	 *
	 * Thus, the rate at which pages are scanned for replacement will
	 * vary linearly between slowscan and the number of pages that
	 * can be scanned using ~4% of processor time instead of varying
	 * linearly between slowscan and fastscan.
	 *
	 * Also, the processor time used by pageout will vary from ~1%
	 * at slowscan to ~4% at fastscan instead of varying between
	 * ~1% at slowscan and ~10% at fastscan.
	 *
	 * The values chosen for the various VM parameters (fastscan,
	 * handspreadpages, etc) are not universally true for all machines,
	 * but appear to be a good rule of thumb for the machines we've
	 * tested.  They have the following ranges:
	 *
	 *	cpu speed:	20 to 70 Mhz
	 *	page size:	4K to 8K
	 *	memory size:	16M to 5G
	 *	page scan rate:	4000 - 17400 4K pages per sec
	 *
	 * The values need to be re-examined for machines which don't
	 * fall into the various ranges (e.g., slower or faster CPUs,
	 * smaller or larger pagesizes etc) shown above.
	 *
	 * On an MP machine, pageout is often unable to maintain the
	 * minimum paging thresholds under heavy load.  This is due to
	 * the fact that user processes running on other CPU's can be
	 * dirtying memory at a much faster pace than pageout can find
	 * pages to free.  The memory demands could be met by enabling
	 * more than one CPU to run the clock algorithm in such a manner
	 * that the various clock hands don't overlap.  This also makes
	 * it more difficult to determine the values for fastscan, slowscan
	 * and handspreadpages.
	 *
	 * The swapper is currently used to free up memory when pageout
	 * is unable to meet memory demands by swapping out processes.
	 * In addition to freeing up memory, swapping also reduces the
	 * demand for memory by preventing user processes from running
	 * and thereby consuming memory.
	 */
	if (clockinit.ci_maxfastscan == 0) {
		if (pageout_new_spread != 0) {
			maxfastscan = pageout_new_spread;
		} else {
			maxfastscan = MAXHANDSPREADPAGES;
		}
	} else {
		maxfastscan = clockinit.ci_maxfastscan;
	}

	if (clockinit.ci_fastscan == 0) {
		fastscan = MIN(looppages / loopfraction, maxfastscan);
	} else {
		fastscan = clockinit.ci_fastscan;
	}

	if (fastscan > looppages / loopfraction) {
		fastscan = looppages / loopfraction;
	}

	/*
	 * Set slow scan time to 1/10 the fast scan time, but
	 * not to exceed maxslowscan.
	 */
	if (clockinit.ci_slowscan == 0) {
		slowscan = MIN(fastscan / 10, maxslowscan);
	} else {
		slowscan = clockinit.ci_slowscan;
	}

	if (slowscan > fastscan / 2) {
		slowscan = fastscan / 2;
	}

	/*
	 * Handspreadpages is distance (in pages) between front and back
	 * pageout daemon hands.  The amount of time to reclaim a page
	 * once pageout examines it increases with this distance and
	 * decreases as the scan rate rises. It must be < the amount
	 * of pageable memory.
	 *
	 * Since pageout is limited to ~4% of the CPU, setting handspreadpages
	 * to be "fastscan" results in the front hand being a few secs
	 * (varies based on the processor speed) ahead of the back hand
	 * at fastscan rates.  This distance can be further reduced, if
	 * necessary, by increasing the processor time used by pageout
	 * to be more than ~4% and preferrably not more than ~10%.
	 *
	 * As a result, user processes have a much better chance of
	 * referencing their pages before the back hand examines them.
	 * This also significantly lowers the number of reclaims from
	 * the freelist since pageout does not end up freeing pages which
	 * may be referenced a sec later.
	 */
	if (clockinit.ci_handspreadpages == 0) {
		handspreadpages = fastscan;
	} else {
		handspreadpages = clockinit.ci_handspreadpages;
	}

	/*
	 * Make sure that back hand follows front hand by at least
	 * 1/SCHEDPAGING_HZ seconds.  Without this test, it is possible for the
	 * back hand to look at a page during the same wakeup of the pageout
	 * daemon in which the front hand cleared its ref bit.
	 */
	if (handspreadpages >= looppages) {
		handspreadpages = looppages - 1;
	}

	/*
	 * If we have been called to recalculate the parameters, set a flag to
	 * re-evaluate the clock hand pointers.
	 */
	if (recalc) {
		reset_hands = 1;
	}
}

/*
 * Pageout scheduling.
 *
 * Schedpaging controls the rate at which the page out daemon runs by
 * setting the global variables nscan and desscan SCHEDPAGING_HZ
 * times a second.  Nscan records the number of pages pageout has examined
 * in its current pass; schedpaging() resets this value to zero each time
 * it runs.  Desscan records the number of pages pageout should examine
 * in its next pass; schedpaging() sets this value based on the amount of
 * currently available memory.
 */
#define	SCHEDPAGING_HZ	4

static kmutex_t	pageout_mutex;	/* held while pageout or schedpaging running */

/*
 * Pool of available async pageout putpage requests.
 */
static struct async_reqs *push_req;
static struct async_reqs *req_freelist;	/* available req structs */
static struct async_reqs *push_list;	/* pending reqs */
static kmutex_t push_lock;		/* protects req pool */
static kcondvar_t push_cv;

/*
 * If pageout() is stuck on a single push for this many seconds,
 * pageout_deadman() will assume the system has hit a memory deadlock.  If set
 * to 0, the deadman will have no effect.
 *
 * Note that we are only looking for stalls in the calls that pageout() makes
 * to VOP_PUTPAGE().  These calls are merely asynchronous requests for paging
 * I/O, which should not take long unless the underlying strategy call blocks
 * indefinitely for memory.  The actual I/O request happens (or fails) later.
 */
uint_t pageout_deadman_seconds = 90;

static uint_t pageout_stucktime = 0;
static bool pageout_pushing = false;
static uint64_t pageout_pushcount = 0;
static uint64_t pageout_pushcount_seen = 0;

static int async_list_size = 256;	/* number of async request structs */

static void pageout_scanner(void);

/*
 * If a page is being shared more than "po_share" times
 * then leave it alone- don't page it out.
 */
#define	MIN_PO_SHARE	(8)
#define	MAX_PO_SHARE	((MIN_PO_SHARE) << 24)
ulong_t	po_share = MIN_PO_SHARE;

/*
 * Schedule rate for paging.
 * Rate is linear interpolation between
 * slowscan with lotsfree and fastscan when out of memory.
 */
static void
schedpaging(void *arg)
{
	spgcnt_t vavail;

	if (freemem < lotsfree + needfree + kmem_reapahead)
		kmem_reap();

	if (freemem < lotsfree + needfree)
		seg_preap();

	if (kcage_on && (kcage_freemem < kcage_desfree || kcage_needfree))
		kcage_cageout_wakeup();

	if (mutex_tryenter(&pageout_mutex)) {
		/* pageout() not running */
		nscan = 0;
		vavail = freemem - deficit;
		if (pageout_new_spread != 0)
			vavail -= needfree;
		if (vavail < 0)
			vavail = 0;
		if (vavail > lotsfree)
			vavail = lotsfree;

		/*
		 * Fix for 1161438 (CRS SPR# 73922).  All variables
		 * in the original calculation for desscan were 32 bit signed
		 * ints.  As freemem approaches 0x0 on a system with 1 Gig or
		 * more of memory, the calculation can overflow.  When this
		 * happens, desscan becomes negative and pageout_scanner()
		 * stops paging out.
		 */
		if (needfree > 0 && pageout_new_spread == 0) {
			/*
			 * If we've not yet collected enough samples to
			 * calculate a spread, use the old logic of kicking
			 * into high gear anytime needfree is non-zero.
			 */
			desscan = fastscan / SCHEDPAGING_HZ;
		} else {
			/*
			 * Once we've calculated a spread based on system
			 * memory and usage, just treat needfree as another
			 * form of deficit.
			 */
			spgcnt_t faststmp, slowstmp, result;

			slowstmp = slowscan * vavail;
			faststmp = fastscan * (lotsfree - vavail);
			result = (slowstmp + faststmp) /
			    nz(lotsfree) / SCHEDPAGING_HZ;
			desscan = (pgcnt_t)result;
		}

		pageout_nsec = min_pageout_nsec + (lotsfree - vavail) *
		    (max_pageout_nsec - min_pageout_nsec) / nz(lotsfree);

		if (freemem < lotsfree + needfree ||
		    pageout_sample_cnt < pageout_sample_lim) {
			/*
			 * Either we need more memory, or we still need to
			 * measure the average scan rate.  Wake the scanner.
			 */
			DTRACE_PROBE(pageout__cv__signal);
			cv_signal(&proc_pageout->p_cv);
		} else {
			/*
			 * There are enough free pages, no need to
			 * kick the scanner thread.  And next time
			 * around, keep more of the `highly shared'
			 * pages.
			 */
			cv_signal_pageout();
			if (po_share > MIN_PO_SHARE) {
				po_share >>= 1;
			}
		}
		mutex_exit(&pageout_mutex);
	}

	/*
	 * Signal threads waiting for available memory.
	 * NOTE: usually we need to grab memavail_lock before cv_broadcast, but
	 * in this case it is not needed - the waiters will be waken up during
	 * the next invocation of this function.
	 */
	if (kmem_avail() > 0)
		cv_broadcast(&memavail_cv);

	(void) timeout(schedpaging, arg, hz / SCHEDPAGING_HZ);
}

pgcnt_t		pushes;
ulong_t		push_list_size;		/* # of requests on pageout queue */

/*
 * Paging out should always be enabled.  This tunable exists to hold pageout
 * for debugging purposes.  If set to 0, pageout_scanner() will go back to
 * sleep each time it is woken by schedpaging().
 */
uint_t dopageout = 1;

/*
 * The page out daemon, which runs as process 2.
 *
 * As long as there are at least lotsfree pages,
 * this process is not run.  When the number of free
 * pages stays in the range desfree to lotsfree,
 * this daemon runs through the pages in the loop
 * at a rate determined in schedpaging().  Pageout manages
 * two hands on the clock.  The front hand moves through
 * memory, clearing the reference bit,
 * and stealing pages from procs that are over maxrss.
 * The back hand travels a distance behind the front hand,
 * freeing the pages that have not been referenced in the time
 * since the front hand passed.  If modified, they are pushed to
 * swap before being freed.
 *
 * There are 2 threads that act on behalf of the pageout process.
 * One thread scans pages (pageout_scanner) and frees them up if
 * they don't require any VOP_PUTPAGE operation. If a page must be
 * written back to its backing store, the request is put on a list
 * and the other (pageout) thread is signaled. The pageout thread
 * grabs VOP_PUTPAGE requests from the list, and processes them.
 * Some filesystems may require resources for the VOP_PUTPAGE
 * operations (like memory) and hence can block the pageout
 * thread, but the scanner thread can still operate. There is still
 * no guarantee that memory deadlocks cannot occur.
 *
 * For now, this thing is in very rough form.
 */
void
pageout()
{
	struct async_reqs *arg;
	pri_t pageout_pri;
	int i;
	pgcnt_t max_pushes;
	callb_cpr_t cprinfo;

	proc_pageout = ttoproc(curthread);
	proc_pageout->p_cstime = 0;
	proc_pageout->p_stime =  0;
	proc_pageout->p_cutime =  0;
	proc_pageout->p_utime = 0;
	bcopy("pageout", PTOU(curproc)->u_psargs, 8);
	bcopy("pageout", PTOU(curproc)->u_comm, 7);

	/*
	 * Create pageout scanner thread
	 */
	mutex_init(&pageout_mutex, NULL, MUTEX_DEFAULT, NULL);
	mutex_init(&push_lock, NULL, MUTEX_DEFAULT, NULL);

	/*
	 * Allocate and initialize the async request structures
	 * for pageout.
	 */
	push_req = (struct async_reqs *)
	    kmem_zalloc(async_list_size * sizeof (struct async_reqs), KM_SLEEP);

	req_freelist = push_req;
	for (i = 0; i < async_list_size - 1; i++) {
		push_req[i].a_next = &push_req[i + 1];
	}

	pageout_pri = curthread->t_pri;

	/* Create the pageout scanner thread. */
	(void) lwp_kernel_create(proc_pageout, pageout_scanner, NULL, TS_RUN,
	    pageout_pri - 1);

	/*
	 * kick off pageout scheduler.
	 */
	schedpaging(NULL);

	/*
	 * Create kernel cage thread.
	 * The kernel cage thread is started under the pageout process
	 * to take advantage of the less restricted page allocation
	 * in page_create_throttle().
	 */
	kcage_cageout_init();

	/*
	 * Limit pushes to avoid saturating pageout devices.
	 */
	max_pushes = maxpgio / SCHEDPAGING_HZ;
	CALLB_CPR_INIT(&cprinfo, &push_lock, callb_generic_cpr, "pageout");

	for (;;) {
		mutex_enter(&push_lock);

		while ((arg = push_list) == NULL || pushes > max_pushes) {
			CALLB_CPR_SAFE_BEGIN(&cprinfo);
			cv_wait(&push_cv, &push_lock);
			pushes = 0;
			CALLB_CPR_SAFE_END(&cprinfo, &push_lock);
		}
		push_list = arg->a_next;
		arg->a_next = NULL;
		pageout_pushing = true;
		mutex_exit(&push_lock);

		if (VOP_PUTPAGE(arg->a_vp, (offset_t)arg->a_off,
		    arg->a_len, arg->a_flags, arg->a_cred, NULL) == 0) {
			pushes++;
		}

		/* vp held by checkpage() */
		VN_RELE(arg->a_vp);

		mutex_enter(&push_lock);
		pageout_pushing = false;
		pageout_pushcount++;
		arg->a_next = req_freelist;	/* back on freelist */
		req_freelist = arg;
		push_list_size--;
		mutex_exit(&push_lock);
	}
}

/*
 * Kernel thread that scans pages looking for ones to free
 */
static void
pageout_scanner(void)
{
	struct page *fronthand, *backhand;
	uint_t laps;
	callb_cpr_t cprinfo;
	pgcnt_t	nscan_limit;
	pgcnt_t	pcount;
	bool sampling;

	CALLB_CPR_INIT(&cprinfo, &pageout_mutex, callb_generic_cpr, "poscan");
	mutex_enter(&pageout_mutex);

	/*
	 * The restart case does not attempt to point the hands at roughly
	 * the right point on the assumption that after one circuit things
	 * will have settled down, and restarts shouldn't be that often.
	 */

	/*
	 * Set the two clock hands to be separated by a reasonable amount,
	 * but no more than 360 degrees apart.
	 */
	backhand = page_first();
	if (handspreadpages >= total_pages) {
		fronthand = page_nextn(backhand, total_pages - 1);
	} else {
		fronthand = page_nextn(backhand, handspreadpages);
	}

	/*
	 * Establish the minimum and maximum length of time to be spent
	 * scanning pages per wakeup, limiting the scanner duty cycle.  The
	 * input percentage values (0-100) must be converted to a fraction of
	 * the number of nanoseconds in a second of wall time, then further
	 * scaled down by the number of scanner wakeups in a second:
	 */
	min_pageout_nsec = MAX(1,
	    NANOSEC * min_percent_cpu / 100 / SCHEDPAGING_HZ);
	max_pageout_nsec = MAX(min_pageout_nsec,
	    NANOSEC * max_percent_cpu / 100 / SCHEDPAGING_HZ);

loop:
	cv_signal_pageout();

	CALLB_CPR_SAFE_BEGIN(&cprinfo);
	cv_wait(&proc_pageout->p_cv, &pageout_mutex);
	CALLB_CPR_SAFE_END(&cprinfo, &pageout_mutex);

	/*
	 * Check if pageout has been disabled for debugging purposes:
	 */
	if (!dopageout) {
		goto loop;
	}

	/*
	 * One may reset the clock hands for debugging purposes.  Hands will
	 * also be reset if memory is added to or removed from the system.
	 */
	if (reset_hands) {
		reset_hands = 0;

		backhand = page_first();
		if (handspreadpages >= total_pages) {
			fronthand = page_nextn(backhand, total_pages - 1);
		} else {
			fronthand = page_nextn(backhand, handspreadpages);
		}
	}

	CPU_STATS_ADDQ(CPU, vm, pgrrun, 1);

	/*
	 * Keep track of the number of times we have scanned all the way around
	 * the loop:
	 */
	laps = 0;

	DTRACE_PROBE(pageout__start);

	/*
	 * Track the number of pages visited during this scan so that we can
	 * periodically measure our duty cycle.
	 */
	pcount = 0;

	if (pageout_sample_cnt < pageout_sample_lim) {
		/*
		 * We need to measure the rate at which the system is able to
		 * scan pages of memory.  Each of these initial samples is a
		 * scan of all system memory, regardless of whether or not we
		 * are experiencing memory pressure.
		 */
		nscan_limit = total_pages;
		sampling = true;
	} else {
		nscan_limit = desscan;
		sampling = false;
	}

	sample_start = gethrtime();

	/*
	 * Scan the appropriate number of pages for a single duty cycle.
	 */
	while (nscan < nscan_limit) {
		checkpage_result_t rvfront, rvback;

		if (!sampling && freemem >= lotsfree + needfree) {
			/*
			 * We are not sampling and enough memory has become
			 * available that scanning is no longer required.
			 */
			break;
		}

		/*
		 * Periodically check to see if we have exceeded the CPU duty
		 * cycle for a single wakeup.
		 */
		if ((pcount & PAGES_POLL_MASK) == PAGES_POLL_MASK) {
			pageout_cycle_nsec = gethrtime() - sample_start;
			if (pageout_cycle_nsec >= pageout_nsec) {
				++pageout_timeouts;
				break;
			}
		}

		/*
		 * If checkpage manages to add a page to the free list,
		 * we give ourselves another couple of trips around the loop.
		 */
		if ((rvfront = checkpage(fronthand, POH_FRONT)) == CKP_FREED) {
			laps = 0;
		}
		if ((rvback = checkpage(backhand, POH_BACK)) == CKP_FREED) {
			laps = 0;
		}

		++pcount;

		/*
		 * Protected by pageout_mutex instead of cpu_stat_lock:
		 */
		CPU_STATS_ADDQ(CPU, vm, scan, 1);

		/*
		 * Don't include ineligible pages in the number scanned.
		 */
		if (rvfront != CKP_INELIGIBLE || rvback != CKP_INELIGIBLE) {
			nscan++;
		}

		backhand = page_next(backhand);
		fronthand = page_next(fronthand);

		/*
		 * The front hand has wrapped around to the first page in the
		 * loop.
		 */
		if (fronthand == page_first()) {
			laps++;
			DTRACE_PROBE1(pageout__hand__wrap, uint_t, laps);

			/*
			 * Protected by pageout_mutex instead of cpu_stat_lock:
			 */
			CPU_STATS_ADDQ(CPU, vm, rev, 1);

			if (laps > 1) {
				/*
				 * Extremely unlikely, but it happens.
				 * We went around the loop at least once
				 * and didn't get far enough.
				 * If we are still skipping `highly shared'
				 * pages, skip fewer of them.  Otherwise,
				 * give up till the next clock tick.
				 */
				if (po_share < MAX_PO_SHARE) {
					po_share <<= 1;
				} else {
					break;
				}
			}
		}
	}

	sample_end = gethrtime();

	DTRACE_PROBE1(pageout__end, uint_t, laps);

	if (pageout_new_spread == 0) {
		if (pageout_sample_cnt < pageout_sample_lim) {
			/*
			 * Continue accumulating samples until we have enough
			 * to get a reasonable value for average scan rate:
			 */
			pageout_sample_pages += pcount;
			pageout_sample_etime += sample_end - sample_start;
			++pageout_sample_cnt;
		}

		if (pageout_sample_cnt >= pageout_sample_lim) {
			/*
			 * We have enough samples, set the spread.
			 */
			pageout_rate = (hrrate_t)pageout_sample_pages *
			    (hrrate_t)(NANOSEC) / pageout_sample_etime;
			pageout_new_spread = pageout_rate / 10;
			setupclock();
		}
	}

	goto loop;
}

/*
 * The pageout deadman is run once per second by clock().
 */
void
pageout_deadman(void)
{
	if (panicstr != NULL) {
		/*
		 * There is no pageout after panic.
		 */
		return;
	}

	if (pageout_deadman_seconds == 0) {
		/*
		 * The deadman is not enabled.
		 */
		return;
	}

	if (!pageout_pushing) {
		goto reset;
	}

	/*
	 * We are pushing a page.  Check to see if it is the same call we saw
	 * last time we looked:
	 */
	if (pageout_pushcount != pageout_pushcount_seen) {
		/*
		 * It is a different call from the last check, so we are not
		 * stuck.
		 */
		goto reset;
	}

	if (++pageout_stucktime >= pageout_deadman_seconds) {
		panic("pageout_deadman: stuck pushing the same page for %d "
		    "seconds (freemem is %lu)", pageout_deadman_seconds,
		    freemem);
	}

	return;

reset:
	/*
	 * Reset our tracking state to reflect that we are not stuck:
	 */
	pageout_stucktime = 0;
	pageout_pushcount_seen = pageout_pushcount;
}

/*
 * Look at the page at hand.  If it is locked (e.g., for physical i/o),
 * system (u., page table) or free, then leave it alone.  Otherwise,
 * if we are running the front hand, turn off the page's reference bit.
 * If the proc is over maxrss, we take it.  If running the back hand,
 * check whether the page has been reclaimed.  If not, free the page,
 * pushing it to disk first if necessary.
 *
 * Return values:
 *	CKP_INELIGIBLE if the page is not a candidate at all,
 *	CKP_NOT_FREED  if the page was not freed, or
 *	CKP_FREED      if we freed it.
 */
static checkpage_result_t
checkpage(struct page *pp, pageout_hand_t whichhand)
{
	int ppattr;
	int isfs = 0;
	int isexec = 0;
	int pagesync_flag;

	/*
	 * Skip pages:
	 *	- associated with the kernel vnode since
	 *	    they are always "exclusively" locked.
	 *	- that are free
	 *	- that are shared more than po_share'd times
	 *	- its already locked
	 *
	 * NOTE:  These optimizations assume that reads are atomic.
	 */

	if (PP_ISKAS(pp) || PAGE_LOCKED(pp) || PP_ISFREE(pp) ||
	    pp->p_lckcnt != 0 || pp->p_cowcnt != 0 ||
	    hat_page_checkshare(pp, po_share)) {
		return (CKP_INELIGIBLE);
	}

	if (!page_trylock(pp, SE_EXCL)) {
		/*
		 * Skip the page if we can't acquire the "exclusive" lock.
		 */
		return (CKP_INELIGIBLE);
	} else if (PP_ISFREE(pp)) {
		/*
		 * It became free between the above check and our actually
		 * locking the page.  Oh well, there will be other pages.
		 */
		page_unlock(pp);
		return (CKP_INELIGIBLE);
	}

	/*
	 * Reject pages that cannot be freed. The page_struct_lock
	 * need not be acquired to examine these
	 * fields since the page has an "exclusive" lock.
	 */
	if (pp->p_lckcnt != 0 || pp->p_cowcnt != 0) {
		page_unlock(pp);
		return (CKP_INELIGIBLE);
	}

	/*
	 * Maintain statistics for what we are freeing
	 */
	if (pp->p_vnode != NULL) {
		if (pp->p_vnode->v_flag & VVMEXEC)
			isexec = 1;

		if (!IS_SWAPFSVP(pp->p_vnode))
			isfs = 1;
	}

	/*
	 * Turn off REF and MOD bits with the front hand.
	 * The back hand examines the REF bit and always considers
	 * SHARED pages as referenced.
	 */
	if (whichhand == POH_FRONT) {
		pagesync_flag = HAT_SYNC_ZERORM;
	} else {
		pagesync_flag = HAT_SYNC_DONTZERO | HAT_SYNC_STOPON_REF |
		    HAT_SYNC_STOPON_SHARED;
	}

	ppattr = hat_pagesync(pp, pagesync_flag);

recheck:
	/*
	 * If page is referenced; make unreferenced but reclaimable.
	 * If this page is not referenced, then it must be reclaimable
	 * and we can add it to the free list.
	 */
	if (ppattr & P_REF) {
		DTRACE_PROBE2(pageout__isref, page_t *, pp,
		    pageout_hand_t, whichhand);

		if (whichhand == POH_FRONT) {
			/*
			 * Checking of rss or madvise flags needed here...
			 *
			 * If not "well-behaved", fall through into the code
			 * for not referenced.
			 */
			hat_clrref(pp);
		}

		/*
		 * Somebody referenced the page since the front
		 * hand went by, so it's not a candidate for
		 * freeing up.
		 */
		page_unlock(pp);
		return (CKP_NOT_FREED);
	}

	VM_STAT_ADD(pageoutvmstats.checkpage[0]);

	/*
	 * If large page, attempt to demote it. If successfully demoted,
	 * retry the checkpage.
	 */
	if (pp->p_szc != 0) {
		if (!page_try_demote_pages(pp)) {
			VM_STAT_ADD(pageoutvmstats.checkpage[1]);
			page_unlock(pp);
			return (CKP_INELIGIBLE);
		}

		ASSERT(pp->p_szc == 0);
		VM_STAT_ADD(pageoutvmstats.checkpage[2]);

		/*
		 * Since page_try_demote_pages() could have unloaded some
		 * mappings it makes sense to reload ppattr.
		 */
		ppattr = hat_page_getattr(pp, P_MOD | P_REF);
	}

	/*
	 * If the page is currently dirty, we have to arrange to have it
	 * cleaned before it can be freed.
	 *
	 * XXX - ASSERT(pp->p_vnode != NULL);
	 */
	if ((ppattr & P_MOD) && pp->p_vnode != NULL) {
		struct vnode *vp = pp->p_vnode;
		u_offset_t offset = pp->p_offset;

		/*
		 * XXX - Test for process being swapped out or about to exit?
		 * [Can't get back to process(es) using the page.]
		 */

		/*
		 * Hold the vnode before releasing the page lock to
		 * prevent it from being freed and re-used by some
		 * other thread.
		 */
		VN_HOLD(vp);
		page_unlock(pp);

		/*
		 * Queue I/O request for the pageout thread.
		 */
		if (!queue_io_request(vp, offset)) {
			VN_RELE(vp);
			return (CKP_NOT_FREED);
		}
		return (CKP_FREED);
	}

	/*
	 * Now we unload all the translations and put the page back on to the
	 * free list.  If the page was used (referenced or modified) after the
	 * pagesync but before it was unloaded we catch it and handle the page
	 * properly.
	 */
	DTRACE_PROBE2(pageout__free, page_t *, pp, pageout_hand_t, whichhand);
	(void) hat_pageunload(pp, HAT_FORCE_PGUNLOAD);
	ppattr = hat_page_getattr(pp, P_MOD | P_REF);
	if ((ppattr & P_REF) || ((ppattr & P_MOD) && pp->p_vnode != NULL)) {
		goto recheck;
	}

	VN_DISPOSE(pp, B_FREE, 0, kcred);

	CPU_STATS_ADD_K(vm, dfree, 1);

	if (isfs) {
		if (isexec) {
			CPU_STATS_ADD_K(vm, execfree, 1);
		} else {
			CPU_STATS_ADD_K(vm, fsfree, 1);
		}
	} else {
		CPU_STATS_ADD_K(vm, anonfree, 1);
	}

	return (CKP_FREED);
}

/*
 * Queue async i/o request from pageout_scanner and segment swapout
 * routines on one common list.  This ensures that pageout devices (swap)
 * are not saturated by pageout_scanner or swapout requests.
 * The pageout thread empties this list by initiating i/o operations.
 */
int
queue_io_request(vnode_t *vp, u_offset_t off)
{
	struct async_reqs *arg;

	/*
	 * If we cannot allocate an async request struct,
	 * skip this page.
	 */
	mutex_enter(&push_lock);
	if ((arg = req_freelist) == NULL) {
		mutex_exit(&push_lock);
		return (0);
	}
	req_freelist = arg->a_next;		/* adjust freelist */
	push_list_size++;

	arg->a_vp = vp;
	arg->a_off = off;
	arg->a_len = PAGESIZE;
	arg->a_flags = B_ASYNC | B_FREE;
	arg->a_cred = kcred;		/* always held */

	/*
	 * Add to list of pending write requests.
	 */
	arg->a_next = push_list;
	push_list = arg;

	if (req_freelist == NULL) {
		/*
		 * No free async requests left. The lock is held so we
		 * might as well signal the pusher thread now.
		 */
		cv_signal(&push_cv);
	}
	mutex_exit(&push_lock);
	return (1);
}

/*
 * Wakeup pageout to initiate i/o if push_list is not empty.
 */
void
cv_signal_pageout()
{
	if (push_list != NULL) {
		mutex_enter(&push_lock);
		cv_signal(&push_cv);
		mutex_exit(&push_lock);
	}
}
