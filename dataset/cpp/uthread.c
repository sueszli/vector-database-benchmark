/* Copyright (c) 2011-2014 The Regents of the University of California
 * Barret Rhoden <brho@cs.berkeley.edu>
 * See LICENSE for details. */

#include <ros/arch/membar.h>
#include <parlib/arch/atomic.h>
#include <parlib/parlib.h>
#include <parlib/vcore.h>
#include <parlib/uthread.h>
#include <parlib/event.h>
#include <stdlib.h>
#include <parlib/assert.h>
#include <parlib/stdio.h>
#include <parlib/arch/trap.h>
#include <parlib/ros_debug.h>

__thread struct uthread *current_uthread = 0;
/* ev_q for all preempt messages (handled here to keep 2LSs from worrying
 * extensively about the details.  Will call out when necessary. */
static struct event_queue *preempt_ev_q;

/* Helpers: */
#define UTH_TLSDESC_NOTLS (void*)(-1)
static inline bool __uthread_has_tls(struct uthread *uthread);
static int __uthread_allocate_tls(struct uthread *uthread);
static int __uthread_reinit_tls(struct uthread *uthread);
static void __uthread_free_tls(struct uthread *uthread);
static void __run_current_uthread_raw(void);

static void handle_vc_preempt(struct event_msg *ev_msg, unsigned int ev_type,
                              void *data);
static void handle_vc_indir(struct event_msg *ev_msg, unsigned int ev_type,
                            void *data);
static void __ros_uth_syscall_blockon(struct syscall *sysc);

/* Helper, initializes a fresh uthread to be thread0. */
static void uthread_init_thread0(struct uthread *uthread)
{
	assert(uthread);
	/* Save a pointer to thread0's tls region (the glibc one) into its tcb*/
	uthread->tls_desc = get_tls_desc();
	/* Save a pointer to the uthread in its own TLS */
	current_uthread = uthread;
	/* Thread is currently running (it is 'us') */
	uthread->state = UT_RUNNING;
	/* Thread is detached */
	atomic_set(&uthread->join_ctl.state, UTH_JOIN_DETACHED);
	/* Reset the signal state */
	uthread->sigstate.mask = 0;
	/* sig alt stack pointer */
	uthread->sigstate.sigalt_stacktop = 0;
	__sigemptyset(&uthread->sigstate.pending);
	uthread->sigstate.data = NULL;
	/* utf/as doesn't represent the state of the uthread (we are running) */
	uthread->flags &= ~(UTHREAD_SAVED | UTHREAD_FPSAVED);
	/* need to track thread0 for TLS deallocation */
	uthread->flags |= UTHREAD_IS_THREAD0;
	uthread->notif_disabled_depth = 0;
	/* setting the uthread's TLS var.  this is idempotent for SCPs (us) */
	__vcoreid = 0;
}

/* Helper, makes VC ctx tracks uthread as its current_uthread in its TLS.
 *
 * Whether or not uthreads have TLS, thread0 has TLS, given to it by glibc.
 * This TLS will get set whenever we use thread0, regardless of whether or not
 * we use TLS for uthreads in general.  glibc cares about this TLS and will use
 * it at exit.  We can't simply use that TLS for VC0 either, since we don't know
 * where thread0 will be running when the program ends. */
static void uthread_track_thread0(struct uthread *uthread)
{
	set_tls_desc(get_vcpd_tls_desc(0));
	begin_safe_access_tls_vars();
	current_uthread = uthread;
	__vcore_context = TRUE;
	end_safe_access_tls_vars();
	set_tls_desc(uthread->tls_desc);
}

/* The real 2LS calls this to transition us into mcp mode.  When it
 * returns, you're in _M mode, still running thread0, on vcore0 */
void uthread_mcp_init()
{
	/* Prevent this from happening more than once. */
	parlib_init_once_racy(return);

	/* Doing this after the init_once check, since we don't want to let the
	 * process/2LS change their mind about being an MCP or not once they
	 * have multiple threads.
	 *
	 * The reason is that once you set "MCP please" on, you could get
	 * interrupted into VC ctx, say for a syscall completion, and then make
	 * decisions based on the fact that you're an MCP (e.g., unblocking a
	 * uthread, asking for vcores, etc), even though you are not an MCP.
	 * Arguably, these things could happen for signals too, but all of this
	 * is less likely than if we have multiple threads.
	 *
	 * Also, we could just abort here, since they shouldn't be calling
	 * mcp_init() if they don't want to be an MCP. */
	if (!parlib_wants_to_be_mcp)
		return;

	/* Receive preemption events.  Note that this merely tells the kernel
	 * how to send the messages, and does not necessarily provide storage
	 * space for the messages.  What we're doing is saying that all PREEMPT
	 * and CHECK_MSGS events should be spammed to vcores that are running,
	 * preferring whatever the kernel thinks is appropriate.  And IPI them.
	 *
	 * It is critical that these are either SPAM_PUB or INDIR|SPAM_INDIR, so
	 * that yielding vcores do not miss the preemption messages. */
	register_ev_handler(EV_VCORE_PREEMPT, handle_vc_preempt, 0);
	register_ev_handler(EV_CHECK_MSGS, handle_vc_indir, 0);
	/* small ev_q, mostly a vehicle for flags */
	preempt_ev_q = get_eventq_slim();
	preempt_ev_q->ev_flags = EVENT_IPI | EVENT_SPAM_PUBLIC |
				 EVENT_VCORE_APPRO | EVENT_VCORE_MUST_RUN |
				 EVENT_WAKEUP;
	/* Tell the kernel to use the ev_q (it's settings) for the two types.
	 * Note that we still have two separate handlers.  We just want the
	 * events delivered in the same way.  If we ever want to have a
	 * big_event_q with INDIRs, we could consider using separate ones. */
	register_kevent_q(preempt_ev_q, EV_VCORE_PREEMPT);
	register_kevent_q(preempt_ev_q, EV_CHECK_MSGS);
	printd("[user] registered %08p (flags %08p) for preempt messages\n",
	       preempt_ev_q, preempt_ev_q->ev_flags);
	/* Get ourselves into _M mode.  Could consider doing this elsewhere. */
	vcore_change_to_m();
}

/* Helper: tells the kernel our SCP is capable of going into vcore context on
 * vcore 0.  Pairs with k/s/process.c scp_is_vcctx_ready(). */
static void scp_vcctx_ready(void)
{
	struct preempt_data *vcpd = vcpd_of(0);
	long old_flags;

	/* the CAS is a bit overkill; keeping it around in case people use this
	 * code in other situations. */
	do {
		old_flags = atomic_read(&vcpd->flags);
		/* Spin if the kernel is mucking with the flags */
		while (old_flags & VC_K_LOCK)
			old_flags = atomic_read(&vcpd->flags);
	} while (!atomic_cas(&vcpd->flags, old_flags,
	                     old_flags & ~VC_SCP_NOVCCTX));
}

/* For both of these, VC ctx uses the usual TLS errno/errstr.  Uthreads use
 * their own storage.  Since we're called after manage_thread0, we should always
 * have current_uthread if we are not in vc ctx. */
static int *__ros_errno_loc(void)
{
	if (in_vcore_context())
		return __errno_location_tls();
	else
		return &current_uthread->err_no;
}

static char *__ros_errstr_loc(void)
{
	if (in_vcore_context())
		return __errstr_location_tls();
	else
		return current_uthread->err_str;
}

static void __attribute__((constructor)) uthread_lib_ctor(void)
{
	/* Surprise!  Parlib's ctors also run in shared objects.  We can't have
	 * multiple versions of parlib (with multiple data structures). */
	if (__in_fake_parlib())
		return;
	/* Need to make sure vcore_lib_init() runs first */
	vcore_lib_init();
	/* Instead of relying on ctors for the specific 2LS, we make sure they
	 * are called next.  They will call uthread_2ls_init().
	 *
	 * The potential issue here is that C++ ctors might make use of the
	 * GCC/C++ threading callbacks, which require the full 2LS.  There's no
	 * linkage dependency  between C++ and the specific 2LS, so there's no
	 * way to be sure the 2LS actually turned on before we started calling
	 * into it.
	 *
	 * Hopefully, the uthread ctor was called in time, since the GCC
	 * threading functions link against parlib.  Note that, unlike
	 * parlib-compat.c, there are no stub functions available to GCC that
	 * could get called by accident and prevent the linkage. */
	sched_ops->sched_init();
}

/* The 2LS calls this, passing in a uthread representing thread0 and its
 * syscall handling routine.  (NULL is fine).  The 2LS sched_ops is known
 * statically (via symbol overrides).
 *
 * This is where parlib (and whatever 2LS is linked in) takes over control of
 * scheduling, including handling notifications, having sched_entry() called,
 * blocking syscalls, and handling syscall completion events.  Before this
 * call, these things are handled by slim functions in glibc (e.g. early
 * function pointers for ros_blockon) and by the kernel.  The kerne's role was
 * to treat the process specially until we call scp_vcctx_ready(): things like
 * no __notify, no sched_entry, etc.
 *
 * We need to be careful to not start using the 2LS before it is fully ready.
 * For instance, once we change ros_blockon, we could have a blocking syscall
 * (e.g. for something glibc does) and the rest of the 2LS code expects things
 * to be in place.
 *
 * In older versions of this code, we would hop from the thread0 sched to the
 * real 2LSs sched, which meant we had to be very careful.  But now that we
 * only do this once, we can do all the prep work and then take over from
 * glibc's early SCP setup.  Specifically, notifs are disabled (due to the
 * early SCP ctx) and syscalls won't use the __ros_uth_syscall_blockon, so we
 * shouldn't get a syscall event.
 *
 * Still, if you have things like an outstanding async syscall, then you'll
 * have issues.  Most likely it would complete and you'd never hear about it.
 *
 * Note that some 2LS ops can be called even before we've initialized the 2LS!
 * Some ops, like the sync_obj ops, are called when initializing an uncontested
 * mutex, which could be called from glibc (e.g. malloc).  Hopefully that's
 * fine - we'll see!  I imagine a contested mutex would be a disaster (during
 * the unblock), which shouldn't happen as we are single threaded. */
void uthread_2ls_init(struct uthread *uthread,
                      void (*handle_sysc)(struct event_msg *, unsigned int,
                                          void *),
                      void *data)
{
	struct ev_handler *new_h = NULL;

	if (handle_sysc) {
		new_h = malloc(sizeof(struct ev_handler));
		assert(new_h);
		new_h->func = handle_sysc;
		new_h->data = data;
		new_h->next = NULL;
		assert(!ev_handlers[EV_SYSCALL]);
		ev_handlers[EV_SYSCALL] = new_h;
	}
	uthread_init_thread0(uthread);
	uthread_track_thread0(uthread);
	/* Switch our errno/errstr functions to be uthread-aware.  See glibc's
	 * errno.c for more info. */
	ros_errno_loc = __ros_errno_loc;
	ros_errstr_loc = __ros_errstr_loc;
	register_ev_handler(EV_EVENT, handle_ev_ev, 0);
	cmb();
	/* Now that we're ready (I hope) to operate as a full process, we tell
	 * the kernel.  We must set vcctx and blockon atomically with respect to
	 * syscalls, meaning no syscalls in between. */
	scp_vcctx_ready();
	/* Change our blockon from glibc's internal one to the regular one,
	 * which uses vcore context and works for SCPs (with or without 2LS) and
	 * MCPs.  Now that we told the kernel we are ready to utilize vcore
	 * context, we need our blocking syscalls to utilize it as well. */
	ros_syscall_blockon = __ros_uth_syscall_blockon;
	cmb();
	init_posix_signals();
	/* Accept diagnostic events.  Other parts of the program/libraries can
	 * register handlers to run.  You can kick these with "notify PID 9". */
	enable_kevent(EV_FREE_APPLE_PIE, 0, EVENT_IPI | EVENT_WAKEUP |
	                                    EVENT_SPAM_PUBLIC);
}

/* 2LSs shouldn't call uthread_vcore_entry directly */
void __attribute__((noreturn)) uthread_vcore_entry(void)
{
	uint32_t vcoreid = vcore_id();
	struct preempt_data *vcpd = vcpd_of(vcoreid);

	/* Should always have notifications disabled when coming in here. */
	assert(!notif_is_enabled(vcoreid));
	assert(in_vcore_context());
	/* It's possible to have our FPSAVED already, e.g. any vcore reentry
	 * (refl fault, some preemption handling, etc) if cur_uth wasn't reset.
	 * In those cases, the FP state should be the same in the processor and
	 * in the uth, so we might be able to drop the FPSAVED check/branch. */
	if (current_uthread && !(current_uthread->flags & UTHREAD_FPSAVED) &&
	    !cur_uth_is_sw_ctx()) {
		save_fp_state(&current_uthread->as);
		current_uthread->flags |= UTHREAD_FPSAVED;
	}
	/* If someone is stealing our uthread (from when we were preempted
	 * before), we can't touch our uthread.  But we might be the last vcore
	 * around, so we'll handle preemption events (spammed to our public
	 * mbox).
	 *
	 * It's important that we only check/handle one message per loop,
	 * otherwise we could get stuck in a ping-pong scenario with a recoverer
	 * (maybe). */
	while (atomic_read(&vcpd->flags) & VC_UTHREAD_STEALING) {
		/* Note we're handling INDIRs and other public messages while
		 * someone is stealing our uthread.  Remember that those event
		 * handlers cannot touch cur_uth, as it is "vcore business". */
		handle_one_mbox_msg(&vcpd->ev_mbox_public);
		cpu_relax();
	}
	/* If we have a current uthread that is DONT_MIGRATE, pop it real quick
	 * and let it disable notifs (like it wants to).  Other than dealing
	 * with preemption events (or other INDIRs), we shouldn't do anything in
	 * vc_ctx when we have a DONT_MIGRATE uthread. */
	if (current_uthread && (current_uthread->flags & UTHREAD_DONT_MIGRATE))
		__run_current_uthread_raw();
	/* Check and see if we wanted ourselves to handle a remote VCPD mbox.
	 * Want to do this after we've handled STEALING and DONT_MIGRATE. */
	try_handle_remote_mbox();
	/* Otherwise, go about our usual vcore business (messages, etc). */
	handle_events(vcoreid);
	__check_preempt_pending(vcoreid);
	/* double check, in case an event changed it */
	assert(in_vcore_context());
	sched_ops->sched_entry();
	assert(0); /* 2LS sched_entry should never return */
}

/* Does the uthread initialization of a uthread that the caller created.  Call
 * this whenever you are "starting over" with a thread. */
void uthread_init(struct uthread *new_thread, struct uth_thread_attr *attr)
{
	int ret;
	assert(new_thread);
	new_thread->state = UT_NOT_RUNNING;
	/* Set the signal state. */
	if (current_uthread)
		new_thread->sigstate.mask = current_uthread->sigstate.mask;
	else
		new_thread->sigstate.mask = 0;
	__sigemptyset(&new_thread->sigstate.pending);
	new_thread->sigstate.data = NULL;
	new_thread->sigstate.sigalt_stacktop = 0;
	new_thread->flags = 0;
	new_thread->sysc = NULL;
	/* the utf holds the GP context of the uthread (set by the 2LS earlier).
	 * There is no FP context to be restored yet.  We only save the FPU when
	 * we were interrupted off a core. */
	new_thread->flags |= UTHREAD_SAVED;
	new_thread->notif_disabled_depth = 0;
	/* TODO: on a reinit, if they changed whether or not they want TLS,
	 * we'll have issues (checking tls_desc, assert in allocate_tls, maybe
	 * more). */
	if (attr && attr->want_tls) {
		/* Get a TLS.  If we already have one, reallocate/refresh it */
		if (new_thread->tls_desc)
			ret = __uthread_reinit_tls(new_thread);
		else
			ret = __uthread_allocate_tls(new_thread);
		assert(!ret);
		begin_access_tls_vars(new_thread->tls_desc);
		current_uthread = new_thread;
		/* ctypes stores locale info in TLS.  we need this only once per
		 * TLS, so we don't have to do it here, but it is convenient
		 * since we already loaded the uthread's TLS. */
		extern void __ctype_init(void);
		__ctype_init();
		end_access_tls_vars();
	} else {
		new_thread->tls_desc = UTH_TLSDESC_NOTLS;
	}
	if (attr && attr->detached)
		atomic_set(&new_thread->join_ctl.state, UTH_JOIN_DETACHED);
	else
		atomic_set(&new_thread->join_ctl.state, UTH_JOIN_JOINABLE);
}

/* This is a wrapper for the sched_ops thread_runnable, for use by functions
 * outside the main 2LS.  Do not put anything important in this, since the 2LSs
 * internally call their sched op.  This is to improve batch wakeups (barriers,
 * etc) */
void uthread_runnable(struct uthread *uthread)
{
	assert(sched_ops->thread_runnable);
	sched_ops->thread_runnable(uthread);
}

/* Informs the 2LS that its thread blocked, and it is not under the control of
 * the 2LS.  This is for informational purposes, and some semantic meaning
 * should be passed by flags (from uthread.h's UTH_EXT_BLK_xxx options).
 * Eventually, whoever calls this will call uthread_runnable(), giving the
 * thread back to the 2LS.  If the 2LS provide sync ops, it will have a say in
 * which thread wakes up at a given time.
 *
 * If code outside the 2LS has blocked a thread (via uthread_yield) and ran its
 * own callback/yield_func instead of some 2LS code, that callback needs to
 * call this.
 *
 * AKA: obviously_a_uthread_has_blocked_in_lincoln_park() */
void uthread_has_blocked(struct uthread *uthread, int flags)
{
	assert(sched_ops->thread_has_blocked);
	sched_ops->thread_has_blocked(uthread, flags);
}

/* Function indicating an external event has temporarily paused a uthread, but
 * it is ok to resume it if possible. */
void uthread_paused(struct uthread *uthread)
{
	/* Call out to the 2LS to let it know the uthread was paused for some
	 * reason, but it is ok to resume it now. */
	assert(uthread->state == UT_NOT_RUNNING);
	assert(sched_ops->thread_paused);
	sched_ops->thread_paused(uthread);
}

/* Need to have this as a separate, non-inlined function since we clobber the
 * stack pointer before calling it, and don't want the compiler to play games
 * with my hart. */
static void __attribute__((noinline, noreturn))
__uthread_yield(void)
{
	struct uthread *uthread = current_uthread;
	assert(in_vcore_context());
	assert(!notif_is_enabled(vcore_id()));
	/* Note: we no longer care if the thread is exiting, the 2LS will call
	 * uthread_destroy() */
	uthread->flags &= ~UTHREAD_DONT_MIGRATE;
	uthread->state = UT_NOT_RUNNING;
	/* Any locks that were held before the yield must be unlocked in the
	 * callback.  That callback won't get a chance to update our disabled
	 * depth.  This sets us up for the next time the uthread runs. */
	assert(uthread->notif_disabled_depth <= 1);
	uthread->notif_disabled_depth = 0;
	/* Do whatever the yielder wanted us to do */
	assert(uthread->yield_func);
	uthread->yield_func(uthread, uthread->yield_arg);
	/* Make sure you do not touch uthread after that func call */
	/* Leave the current vcore completely */
	/* TODO: if the yield func can return a failure, we can abort the yield
	 */
	current_uthread = NULL;
	/* Go back to the entry point, where we can handle notifications or
	 * reschedule someone. */
	uthread_vcore_entry();
}

/* Calling thread yields for some reason.  Set 'save_state' if you want to ever
 * run the thread again.  Once in vcore context in __uthread_yield, yield_func
 * will get called with the uthread and yield_arg passed to it.  This way, you
 * can do whatever you want when you get into vcore context, which can be
 * thread_blockon_sysc, unlocking mutexes, joining, whatever.
 *
 * If you do *not* pass a 2LS sched op or other 2LS function as yield_func,
 * then you must also call uthread_has_blocked(flags), which will let the 2LS
 * know a thread blocked beyond its control (and why). */
void uthread_yield(bool save_state, void (*yield_func)(struct uthread*, void*),
                   void *yield_arg)
{
	struct uthread *uthread = current_uthread;
	volatile bool yielding = TRUE; /* signal to short circuit on restart */
	assert(!in_vcore_context());
	assert(uthread->state == UT_RUNNING);
	/* Pass info to ourselves across the uth_yield -> __uth_yield
	 * transition. */
	uthread->yield_func = yield_func;
	uthread->yield_arg = yield_arg;
	/* Don't migrate this thread to another vcore, since it depends on being
	 * on the same vcore throughout (once it disables notifs).  The race is
	 * that we read vcoreid, then get interrupted / migrated before
	 * disabling notifs. */
	uthread->flags |= UTHREAD_DONT_MIGRATE;
	cmb();	/* don't let DONT_MIGRATE write pass the vcoreid read */
	uint32_t vcoreid = vcore_id();

	printd("[U] Uthread %08p is yielding on vcore %d\n", uthread, vcoreid);
	struct preempt_data *vcpd = vcpd_of(vcoreid);

	/* once we do this, we might miss a notif_pending, so we need to enter
	 * vcore entry later.  Need to disable notifs so we don't get in weird
	 * loops with save_user_ctx() and pop_user_ctx(). */
	disable_notifs(vcoreid);
	/* take the current state and save it into t->utf when this pthread
	 * restarts, it will continue from right after this, see yielding is
	 * false, and short ciruit the function.  Don't do this if we're dying.
	 * */
	if (save_state) {
		/* Need to signal this before we actually save, since
		 * save_user_ctx returns() twice (once now, once when woken up)
		 */
		uthread->flags |= UTHREAD_SAVED;
		save_user_ctx(&uthread->u_ctx);
	}
	/* Force reread of yielding. Technically save_user_ctx() suffices*/
	cmb();
	/* Restart path doesn't matter if we're dying */
	if (!yielding)
		goto yield_return_path;
	/* From here on down is only executed on the save path (not the wake up)
	 */
	yielding = FALSE; /* for when it starts back up */
	/* TODO: remove this when all arches support SW contexts */
	if (save_state && (uthread->u_ctx.type != ROS_SW_CTX)) {
		save_fp_state(&uthread->as);
		uthread->flags |= UTHREAD_FPSAVED;
	}
	/* Change to the transition context (both TLS (if applicable) and
	 * stack). */
	if (__uthread_has_tls(uthread)) {
		set_tls_desc(get_vcpd_tls_desc(vcoreid));
		begin_safe_access_tls_vars();
		assert(current_uthread == uthread);
		/* If this assert fails, see the note in uthread_track_thread0
		 */
		assert(in_vcore_context());
		end_safe_access_tls_vars();
	} else {
		/* Since uthreads and vcores share TLS (it's always the vcore's
		 * TLS, the uthread one just bootstraps from it), we need to
		 * change our state at boundaries between the two 'contexts' */
		__vcore_context = TRUE;
	}
	/* After this, make sure you don't use local variables.  Also, make sure
	 * the compiler doesn't use them without telling you (TODO).
	 *
	 * In each arch's set_stack_pointer, make sure you subtract off as much
	 * room as you need to any local vars that might be pushed before
	 * calling the next function, or for whatever other reason the
	 * compiler/hardware might walk up the stack a bit when calling a
	 * noreturn function. */
	set_stack_pointer((void*)vcpd->vcore_stack);
	/* Finish exiting in another function. */
	__uthread_yield();
	/* Should never get here */
	assert(0);
	/* Will jump here when the uthread's trapframe is restarted/popped. */
yield_return_path:
	printd("[U] Uthread %08p returning from a yield!\n", uthread);
}

/* We explicitly don't support sleep(), since old callers of it have
 * expectations of being woken up by signal handlers.  If we need that, we can
 * build it in to sleep() later.  If you just want to sleep for a while, call
 * this helper. */
void uthread_sleep(unsigned int seconds)
{
	sys_block(seconds * 1000000);	/* usec sleep */
}
/* If we are providing a dummy sleep function, might as well provide the more
 * accurate/useful one. */
void uthread_usleep(unsigned int usecs)
{
	sys_block(usecs);	/* usec sleep */
}

static void __sleep_forever_cb(struct uthread *uth, void *arg)
{
	uthread_has_blocked(uth, UTH_EXT_BLK_MISC);
}

void __attribute__((noreturn)) uthread_sleep_forever(void)
{
	uthread_yield(FALSE, __sleep_forever_cb, NULL);
	assert(0);
}

/* Cleans up the uthread (the stuff we did in uthread_init()).  If you want to
 * destroy a currently running uthread, you'll want something like
 * pthread_exit(), which yields, and calls this from its sched_ops yield. */
void uthread_cleanup(struct uthread *uthread)
{
	printd("[U] thread %08p on vcore %d is DYING!\n", uthread, vcore_id());
	/* we alloc and manage the TLS, so lets get rid of it, except for
	 * thread0.  glibc owns it.  might need to keep it around for a full
	 * exit() */
	if (__uthread_has_tls(uthread) && !(uthread->flags & UTHREAD_IS_THREAD0))
		__uthread_free_tls(uthread);
}

static void __ros_syscall_spinon(struct syscall *sysc)
{
	while (!(atomic_read(&sysc->flags) & (SC_DONE | SC_PROGRESS)))
		cpu_relax();
}

static void __ros_vcore_ctx_syscall_blockon(struct syscall *sysc)
{
	if (in_multi_mode()) {
		/* MCP vcore's don't know what to do yet, so we have to spin */
		__ros_syscall_spinon(sysc);
	} else {
		/* SCPs can use the early blockon, which acts like VC ctx. */
		__ros_early_syscall_blockon(sysc);
	}
}

/* Attempts to block on sysc, returning when it is done or progress has been
 * made.  Made for initialized processes using uthreads. */
static void __ros_uth_syscall_blockon(struct syscall *sysc)
{
	if (in_vcore_context()) {
		__ros_vcore_ctx_syscall_blockon(sysc);
		return;
	}
	/* At this point, we know we're a uthread.  If we're a DONT_MIGRATE
	 * uthread, then it's disabled notifs and is basically in vcore context,
	 * enough so that it can't call into the 2LS. */
	assert(current_uthread);
	if (current_uthread->flags & UTHREAD_DONT_MIGRATE) {
		assert(!notif_is_enabled(vcore_id()));	/* catch bugs */
		/* if we had a notif_disabled_depth, then we should also have
		 * DONT_MIGRATE set */
		__ros_vcore_ctx_syscall_blockon(sysc);
		return;
	}
	assert(!current_uthread->notif_disabled_depth);
	/* double check before doing all this crap */
	if (atomic_read(&sysc->flags) & (SC_DONE | SC_PROGRESS))
		return;
	/* for both debugging and syscall cancelling */
	current_uthread->sysc = sysc;
	/* yield, calling 2ls-blockon(cur_uth, sysc) on the other side */
	uthread_yield(TRUE, sched_ops->thread_blockon_sysc, sysc);
}

/* 2LS helper.  Run this from vcore context.  It will block a uthread on it's
 * internal syscall struct, which should be an async call.  You'd use this in
 * e.g. thread_refl_fault when the 2LS initiates a syscall on behalf of the
 * uthread. */
void __block_uthread_on_async_sysc(struct uthread *uth)
{
	assert(in_vcore_context());
	uth->sysc = &uth->local_sysc;
	/* If a DONT_MIGRATE issued a syscall that blocks, we gotta spin, same
	 * as with the usual blockon. */
	if (uth->flags & UTHREAD_DONT_MIGRATE) {
		__ros_vcore_ctx_syscall_blockon(uth->sysc);
		uth->sysc = 0;
		return;
	}
	sched_ops->thread_blockon_sysc(uth, uth->sysc);
}

/* Simply sets current uthread to be whatever the value of uthread is.  This
 * can be called from outside of sched_entry() to highjack the current context,
 * and make sure that the new uthread struct is used to store this context upon
 * yielding, etc. USE WITH EXTREME CAUTION! */
void highjack_current_uthread(struct uthread *uthread)
{
	uint32_t vcoreid = vcore_id();

	assert(uthread != current_uthread);
	current_uthread->state = UT_NOT_RUNNING;
	uthread->state = UT_RUNNING;
	/* Make sure the vcore is tracking the new uthread struct */
	if (__uthread_has_tls(current_uthread))
		vcore_set_tls_var(current_uthread, uthread);
	else
		current_uthread = uthread;
	/* and make sure we are using the correct TLS for the new uthread */
	if (__uthread_has_tls(uthread)) {
		assert(uthread->tls_desc);
		set_tls_desc(uthread->tls_desc);
		begin_safe_access_tls_vars();
		__vcoreid = vcoreid;	/* setting the uthread's TLS var */
		end_safe_access_tls_vars();
	}
}

/* Helper: loads a uthread's TLS on this vcore, if applicable.  If our uthreads
 * do not have their own TLS, we simply switch the __vc_ctx, signalling that the
 * context running here is (soon to be) a uthread. */
static void set_uthread_tls(struct uthread *uthread, uint32_t vcoreid)
{
	if (__uthread_has_tls(uthread)) {
		set_tls_desc(uthread->tls_desc);
		begin_safe_access_tls_vars();
		__vcoreid = vcoreid;	/* setting the uthread's TLS var */
		end_safe_access_tls_vars();
	} else {
		__vcore_context = FALSE;
	}
}

/* Attempts to handle a fault for uth, etc */
static void handle_refl_fault(struct uthread *uth, struct user_context *ctx)
{
	sched_ops->thread_refl_fault(uth, ctx);
}

/* 2LS helper: stops the current uthread, saves its state, and returns a pointer
 * to it.  Unlike __uthread_pause, which is called by non-specific 2LS code,
 * this function is called by a specific 2LS to stop it's current uthread. */
struct uthread *stop_current_uthread(void)
{
	struct uthread *uth;
	struct preempt_data *vcpd = vcpd_of(vcore_id());

	uth = current_uthread;
	current_uthread = 0;
	if (!(uth->flags & UTHREAD_SAVED)) {
		uth->u_ctx = vcpd->uthread_ctx;
		uth->flags |= UTHREAD_SAVED;
	}
	if ((uth->u_ctx.type != ROS_SW_CTX) && !(uth->flags & UTHREAD_FPSAVED))
	{
		save_fp_state(&uth->as);
		uth->flags |= UTHREAD_FPSAVED;
	}
	uth->state = UT_NOT_RUNNING;
	return uth;
}

/* Run the thread that was current_uthread, from a previous run.  Should be
 * called only when the uthread already was running, and we were interrupted by
 * the kernel (event, etc).  Do not call this to run a fresh uthread, even if
 * you've set it to be current. */
void __attribute__((noreturn)) run_current_uthread(void)
{
	struct uthread *uth;
	uint32_t vcoreid = vcore_id();
	struct preempt_data *vcpd = vcpd_of(vcoreid);

	assert(current_uthread);
	assert(current_uthread->state == UT_RUNNING);
	/* Uth was already running, should not have been saved */
	assert(!(current_uthread->flags & UTHREAD_SAVED));
	/* SW CTX FP wasn't saved, but HW/VM was.  There might be some case
	 * where a VMTF hadn't run yet, and thus wasn't interrupted, but it
	 * shouldn't have made it to be current_uthread. */
	if (cur_uth_is_sw_ctx())
		assert(!(current_uthread->flags & UTHREAD_FPSAVED));
	else
		assert(current_uthread->flags & UTHREAD_FPSAVED);
	printd("[U] Vcore %d is restarting uthread %08p\n", vcoreid,
	       current_uthread);
	if (has_refl_fault(&vcpd->uthread_ctx)) {
		clear_refl_fault(&vcpd->uthread_ctx);
		/* we preemptively copy out and make non-running, so that there
		 * is a consistent state for the handler.  it can then block the
		 * uth or whatever. */
		uth = stop_current_uthread();
		handle_refl_fault(uth, &vcpd->uthread_ctx);
		/* we abort no matter what.  up to the 2LS to reschedule the
		 * thread */
		set_stack_pointer((void*)vcpd->vcore_stack);
		vcore_entry();
	}
	if (current_uthread->flags & UTHREAD_FPSAVED) {
		current_uthread->flags &= ~UTHREAD_FPSAVED;
		restore_fp_state(&current_uthread->as);
	}
	set_uthread_tls(current_uthread, vcoreid);
	pop_user_ctx(&vcpd->uthread_ctx, vcoreid);
	assert(0);
}

/* Launches the uthread on the vcore.  Don't call this on current_uthread. 
 *
 * In previous versions of this, we used to check for events after setting
 * current_uthread.  That is super-dangerous.  handle_events() doesn't always
 * return (which we used to handle), and it may also clear current_uthread.  We
 * needed to save uthread in current_uthread, in case we didn't return.  If we
 * didn't return, the vcore started over at vcore_entry, with current set.  When
 * this happens, we never actually had loaded cur_uth's FP and GP onto the core,
 * so cur_uth fails.  Check out 4602599be for more info.
 *
 * Ultimately, handling events again in these 'popping helpers' isn't even
 * necessary (we only must do it once for an entire time in VC ctx, and in
 * loops), and might have been optimizing a rare event at a cost in both
 * instructions and complexity. */
void __attribute__((noreturn)) run_uthread(struct uthread *uthread)
{
	uint32_t vcoreid = vcore_id();
	struct preempt_data *vcpd = vcpd_of(vcoreid);

	assert(!current_uthread);
	assert(uthread->state == UT_NOT_RUNNING);
	assert(uthread->flags & UTHREAD_SAVED);
	/* For HW CTX, FPSAVED must match UTH SAVE (and both be on here).  For
	 * SW, FP should never be saved. */
	switch (uthread->u_ctx.type) {
	case ROS_HW_CTX:
		assert(uthread->flags & UTHREAD_FPSAVED);
		break;
	case ROS_SW_CTX:
		assert(!(uthread->flags & UTHREAD_FPSAVED));
		break;
	case ROS_VM_CTX:
		/* Don't care.  This gives it the state of the vcore when it
		 * starts up.  If we care about leaking FPU / XMM state, we can
		 * create a new one for every VM TF (or vthread reuse). */
		break;
	}
	if (has_refl_fault(&uthread->u_ctx)) {
		clear_refl_fault(&uthread->u_ctx);
		handle_refl_fault(uthread, &uthread->u_ctx);
		/* we abort no matter what.  up to the 2LS to reschedule the
		 * thread */
		set_stack_pointer((void*)vcpd->vcore_stack);
		vcore_entry();
	}
	uthread->state = UT_RUNNING;
	/* Save a ptr to the uthread we'll run in the transition context's TLS
	 */
	current_uthread = uthread;
	if (uthread->flags & UTHREAD_FPSAVED) {
		uthread->flags &= ~UTHREAD_FPSAVED;
		restore_fp_state(&uthread->as);
	}
	set_uthread_tls(uthread, vcoreid);
	/* the uth's context will soon be in the cpu (or VCPD), no longer saved
	 */
	uthread->flags &= ~UTHREAD_SAVED;
	pop_user_ctx(&uthread->u_ctx, vcoreid);
	assert(0);
}

/* Runs the uthread, but doesn't care about notif pending.  Only call this when
 * there was a DONT_MIGRATE uthread, or a similar situation where the uthread
 * will check messages soon (like calling enable_notifs()). */
static void __run_current_uthread_raw(void)
{
	uint32_t vcoreid = vcore_id();
	struct preempt_data *vcpd = vcpd_of(vcoreid);

	if (has_refl_fault(&vcpd->uthread_ctx)) {
		printf("Raw / DONT_MIGRATE uthread took a fault, exiting.\n");
		exit(-1);
	}
	/* We need to manually say we have a notif pending, so we eventually
	 * return to vcore context.  (note the kernel turned it off for us) */
	vcpd->notif_pending = TRUE;
	assert(!(current_uthread->flags & UTHREAD_SAVED));
	if (current_uthread->flags & UTHREAD_FPSAVED) {
		current_uthread->flags &= ~UTHREAD_FPSAVED;
		restore_fp_state(&current_uthread->as);
	}
	set_uthread_tls(current_uthread, vcoreid);
	pop_user_ctx_raw(&vcpd->uthread_ctx, vcoreid);
	assert(0);
}

/* Copies the uthread trapframe and silly state from the vcpd to the uthread,
 * subject to the uthread's flags and whatnot.
 *
 * For example: The uthread state might still be in the uthread struct.  Imagine
 * the 2LS decides to run a new uthread and sets it up as current, but doesn't
 * actually run it yet.  The 2LS happened to voluntarily give up the VC (due to
 * some other event) and then wanted to copy out the thread.  This is pretty
 * rare - the normal case is when an IRQ of some sort hit the core and the
 * kernel copied the running state into VCPD.
 *
 * The FP state could also be in VCPD (e.g. preemption being handled remotely),
 * it could be in the uthread struct (e.g. hasn't started running yet) or even
 * in the FPU (e.g. took an IRQ/notif and we're handling the preemption of
 * another vcore).
 *
 * There are some cases where we'll have a uthread SW ctx that needs to be
 * copied out: uth syscalls, notif happens, and the core comes back from the
 * kernel in VC ctx.  VC ctx calls copy_out (response to preempt_pending or done
 * while handling a preemption). */
static void copyout_uthread(struct preempt_data *vcpd, struct uthread *uthread,
                            bool vcore_local)
{
	assert(uthread);
	if (uthread->flags & UTHREAD_SAVED) {
		/* GP saved -> FP saved, but not iff.  FP could be saved due to
		 * aggressive save/restore. */
		switch (uthread->u_ctx.type) {
		case ROS_HW_CTX:
		case ROS_VM_CTX:
			assert(uthread->flags & UTHREAD_FPSAVED);
		}
		assert(vcore_local);
		return;
	}
	/* If we're copying GP state, it must be in VCPD */
	uthread->u_ctx = vcpd->uthread_ctx;
	uthread->flags |= UTHREAD_SAVED;
	printd("VC %d copying out uthread %08p\n", vcore_id(), uthread);
	/* Software contexts do not need FP state, nor should we think it has
	 * any */
	if (uthread->u_ctx.type == ROS_SW_CTX) {
		assert(!(uthread->flags & UTHREAD_FPSAVED));
		return;
	}
	/* We might have aggressively saved for non-SW ctx in vc_entry before we
	 * got to the event handler. */
	if (uthread->flags & UTHREAD_FPSAVED) {
		/* If this fails, we're remote.  But the target vcore should not
		 * be in uth context (which is when we'd be stealing a uthread)
		 * with FPSAVED, just like how it shouldn't have GP saved. */
		assert(vcore_local);
		return;
	}
	/* When we're dealing with the uthread running on our own vcore, the FP
	 * state is in the actual FPU, not VCPD.  It might also be in VCPD, but
	 * it will always be in the FPU (the kernel maintains this for us, in
	 * the event we were preempted since the uthread was last running). */
	if (vcore_local)
		save_fp_state(&uthread->as);
	else
		uthread->as = vcpd->preempt_anc;
	uthread->flags |= UTHREAD_FPSAVED;
}

/* Helper, packages up and pauses a uthread that was running on vcoreid.  Used
 * by preemption handling (and detection) so far.  Careful using this, esp if
 * it is on another vcore (need to make sure it's not running!).  If you are
 * using it on the local vcore, set vcore_local = TRUE. */
static void __uthread_pause(struct preempt_data *vcpd, struct uthread *uthread,
                            bool vcore_local)
{
	assert(!(uthread->flags & UTHREAD_DONT_MIGRATE));
	copyout_uthread(vcpd, uthread, vcore_local);
	uthread->state = UT_NOT_RUNNING;
	/* Call out to the 2LS to package up its uthread */
	assert(sched_ops->thread_paused);
	sched_ops->thread_paused(uthread);
}

/* Deals with a pending preemption (checks, responds).  If the 2LS registered a
 * function, it will get run.  Returns true if you got preempted.  Called
 * 'check' instead of 'handle', since this isn't an event handler.  It's the "Oh
 * shit a preempt is on its way ASAP".
 *
 * Be careful calling this: you might not return, so don't call it if you can't
 * handle that.  If you are calling this from an event handler, you'll need to
 * do things like ev_might_not_return().  If the event can via an INDIR ev_q,
 * that ev_q must be a NOTHROTTLE.
 *
 * Finally, don't call this from a place that might have a DONT_MIGRATE
 * cur_uth.  This should be safe for most 2LS code. */
bool __check_preempt_pending(uint32_t vcoreid)
{
	bool retval = FALSE;
	assert(in_vcore_context());
	if (__preempt_is_pending(vcoreid)) {
		retval = TRUE;
		if (sched_ops->preempt_pending)
			sched_ops->preempt_pending();
		/* If we still have a cur_uth, copy it out and hand it back to
		 * the 2LS before yielding. */
		if (current_uthread) {
			__uthread_pause(vcpd_of(vcoreid), current_uthread,
					TRUE);
			current_uthread = 0;
		}
		/* vcore_yield tries to yield, and will pop back up if this was
		 * a spurious preempt_pending or if it handled an event.  For
		 * now, we'll just keep trying to yield so long as a preempt is
		 * coming in.  Eventually, we'll handle all of our events and
		 * yield, or else the preemption will hit and someone will
		 * recover us (at which point we'll break out of the loop) */
		while (__procinfo.vcoremap[vcoreid].preempt_pending) {
			vcore_yield(TRUE);
			cpu_relax();
		}
	}
	return retval;
}

/* Helper: This is a safe way for code to disable notifs if it *might* be called
 * from uthread context (like from a notif_safe lock).  Pair this with
 * uth_enable_notifs() unless you know what you're doing. */
void uth_disable_notifs(void)
{
	if (!in_vcore_context()) {
		if (current_uthread) {
			if (current_uthread->notif_disabled_depth++)
				goto out;
			current_uthread->flags |= UTHREAD_DONT_MIGRATE;
			/* don't issue the flag write before the vcore_id() read
			 */
			cmb();
		}
		disable_notifs(vcore_id());
	}
out:
	assert(!notif_is_enabled(vcore_id()));
}

/* Helper: Pair this with uth_disable_notifs(). */
void uth_enable_notifs(void)
{
	if (!in_vcore_context()) {
		if (current_uthread) {
			if (--current_uthread->notif_disabled_depth)
				return;
			current_uthread->flags &= ~UTHREAD_DONT_MIGRATE;
			cmb();	/* don't enable before ~DONT_MIGRATE */
		}
		enable_notifs(vcore_id());
	}
}

void assert_can_block(void)
{
	if (in_vcore_context())
		panic("Vcore context tried to block!");
	if (!current_uthread) {
		/* Pre-parlib SCPs can do whatever. */
		if (atomic_read(&vcpd_of(0)->flags) & VC_SCP_NOVCCTX)
			return;
		panic("No current_uthread and tried to block!");
	}
	if (current_uthread->notif_disabled_depth)
		panic("Uthread tried to block with notifs disabled!");
	if (current_uthread->flags & UTHREAD_DONT_MIGRATE)
		panic("Uthread tried to block with DONT_MIGRATE!");
}

/* Helper: returns TRUE if it succeeded in starting the uth stealing process. */
static bool start_uth_stealing(struct preempt_data *vcpd)
{
	long old_flags;
	do {
		old_flags = atomic_read(&vcpd->flags);
		/* Spin if the kernel is mucking with the flags */
		while (old_flags & VC_K_LOCK)
			old_flags = atomic_read(&vcpd->flags);
		/* Someone else is stealing, we failed */
		if (old_flags & VC_UTHREAD_STEALING)
			return FALSE;
	} while (!atomic_cas(&vcpd->flags, old_flags,
	                     old_flags | VC_UTHREAD_STEALING));
	return TRUE;
}

/* Helper: pairs with stop_uth_stealing */
static void stop_uth_stealing(struct preempt_data *vcpd)
{
	long old_flags;
	do {
		old_flags = atomic_read(&vcpd->flags);
		assert(old_flags & VC_UTHREAD_STEALING);	/* sanity */
		while (old_flags & VC_K_LOCK)
			old_flags = atomic_read(&vcpd->flags);
	} while (!atomic_cas(&vcpd->flags, old_flags,
	                     old_flags & ~VC_UTHREAD_STEALING));
}

/* Handles INDIRS for another core (the public mbox).  We synchronize with the
 * kernel (__set_curtf_to_vcoreid). */
static void handle_indirs(uint32_t rem_vcoreid)
{
	long old_flags;
	struct preempt_data *rem_vcpd = vcpd_of(rem_vcoreid);
	/* Turn off their message reception if they are still preempted.  If
	 * they are no longer preempted, we do nothing - they will handle their
	 * own messages.  Turning off CAN_RCV will route this vcore's messages
	 * to fallback vcores (if those messages were 'spammed'). */
	do {
		old_flags = atomic_read(&rem_vcpd->flags);
		while (old_flags & VC_K_LOCK)
			old_flags = atomic_read(&rem_vcpd->flags);
		if (!(old_flags & VC_PREEMPTED))
			return;
	} while (!atomic_cas(&rem_vcpd->flags, old_flags,
	                     old_flags & ~VC_CAN_RCV_MSG));
	wrmb();	/* don't let the CAN_RCV write pass reads of the mbox status */
	/* handle all INDIRs of the remote vcore */
	handle_vcpd_mbox(rem_vcoreid);
}

/* Helper.  Will ensure a good attempt at changing vcores, meaning we try again
 * if we failed for some reason other than the vcore was already running. */
static void __change_vcore(uint32_t rem_vcoreid, bool enable_my_notif)
{
	/* okay to do a normal spin/relax here, even though we are in vcore
	 * context. */
	while (-EAGAIN == sys_change_vcore(rem_vcoreid, enable_my_notif))
		cpu_relax();
}

/* Helper, used in preemption recovery.  When you can freely leave vcore
 * context and need to change to another vcore, call this.  vcpd is the caller,
 * rem_vcoreid is the remote vcore.  This will try to package up your uthread.
 * It may return, either because the other core already started up (someone else
 * got it), or in some very rare cases where we had to stay in our vcore
 * context */
static void change_to_vcore(struct preempt_data *vcpd, uint32_t rem_vcoreid)
{
	bool were_handling_remotes;

	/* Unlikely, but if we have no uthread we can just change.  This is the
	 * check, sync, then really check pattern: we can only really be sure
	 * about current_uthread after we check STEALING. */
	if (!current_uthread) {
		/* there might be an issue with doing this while someone is
		 * recovering.  once they 0'd it, we should be good to yield.
		 * just a bit dangerous. */
		were_handling_remotes = ev_might_not_return();
		__change_vcore(rem_vcoreid, TRUE);/* noreturn on success */
		goto out_we_returned;
	}
	/* Note that the reason we need to check STEALING is because we can get
	 * into vcore context and slip past that check in vcore_entry when we
	 * are handling a preemption message.  Anytime preemption recovery cares
	 * about the calling vcore's cur_uth, it needs to be careful about
	 * STEALING.  But it is safe to do the check up above (if it's 0, it
	 * won't concurrently become non-zero).
	 *
	 * STEALING might be turned on at any time.  Whoever turns it on will do
	 * nothing if we are online or were in vc_ctx.  So if it is on, we can't
	 * touch current_uthread til it is turned off (not sure what state they
	 * saw us in).  We could spin here til they unset STEALING (since they
	 * will soon), but there is a chance they were preempted, so we need to
	 * make progress by doing a sys_change_vcore(). */
	/* Crap, someone is stealing (unlikely).  All we can do is change. */
	if (atomic_read(&vcpd->flags) & VC_UTHREAD_STEALING) {
		__change_vcore(rem_vcoreid, FALSE);	/* returns on success */
		return;
	}
	cmb();
	/* Need to recheck, in case someone stole it and finished before we
	 * checked VC_UTHREAD_STEALING. */
	if (!current_uthread) {
		were_handling_remotes = ev_might_not_return();
		__change_vcore(rem_vcoreid, TRUE); 	/* noreturn on success*/
		goto out_we_returned;
	}
	/* Need to make sure we don't have a DONT_MIGRATE (very rare, someone
	 * would have to steal from us to get us to handle a preempt message,
	 * and then had to finish stealing (and fail) fast enough for us to miss
	 * the previous check). */
	if (current_uthread->flags & UTHREAD_DONT_MIGRATE) {
		__change_vcore(rem_vcoreid, FALSE);	/* returns on success */
		return;
	}
	/* Now save our uthread and restart them */
	assert(current_uthread);
	__uthread_pause(vcpd, current_uthread, TRUE);
	current_uthread = 0;
	were_handling_remotes = ev_might_not_return();
	__change_vcore(rem_vcoreid, TRUE);		/* noreturn on success*/
	/* Fall-through to out_we_returned */
out_we_returned:
	ev_we_returned(were_handling_remotes);
}

/* This handles a preemption message.  When this is done, either we recovered,
 * or recovery *for our message* isn't needed. */
static void handle_vc_preempt(struct event_msg *ev_msg, unsigned int ev_type,
                              void *data)
{
	uint32_t vcoreid = vcore_id();
	struct preempt_data *vcpd = vcpd_of(vcoreid);
	uint32_t rem_vcoreid = ev_msg->ev_arg2;
	struct preempt_data *rem_vcpd = vcpd_of(rem_vcoreid);
	struct uthread *uthread_to_steal = 0;
	struct uthread **rem_cur_uth;
	bool cant_migrate = FALSE;

	assert(in_vcore_context());
	/* Just drop messages about ourselves.  They are old.  If we happen to
	 * be getting preempted right now, there's another message out there
	 * about that. */
	if (rem_vcoreid == vcoreid)
		return;
	printd("Vcore %d was preempted (i'm %d), it's flags %08p!\n",
	       ev_msg->ev_arg2, vcoreid, rem_vcpd->flags);
	/* Spin til the kernel is done with flags.  This is how we avoid
	 * handling the preempt message before the preemption. */
	while (atomic_read(&rem_vcpd->flags) & VC_K_LOCK)
		cpu_relax();
	/* If they aren't preempted anymore, just return (optimization). */
	if (!(atomic_read(&rem_vcpd->flags) & VC_PREEMPTED))
		return;
	/* At this point, we need to try to recover */
	/* This case handles when the remote core was in vcore context */
	if (rem_vcpd->notif_disabled) {
		printd("VC %d recovering %d, notifs were disabled\n", vcoreid,
		       rem_vcoreid);
		change_to_vcore(vcpd, rem_vcoreid);
		return;	/* in case it returns.  we've done our job recovering */
	}
	/* So now it looks like they were not in vcore context.  We want to
	 * steal the uthread.  Set stealing, then doublecheck everything.  If
	 * stealing fails, someone else is stealing and we can just leave.  That
	 * other vcore who is stealing will check the VCPD/INDIRs when it is
	 * done. */
	if (!start_uth_stealing(rem_vcpd))
		return;
	/* Now we're stealing.  Double check everything.  A change in preempt
	 * status or notif_disable status means the vcore has since restarted.
	 * The vcore may or may not have started after we set STEALING.  If it
	 * didn't, we'll need to bail out (but still check messages, since above
	 * we assumed the uthread stealer handles the VCPD/INDIRs).  Since the
	 * vcore is running, we don't need to worry about handling the message
	 * any further.  Future preemptions will generate another message, so we
	 * can ignore getting the uthread or anything like that. */
	printd("VC %d recovering %d, trying to steal uthread\n", vcoreid,
	       rem_vcoreid);
	if (!(atomic_read(&rem_vcpd->flags) & VC_PREEMPTED))
		goto out_stealing;
	/* Might be preempted twice quickly, and the second time had notifs
	 * disabled.
	 *
	 * Also note that the second preemption event had another message sent,
	 * which either we or someone else will deal with.  And also, we don't
	 * need to worry about how we are stealing still and plan to abort.  If
	 * another vcore handles that second preemption message, either the
	 * original vcore is in vc ctx or not.  If so, we bail out and the
	 * second preemption handling needs to change_to.  If not, we aren't
	 * bailing out, and we'll handle the preemption as normal, and the
	 * second handler will bail when it fails to steal. */
	if (rem_vcpd->notif_disabled)
		goto out_stealing;
	/* At this point, we're clear to try and steal the uthread.  We used to
	 * switch to their TLS to steal the uthread, but we can access their
	 * current_uthread directly. */
	rem_cur_uth = get_tlsvar_linaddr(rem_vcoreid, current_uthread);
	uthread_to_steal = *rem_cur_uth;
	if (uthread_to_steal) {
		/* Extremely rare: they have a uthread, but it can't migrate.
		 * So we'll need to change to them. */
		if (uthread_to_steal->flags & UTHREAD_DONT_MIGRATE) {
			printd("VC %d recovering %d, can't migrate uthread!\n",
			       vcoreid, rem_vcoreid);
			stop_uth_stealing(rem_vcpd);
			change_to_vcore(vcpd, rem_vcoreid);
			/* in case it returns.  we've done our job recovering */
			return;
		} else {
			*rem_cur_uth = 0;
			/* we're clear to steal it */
			printd("VC %d recovering %d, uthread %08p stolen\n",
			       vcoreid, rem_vcoreid, uthread_to_steal);
			__uthread_pause(rem_vcpd, uthread_to_steal, FALSE);
			/* can't let the cur_uth = 0 write and any writes from
			 * __uth_pause() to pass stop_uth_stealing. */
			wmb();
		}
	}
	/* Fallthrough */
out_stealing:
	stop_uth_stealing(rem_vcpd);
	handle_indirs(rem_vcoreid);
}

/* This handles a "check indirs" message.  When this is done, either we checked
 * their indirs, or the vcore restarted enough so that checking them is
 * unnecessary.  If that happens and they got preempted quickly, then another
 * preempt/check_indirs was sent out. */
static void handle_vc_indir(struct event_msg *ev_msg, unsigned int ev_type,
                            void *data)
{
	uint32_t vcoreid = vcore_id();
	uint32_t rem_vcoreid = ev_msg->ev_arg2;

	if (rem_vcoreid == vcoreid)
		return;
	handle_indirs(rem_vcoreid);
}

static inline bool __uthread_has_tls(struct uthread *uthread)
{
	return uthread->tls_desc != UTH_TLSDESC_NOTLS;
}

/* TLS helpers */
static int __uthread_allocate_tls(struct uthread *uthread)
{
	assert(!uthread->tls_desc);
	uthread->tls_desc = allocate_tls();
	if (!uthread->tls_desc) {
		errno = ENOMEM;
		return -1;
	}
	return 0;
}

static int __uthread_reinit_tls(struct uthread *uthread)
{
	uthread->tls_desc = reinit_tls(uthread->tls_desc);
	if (!uthread->tls_desc) {
		errno = ENOMEM;
		return -1;
	}
	return 0;
}

static void __uthread_free_tls(struct uthread *uthread)
{
	free_tls(uthread->tls_desc);
	uthread->tls_desc = NULL;
}

bool uth_2ls_is_multithreaded(void)
{
	/* thread 0 is single threaded.  For the foreseeable future, every other
	 * 2LS will be multithreaded. */
	extern struct schedule_ops thread0_2ls_ops;

	return sched_ops != &thread0_2ls_ops;
}

struct uthread *uthread_create(void *(*func)(void *), void *arg)
{
	return sched_ops->thread_create(func, arg);
}

/* Who does the thread_exited callback (2LS-specific cleanup)?  It depends.  If
 * the thread exits first, then join/detach does it.  o/w, the exit path does.
 *
 * What are the valid state changes?
 *
 * 	JOINABLE   -> DETACHED (only by detach())
 * 	JOINABLE   -> HAS_JOINER (only by join())
 * 	JOINABLE   -> EXITED (only by uth_2ls_thread_exit())
 *
 * That's it.  The initial state is either JOINABLE or DETACHED. */
void uthread_detach(struct uthread *uth)
{
	struct uth_join_ctl *jc = &uth->join_ctl;
	long old_state;

	do {
		old_state = atomic_read(&jc->state);
		switch (old_state) {
		case UTH_JOIN_EXITED:
			sched_ops->thread_exited(uth);
			return;
		case UTH_JOIN_DETACHED:
			panic("Uth %p has already been detached!", uth);
		case UTH_JOIN_HAS_JOINER:
			panic("Uth %p has a pending joiner, can't detach!",
			      uth);
		};
		assert(old_state == UTH_JOIN_JOINABLE);
	} while (!atomic_cas(&jc->state, old_state, UTH_JOIN_DETACHED));
}

/* Helper.  We have a joiner.  So we'll write the retval to the final location
 * (the one passed to join() and decref to wake the joiner.  This may seem a
 * little odd for a normal join, but it works identically a parallel join - and
 * there's only one wakeup (hence the kref). */
static void uth_post_and_kick_joiner(struct uthread *uth, void *retval)
{
	struct uth_join_ctl *jc = &uth->join_ctl;

	if (jc->retval_loc)
		*jc->retval_loc = retval;
	/* Note the JC has a pointer to the kicker.  There's one kicker for the
	 * joiner, but there could be many joinees. */
	kref_put(&jc->kicker->kref);
}

/* Callback after the exiting uthread has yielded and is in vcore context.  Note
 * that the thread_exited callback can be called concurrently (e.g., a racing
 * call to detach()), so it's important to not be in the uthread's context. */
static void __uth_2ls_thread_exit_cb(struct uthread *uth, void *retval)
{
	struct uth_join_ctl *jc = &uth->join_ctl;
	long old_state;

	do {
		old_state = atomic_read(&jc->state);
		switch (old_state) {
		case UTH_JOIN_DETACHED:
			sched_ops->thread_exited(uth);
			return;
		case UTH_JOIN_HAS_JOINER:
			uth_post_and_kick_joiner(uth, retval);
			sched_ops->thread_exited(uth);
			return;
		case UTH_JOIN_JOINABLE:
			/* This write is harmless and idempotent; we can lose
			 * the race and still be safe.  Assuming we don't, the
			 * joiner will look here for the retval.  It's temporary
			 * storage since we don't know the final retval location
			 * (since join hasn't happened yet). */
			jc->retval = retval;
			break;
		};
		assert(old_state == UTH_JOIN_JOINABLE);
	} while (!atomic_cas(&jc->state, old_state, UTH_JOIN_EXITED));
	/* We were joinable, now we have exited.  A detacher or joiner will
	 * trigger thread_exited. */
}

/* 2LSs call this when their threads are exiting.  The 2LS will regain control
 * of the thread in sched_ops->thread_exited.  This will be after the
 * join/detach/exit has completed, and might be in vcore context. */
void __attribute__((noreturn)) uth_2ls_thread_exit(void *retval)
{
	uthread_yield(FALSE, __uth_2ls_thread_exit_cb, retval);
	assert(0);
}

/* Helper: Attaches the caller (specifically the jk) to the target uthread.
 * When the thread has been joined (either due to the UTH_EXITED case or due to
 * __uth_2ls_thread_exit_cb), the join kicker will be decreffed. */
static void join_one(struct uthread *uth, struct uth_join_kicker *jk,
                     void **retval_loc)
{
	struct uth_join_ctl *jc = &uth->join_ctl;
	long old_state;

	/* We can safely write to the join_ctl, even if we don't end up setting
	 * HAS_JOINER.  There's only supposed to be one joiner, and if not,
	 * we'll catch the bad state. */
	jc->retval_loc = retval_loc;
	jc->kicker = jk;
	do {
		old_state = atomic_read(&jc->state);
		switch (old_state) {
		case UTH_JOIN_EXITED:
			if (retval_loc)
				*retval_loc = jc->retval;
			sched_ops->thread_exited(uth);
			kref_put(&jk->kref);
			return;
		case UTH_JOIN_DETACHED:
			panic("Uth %p has been detached, can't join!", uth);
		case UTH_JOIN_HAS_JOINER:
			panic("Uth %p has another pending joiner!", uth);
		};
		assert(old_state == UTH_JOIN_JOINABLE);
	} while (!atomic_cas(&jc->state, old_state, UTH_JOIN_HAS_JOINER));
}

/* Bottom half of the join, in vcore context */
static void __uth_join_cb(struct uthread *uth, void *arg)
{
	struct uth_join_kicker *jk = (struct uth_join_kicker*)arg;

	uthread_has_blocked(uth, UTH_EXT_BLK_MISC);
	/* After this, and after all threads join, we could be woken up. */
	kref_put(&jk->kref);
}

static void kicker_release(struct kref *k)
{
	struct uth_join_kicker *jk = container_of(k, struct uth_join_kicker,
						  kref);

	uthread_runnable(jk->joiner);
}

void uthread_join_arr(struct uth_join_request reqs[], size_t nr_req)
{
	struct uth_join_kicker jk[1];

	jk->joiner = current_uthread;
	/* One ref for each target, another for *us*, which we drop in the yield
	 * callback.  As as soon as it is fully decreffed, our thread will be
	 * restarted.  We must block before that (in the yield callback). */
	kref_init(&jk->kref, kicker_release, nr_req + 1);
	for (int i = 0; i < nr_req; i++)
		join_one(reqs[i].uth, jk, reqs[i].retval_loc);
	uthread_yield(TRUE, __uth_join_cb, jk);
}

/* Unlike POSIX, we don't bother with returning error codes.  Anything that can
 * go wrong is so horrendous that you should crash (the specs say the behavior
 * is undefined). */
void uthread_join(struct uthread *uth, void **retval_loc)
{
	struct uth_join_request req[1];

	req->uth = uth;
	req->retval_loc = retval_loc;
	uthread_join_arr(req, 1);
}

static void __uth_sched_yield_cb(struct uthread *uth, void *arg)
{
	uthread_has_blocked(uth, UTH_EXT_BLK_YIELD);
	uthread_runnable(uth);
}

void uthread_sched_yield(void)
{
	if (!uth_2ls_is_multithreaded()) {
		/* We're an SCP with no other threads, so we want to yield to
		 * other processes.  For SCPs, this will yield to the OS/other
		 * procs. */
		syscall(SYS_proc_yield, TRUE);
		return;
	}
	uthread_yield(TRUE, __uth_sched_yield_cb, NULL);
}

struct uthread *uthread_self(void)
{
	return current_uthread;
}
