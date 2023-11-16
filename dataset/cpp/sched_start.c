/**
 * @file
 * @brief
 *
 * @author  Anton Kozlov
 * @date    26.08.2014
 */

#include <assert.h>

#include <util/log.h>

#include <embox/unit.h>
#include <kernel/sched.h>
#include <kernel/time/clock_source.h>

EMBOX_UNIT_INIT(sched_start_init);

extern struct schedee *boot_thread_create(void);
extern int idle_thread_create(void);

static int sched_start_init(void) {
	struct schedee *current;
	int err;

	current = boot_thread_create(); /* 'init' thread ID=1 */
	assert(current != NULL);
	log_debug("boot_thread = 0x%p", current);

	err = sched_init(current);
	if (err) {
		return err;
	}

	err = idle_thread_create(); /* idle thread always has ID=0 */

	jiffies_init();

	return err;
}
