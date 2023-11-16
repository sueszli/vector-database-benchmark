/**
 * @file
 *
 * @brief
 *
 * @date 15.05.2012
 * @author Anton Bondarev
 */

#include <embox/unit.h>
#include <mem/misc/pool.h>

#include <kernel/time/itimer.h>
#include <hal/clock.h>

POOL_DEF(itimer_pool, struct itimer, OPTION_GET(NUMBER, itimer_quantity));

struct itimer *itimer_alloc(void) {
	return pool_alloc(&itimer_pool);
}

void itimer_free(struct itimer *it) {
	pool_free(&itimer_pool, it);
}

void itimer_init(struct itimer *it, struct clock_source *cs,
		time64_t start_tstamp) {
	struct timespec ts;
	assert(it && cs);

	ts = ns_to_timespec(start_tstamp);
	it->cs = cs;
	it->start_value = timespec_add(ts, clock_source_read(it->cs));
}

time64_t itimer_read_ns(struct itimer *it) {
	struct timespec ts;
	itimer_read_timespec(it, &ts);
	return timespec_to_ns(&ts);
}

void itimer_read_timespec(struct itimer *it, struct timespec *ts) {
	assert(it && it->cs);
	*ts = timespec_sub(clock_source_read(it->cs), it->start_value);
}
