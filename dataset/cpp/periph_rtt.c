/*
 * Copyright (C) 2020 Kaspar Schleiser <kaspar@schleiser.de>
 *               2020 Freie Universität Berlin
 *               2020 Inria
 *
 * This file is subject to the terms and conditions of the GNU Lesser General
 * Public License v2.1. See the file LICENSE in the top level directory for more
 * details.
 */

/**
 * @ingroup     sys_ztimer_periph_rtt
 * @{
 *
 * @file
 * @brief       ztimer periph/rtt implementation
 *
 * @author      Kaspar Schleiser <kaspar@schleiser.de>
 *
 * @}
 */
#include "assert.h"

#include "irq.h"
#include "periph/rtt.h"
#include "ztimer/periph_rtt.h"

#define ENABLE_DEBUG 0
#include "debug.h"

static void _ztimer_periph_rtt_callback(void *arg)
{
    ztimer_handler((ztimer_clock_t *)arg);
}

static void _ztimer_periph_rtt_set(ztimer_clock_t *clock, uint32_t val)
{
    if (val < RTT_MIN_OFFSET) {
        /* the rtt might advance right between the call to rtt_get_counter()
         * and rtt_set_alarm(). If that happens with val==1, we'd set an alarm
         * to the current time, which would then underflow.  To avoid this, we
         * set the alarm at least two ticks in the future.  TODO: confirm this
         * is sufficient, or conceive logic to lower this value.
         *
         * @note RTT_MIN_OFFSET defaults to 2, but some platforms might have
         * different values.
         */
        val = RTT_MIN_OFFSET;
    }

    unsigned state = irq_disable();

    /* ensure RTT_MAX_VALUE is (2^n - 1) */
    static_assert((RTT_MAX_VALUE == UINT32_MAX) ||
                  !((RTT_MAX_VALUE + 1) & (RTT_MAX_VALUE)),
                  "RTT_MAX_VALUE needs to be (2^n) - 1");

    rtt_set_alarm(
        (rtt_get_counter() + val) & RTT_MAX_VALUE, _ztimer_periph_rtt_callback,
        clock);

    irq_restore(state);
}

static uint32_t _ztimer_periph_rtt_now(ztimer_clock_t *clock)
{
    (void)clock;
    return rtt_get_counter();
}

static void _ztimer_periph_rtt_cancel(ztimer_clock_t *clock)
{
    (void)clock;
    rtt_clear_alarm();
}

#if MODULE_ZTIMER_ONDEMAND_RTT
static void _ztimer_periph_rtt_start(ztimer_clock_t *clock)
{
    (void)clock;
    rtt_poweron();
}

static void _ztimer_periph_rtt_stop(ztimer_clock_t *clock)
{
    (void)clock;
    rtt_poweroff();
}
#endif /* MODULE_ZTIMER_ONDEMAND_RTT */

static const ztimer_ops_t _ztimer_periph_rtt_ops = {
    .set = _ztimer_periph_rtt_set,
    .now = _ztimer_periph_rtt_now,
    .cancel = _ztimer_periph_rtt_cancel,
#if MODULE_ZTIMER_ONDEMAND_RTT
    .start = _ztimer_periph_rtt_start,
    .stop = _ztimer_periph_rtt_stop,
#endif
};

void ztimer_periph_rtt_init(ztimer_periph_rtt_t *clock)
{
    clock->ops = &_ztimer_periph_rtt_ops;
    clock->max_value = RTT_MAX_VALUE;
    rtt_init();
#if MODULE_ZTIMER_ONDEMAND_RTT
    rtt_poweroff();
#else
    rtt_poweron();
#endif

#if !MODULE_ZTIMER_ONDEMAND
    ztimer_init_extend(clock);
#endif
}
