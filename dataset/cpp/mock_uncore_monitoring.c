/*
 * BSD LICENSE
 *
 * Copyright(c) 2022-2023 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "mock_uncore_monitoring.h"

#include "mock_test.h"

#include <string.h>

int
__wrap_uncore_mon_discover(enum pqos_mon_event *event)
{
        *event = 0;
        return PQOS_RETVAL_OK;
}

int
__wrap_uncore_mon_init(const struct pqos_cpuinfo *cpu,
                       const struct pqos_cap *cap)
{
        assert_non_null(cpu);
        assert_non_null(cap);

        return PQOS_RETVAL_RESOURCE;
}

int
__wrap_uncore_mon_fini(void)
{
        return PQOS_RETVAL_OK;
}

int
__wrap_uncore_mon_start(struct pqos_mon_data *group,
                        const enum pqos_mon_event event)
{
        assert_non_null(group);
        assert_int_not_equal(event, 0);

        return PQOS_RETVAL_OK;
}

int
__wrap_uncore_mon_stop(struct pqos_mon_data *group)
{
        assert_non_null(group);

        return PQOS_RETVAL_OK;
}

int
__wrap_uncore_mon_poll(struct pqos_mon_data *group,
                       const enum pqos_mon_event event)
{
        assert_non_null(group);
        assert_int_not_equal(event, 0);

        return PQOS_RETVAL_OK;
}

int
__wrap_uncore_mon_is_event_supported(const enum pqos_mon_event event)
{
        assert_int_not_equal(event, 0);

        return 0;
}
