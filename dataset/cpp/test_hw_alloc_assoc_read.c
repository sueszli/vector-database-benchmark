/*
 * BSD LICENSE
 *
 * Copyright(c) 2020-2023 Intel Corporation. All rights reserved.
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

#include "allocation.h"
#include "cpu_registers.h"
#include "test.h"

/* ======== hw_alloc_assoc_read ======== */

static void
test_hw_alloc_assoc_read(void **state __attribute__((unused)))
{
        int ret;
        unsigned class_id = 0;
        unsigned lcore = 2;

        expect_value(__wrap_msr_read, lcore, lcore);
        expect_value(__wrap_msr_read, reg, PQOS_MSR_ASSOC);
        will_return(__wrap_msr_read, PQOS_RETVAL_OK);
        will_return(__wrap_msr_read, ((uint64_t)2)
                                         << PQOS_MSR_ASSOC_QECOS_SHIFT);

        ret = hw_alloc_assoc_read(lcore, &class_id);
        assert_int_equal(ret, PQOS_RETVAL_OK);
        assert_int_equal(class_id, 2);
}

static void
test_hw_alloc_assoc_read_param(void **state __attribute__((unused)))
{
        int ret;
        unsigned lcore = 2;

        ret = hw_alloc_assoc_read(lcore, NULL);
        assert_int_equal(ret, PQOS_RETVAL_PARAM);
}

static void
test_hw_alloc_assoc_read_error(void **state __attribute__((unused)))
{
        int ret;
        unsigned class_id = 0;
        unsigned lcore = 2;

        expect_value(__wrap_msr_read, lcore, lcore);
        expect_value(__wrap_msr_read, reg, PQOS_MSR_ASSOC);
        will_return(__wrap_msr_read, PQOS_RETVAL_ERROR);

        ret = hw_alloc_assoc_read(lcore, &class_id);
        assert_int_equal(ret, PQOS_RETVAL_ERROR);
        assert_int_equal(class_id, 0);
}

int
main(void)
{
        int result = 0;

        const struct CMUnitTest tests[] = {
            cmocka_unit_test(test_hw_alloc_assoc_read),
            cmocka_unit_test(test_hw_alloc_assoc_read_param),
            cmocka_unit_test(test_hw_alloc_assoc_read_error)};

        result += cmocka_run_group_tests(tests, NULL, NULL);

        return result;
}
