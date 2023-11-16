/*
 * BSD LICENSE
 *
 * Copyright(c) 2021 Intel Corporation. All rights reserved.
 * All rights reserved.
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
#include "test.h"

/* ======== mock ======== */

int
hw_alloc_assoc_write(const unsigned lcore, const unsigned class_id)
{
        function_called();
        check_expected(lcore);
        check_expected(class_id);

        return mock_type(int);
}

int
hw_alloc_assoc_set_channel(const pqos_channel_t channel,
                           const unsigned class_id)
{
        function_called();
        check_expected(channel);
        check_expected(class_id);

        return mock_type(int);
}

/* ======== hw_alloc_reset_assoc_core ======== */

static void
test_hw_alloc_reset_assoc_cores(void **state)
{
        struct test_data *data = (struct test_data *)*state;
        int ret;
        unsigned i;

        will_return_maybe(__wrap__pqos_get_cpu, data->cpu);

        for (i = 0; i < data->cpu->num_cores; i++) {
                unsigned lcore = data->cpu->cores[i].lcore;

                expect_function_call(hw_alloc_assoc_write);
                expect_value(hw_alloc_assoc_write, lcore, lcore);
                expect_value(hw_alloc_assoc_write, class_id, 0);
                will_return(hw_alloc_assoc_write, PQOS_RETVAL_OK);
        }

        ret = hw_alloc_reset_assoc_cores();
        assert_int_equal(ret, PQOS_RETVAL_OK);
}

static void
test_hw_alloc_reset_assoc_cores_error(void **state)
{
        struct test_data *data = (struct test_data *)*state;
        int ret;
        unsigned i;

        will_return_maybe(__wrap__pqos_get_cpu, data->cpu);

        for (i = 0; i < data->cpu->num_cores; i++) {
                unsigned lcore = data->cpu->cores[i].lcore;

                expect_function_call(hw_alloc_assoc_write);
                expect_value(hw_alloc_assoc_write, lcore, lcore);
                expect_value(hw_alloc_assoc_write, class_id, 0);
                will_return(hw_alloc_assoc_write, PQOS_RETVAL_ERROR);
        }

        ret = hw_alloc_reset_assoc_cores();
        assert_int_equal(ret, PQOS_RETVAL_ERROR);
}

/* ======== hw_alloc_reset_assoc_channels ======== */

static void
test_hw_alloc_reset_assoc_channels(void **state)
{
        struct test_data *data = (struct test_data *)*state;
        int ret;

        will_return_maybe(__wrap__pqos_get_cap, data->cap);
        will_return_maybe(__wrap__pqos_get_dev, data->dev);

        expect_function_call(__wrap_pqos_l3ca_iordt_enabled);
        expect_value(__wrap_pqos_l3ca_iordt_enabled, cap, data->cap);
        will_return(__wrap_pqos_l3ca_iordt_enabled, PQOS_RETVAL_OK);
        will_return(__wrap_pqos_l3ca_iordt_enabled, 1);
        will_return(__wrap_pqos_l3ca_iordt_enabled, 1);

        expect_function_call(__wrap_iordt_assoc_reset);
        will_return(__wrap_iordt_assoc_reset, PQOS_RETVAL_OK);

        ret = hw_alloc_reset_assoc_channels();
        assert_int_equal(ret, PQOS_RETVAL_OK);
}

static void
test_hw_alloc_reset_assoc_channels_error(void **state)
{
        struct test_data *data = (struct test_data *)*state;
        int ret;

        will_return_maybe(__wrap__pqos_get_cap, data->cap);
        will_return_maybe(__wrap__pqos_get_dev, data->dev);

        /* pqos_l3ca_iordt_enabled failed */
        expect_function_call(__wrap_pqos_l3ca_iordt_enabled);
        expect_value(__wrap_pqos_l3ca_iordt_enabled, cap, data->cap);
        will_return(__wrap_pqos_l3ca_iordt_enabled, PQOS_RETVAL_ERROR);

        ret = hw_alloc_reset_assoc_channels();
        assert_int_not_equal(ret, PQOS_RETVAL_OK);

        /* hw_alloc_assoc_set_channel fails!*/
        expect_function_call(__wrap_pqos_l3ca_iordt_enabled);
        expect_value(__wrap_pqos_l3ca_iordt_enabled, cap, data->cap);
        will_return(__wrap_pqos_l3ca_iordt_enabled, PQOS_RETVAL_OK);
        will_return(__wrap_pqos_l3ca_iordt_enabled, 1);
        will_return(__wrap_pqos_l3ca_iordt_enabled, 1);

        expect_function_call(__wrap_iordt_assoc_reset);
        will_return(__wrap_iordt_assoc_reset, PQOS_RETVAL_ERROR);

        ret = hw_alloc_reset_assoc_channels();
        assert_int_not_equal(ret, PQOS_RETVAL_OK);
}

int
main(void)
{
        int result = 0;

        const struct CMUnitTest tests[] = {
            cmocka_unit_test(test_hw_alloc_reset_assoc_cores),
            cmocka_unit_test(test_hw_alloc_reset_assoc_cores_error),
            cmocka_unit_test(test_hw_alloc_reset_assoc_channels),
            cmocka_unit_test(test_hw_alloc_reset_assoc_channels_error)};

        result += cmocka_run_group_tests(tests, test_init_l3ca, test_fini);

        return result;
}
