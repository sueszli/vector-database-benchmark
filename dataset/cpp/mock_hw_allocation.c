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

#include "mock_hw_allocation.h"

#include "mock_test.h"

int
__wrap_hw_alloc_assoc_set(const unsigned lcore, const unsigned class_id)
{
        check_expected(lcore);
        check_expected(class_id);

        return mock_type(int);
}

int
__wrap_hw_alloc_assoc_get(const unsigned lcore, unsigned *class_id)
{
        int ret;

        check_expected(lcore);
        check_expected_ptr(class_id);

        ret = mock_type(int);
        if (ret == PQOS_RETVAL_OK)
                *class_id = mock_type(int);

        return ret;
}

int
__wrap_hw_alloc_assign(const unsigned technology,
                       const unsigned *core_array,
                       const unsigned core_num,
                       unsigned *class_id)
{
        int ret;

        check_expected(technology);
        check_expected_ptr(core_array);
        check_expected(core_num);
        check_expected_ptr(class_id);

        ret = mock_type(int);
        if (ret == PQOS_RETVAL_OK)
                *class_id = mock_type(int);

        return ret;
}

int
__wrap_hw_alloc_release(const unsigned *core_array, const unsigned core_num)
{
        check_expected_ptr(core_array);
        check_expected(core_num);

        return mock_type(int);
}

int
__wrap_hw_alloc_reset(const struct pqos_alloc_config *cfg)
{
        check_expected_ptr(cfg);

        return mock_type(int);
}

int
__wrap_hw_l3ca_set(const unsigned l3cat_id,
                   const unsigned num_cos,
                   const struct pqos_l3ca *ca)
{
        check_expected(l3cat_id);
        check_expected(num_cos);
        check_expected_ptr(ca);

        return mock_type(int);
}

int
__wrap_hw_l3ca_get(const unsigned l3cat_id,
                   const unsigned max_num_ca,
                   unsigned *num_ca,
                   struct pqos_l3ca *ca)
{
        check_expected(l3cat_id);
        check_expected(max_num_ca);
        check_expected_ptr(num_ca);
        check_expected_ptr(ca);

        return mock_type(int);
}

int
__wrap_hw_l3ca_get_min_cbm_bits(unsigned *min_cbm_bits)
{
        check_expected_ptr(min_cbm_bits);

        return mock_type(int);
}

int
__wrap_hw_l2ca_set(const unsigned l2id,
                   const unsigned num_cos,
                   const struct pqos_l2ca *ca)
{
        check_expected(l2id);
        check_expected(num_cos);
        check_expected_ptr(ca);

        return mock_type(int);
}

int
__wrap_hw_l2ca_get(const unsigned l2id,
                   const unsigned max_num_ca,
                   unsigned *num_ca,
                   struct pqos_l2ca *ca)
{
        check_expected(l2id);
        check_expected(max_num_ca);
        check_expected_ptr(num_ca);
        check_expected_ptr(ca);

        return mock_type(int);
}

int
__wrap_hw_l2ca_get_min_cbm_bits(unsigned *min_cbm_bits)
{
        check_expected_ptr(min_cbm_bits);

        return mock_type(int);
}

int
__wrap_hw_mba_set(const unsigned mba_id,
                  const unsigned num_cos,
                  const struct pqos_mba *requested,
                  struct pqos_mba *actual)
{
        check_expected(mba_id);
        check_expected(num_cos);
        check_expected_ptr(requested);
        check_expected_ptr(actual);

        return mock_type(int);
}

int
__wrap_hw_mba_get(const unsigned mba_id,
                  const unsigned max_num_cos,
                  unsigned *num_cos,
                  struct pqos_mba *mba_tab)
{
        check_expected(mba_id);
        check_expected(max_num_cos);
        check_expected_ptr(num_cos);
        check_expected_ptr(mba_tab);

        return mock_type(int);
}

int
__wrap_hw_alloc_assoc_write(const unsigned lcore, const unsigned class_id)
{
        check_expected(lcore);
        check_expected(class_id);

        return mock_type(int);
}

int
__wrap_hw_alloc_assoc_read(const unsigned lcore, unsigned *class_id)
{
        check_expected(lcore);
        *class_id = mock_type(int);

        return mock_type(int);
}

int
__wrap_hw_alloc_assoc_get_channel(const pqos_channel_t channel,
                                  unsigned *class_id)
{
        int ret;

        check_expected(channel);
        check_expected_ptr(class_id);

        ret = mock_type(int);
        if (ret == PQOS_RETVAL_OK)
                *class_id = mock_type(int);

        return ret;
}

int
__wrap_hw_alloc_assoc_get_dev(const uint16_t segment,
                              const uint16_t bdf,
                              const unsigned vc,
                              unsigned *class_id)
{
        int ret;

        check_expected(segment);
        check_expected(bdf);
        check_expected(vc);
        check_expected_ptr(class_id);

        ret = mock_type(int);
        if (ret == PQOS_RETVAL_OK)
                *class_id = mock_type(int);

        return ret;
}

int
__wrap_hw_alloc_assoc_set_channel(const pqos_channel_t channel,
                                  const unsigned class_id)
{
        check_expected(channel);
        check_expected(class_id);

        return mock_type(int);
}

int
__wrap_hw_alloc_assoc_set_dev(const uint16_t segment,
                              const uint16_t bdf,
                              const unsigned vc,
                              const unsigned class_id)
{
        check_expected(segment);
        check_expected(bdf);
        check_expected(vc);
        check_expected(class_id);

        return mock_type(int);
}
