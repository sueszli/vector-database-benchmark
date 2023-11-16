/*
 * ngtcp2
 *
 * Copyright (c) 2017 ngtcp2 contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "ngtcp2_range_test.h"

#include <stdio.h>

#include <CUnit/CUnit.h>

#include "ngtcp2_range.h"
#include "ngtcp2_test_helper.h"

void test_ngtcp2_range_intersect(void) {
  ngtcp2_range a, b, c;

  ngtcp2_range_init(&a, 0, UINT64_MAX);
  ngtcp2_range_init(&b, 0, 1000000007);
  c = ngtcp2_range_intersect(&a, &b);

  CU_ASSERT(0 == c.begin);
  CU_ASSERT(1000000007 == c.end);

  ngtcp2_range_init(&a, 1000000007, UINT64_MAX);
  ngtcp2_range_init(&b, 0, UINT64_MAX);
  c = ngtcp2_range_intersect(&a, &b);

  CU_ASSERT(1000000007 == c.begin);
  CU_ASSERT(UINT64_MAX == c.end);

  ngtcp2_range_init(&a, 0, UINT64_MAX);
  ngtcp2_range_init(&b, 33333, 55555);
  c = ngtcp2_range_intersect(&a, &b);

  CU_ASSERT(33333 == c.begin);
  CU_ASSERT(55555 == c.end);

  ngtcp2_range_init(&a, 0, 1000000009);
  ngtcp2_range_init(&b, 1000000007, UINT64_MAX);
  c = ngtcp2_range_intersect(&a, &b);

  CU_ASSERT(1000000007 == c.begin);
  CU_ASSERT(1000000009 == c.end);
}

void test_ngtcp2_range_cut(void) {
  ngtcp2_range a, b, l, r;

  ngtcp2_range_init(&a, 0, UINT64_MAX);
  ngtcp2_range_init(&b, 1000000007, 1000000009);
  ngtcp2_range_cut(&l, &r, &a, &b);

  CU_ASSERT(0 == l.begin);
  CU_ASSERT(1000000007 == l.end);
  CU_ASSERT(1000000009 == r.begin);
  CU_ASSERT(UINT64_MAX == r.end);

  ngtcp2_range_init(&a, 0, UINT64_MAX);
  ngtcp2_range_init(&b, 0, 1000000007);
  ngtcp2_range_cut(&l, &r, &a, &b);

  CU_ASSERT(0 == ngtcp2_range_len(&l));
  CU_ASSERT(1000000007 == r.begin);
  CU_ASSERT(UINT64_MAX == r.end);

  ngtcp2_range_init(&a, 0, UINT64_MAX);
  ngtcp2_range_init(&b, 1000000009, UINT64_MAX);
  ngtcp2_range_cut(&l, &r, &a, &b);

  CU_ASSERT(0 == l.begin);
  CU_ASSERT(1000000009 == l.end);
  CU_ASSERT(0 == ngtcp2_range_len(&r));
}

void test_ngtcp2_range_not_after(void) {
  ngtcp2_range a, b;

  ngtcp2_range_init(&a, 1, 1000000007);
  ngtcp2_range_init(&b, 0, 1000000007);

  CU_ASSERT(ngtcp2_range_not_after(&a, &b));

  ngtcp2_range_init(&a, 1, 1000000008);
  ngtcp2_range_init(&b, 0, 1000000007);

  CU_ASSERT(!ngtcp2_range_not_after(&a, &b));
}
