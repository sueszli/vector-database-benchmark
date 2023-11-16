/**
 * Copyright (c) 2016 DeepCortex GmbH <legal@eventql.io>
 * Authors:
 *   - Paul Asmuth <paul@eventql.io>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License ("the license") as
 * published by the Free Software Foundation, either version 3 of the License,
 * or any later version.
 *
 * In accordance with Section 7(e) of the license, the licensing of the Program
 * under the license does not imply a trademark license. Therefore any rights,
 * title and interest in our trademarks remain entirely with us.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the license for more details.
 *
 * You can be released from the requirements of the license by purchasing a
 * commercial license. Buying such a license is mandatory as soon as you develop
 * commercial activities involving this program without disclosing the source
 * code of your own applications
 */
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <eventql/sql/expressions/boolean.h>
#include <eventql/util/exception.h>

namespace csql {
namespace expressions {


/**
 * logical_and(bool, bool) -> bool
 */
void logical_and_call(sql_txn* ctx, VMStack* stack) {
  auto right = popBool(stack);
  auto left = popBool(stack);
  pushBool(stack, left && right);
}

const SFunction logical_and(
    { SType::BOOL, SType::BOOL },
    SType::BOOL,
    &logical_and_call);


/**
 * logical_or(bool, bool) -> bool
 */
void logical_or_call(sql_txn* ctx, VMStack* stack) {
  auto right = popBool(stack);
  auto left = popBool(stack);
  pushBool(stack, left || right);
}

const SFunction logical_or(
    { SType::BOOL, SType::BOOL },
    SType::BOOL,
    &logical_or_call);


/**
 * neg(bool) -> bool
 */
void neg_call(sql_txn* ctx, VMStack* stack) {
  auto arg = popBool(stack);
  pushBool(stack, !arg);
}

const SFunction neg(
    { SType::BOOL },
    SType::BOOL,
    &neg_call);


/**
 * cmp(uint64, uint64) -> int64
 * cmp(timestamp64, timestamp64) -> int64
 */
void cmp_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  if (left < right) {
    pushInt64(stack, -1);
  } else if (left > right) {
    pushInt64(stack, 1);
  } else {
    pushInt64(stack, 0);
  }
}

const SFunction cmp_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::INT64,
    &cmp_uint64_call);

const SFunction cmp_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::INT64,
    &cmp_uint64_call);


/**
 * cmp(int64, int64) -> int64
 */
void cmp_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  if (left < right) {
    pushInt64(stack, -1);
  } else if (left > right) {
    pushInt64(stack, 1);
  } else {
    pushInt64(stack, 0);
  }
}

const SFunction cmp_int64(
    { SType::INT64, SType::INT64 },
    SType::INT64,
    &cmp_int64_call);


/**
 * cmp(float64, float64) -> int64
 */
void cmp_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  if (left < right) {
    pushInt64(stack, -1);
  } else if (left > right) {
    pushInt64(stack, 1);
  } else {
    pushInt64(stack, 0);
  }
}

const SFunction cmp_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::INT64,
    &cmp_float64_call);


/**
 * cmp(string, string) -> int64
 */
void cmp_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  auto cmp = strncmp(left, right, std::min(left_len, right_len));
  if (left_len == right_len && cmp == 0) {
    pushInt64(stack, 0);
  } else if (cmp < 0 || (cmp == 0 && left_len < right_len)) {
    pushInt64(stack, -1);
  } else if (cmp > 0 || (cmp == 0 && left_len > right_len)) {
    pushInt64(stack, 1);
  }
}

const SFunction cmp_string(
    { SType::STRING, SType::STRING },
    SType::INT64,
    &cmp_string_call);


/**
 * eq(uint64, uint64) -> bool
 * eq(timestamp64, timestamp64) -> bool
 */
void eq_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left == right);
}

const SFunction eq_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &eq_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction eq_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &eq_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

/**
 * eq(int64, int64) -> bool
 */
void eq_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left == right);
}

const SFunction eq_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &eq_int64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * eq(float64, float64) -> bool
 */
void eq_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left == right);
}

const SFunction eq_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &eq_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * eq(string, string) -> bool
 */
void eq_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  if (left_len != right_len) {
    pushBool(stack, false);
  } else {
    pushBool(stack, memcmp(left, right, left_len) == 0);
  }
}

const SFunction eq_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &eq_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * eq(bool, bool) -> bool
 */
void eq_bool_call(sql_txn* ctx, VMStack* stack) {
  auto right = popBool(stack);
  auto left = popBool(stack);
  pushBool(stack, left == right);
}

const SFunction eq_bool(
    { SType::BOOL, SType::BOOL },
    SType::BOOL,
    &eq_bool_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * neq(uint64, uint64) -> bool
 * neq(timestamp64, timestamp64) -> bool
 */
void neq_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left != right);
}

const SFunction neq_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &neq_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction neq_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &neq_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * neq(int64, int64) -> bool
 */
void neq_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left != right);
}

const SFunction neq_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &neq_int64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * neq(float64, float64) -> bool
 */
void neq_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left != right);
}

const SFunction neq_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &neq_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * neq(bool, bool) -> bool
 */
void neq_bool_call(sql_txn* ctx, VMStack* stack) {
  auto right = popBool(stack);
  auto left = popBool(stack);
  pushBool(stack, left != right);
}

const SFunction neq_bool(
    { SType::BOOL, SType::BOOL },
    SType::BOOL,
    &neq_bool_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * eq(string, string) -> bool
 */
void neq_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  if (left_len != right_len) {
    pushBool(stack, true);
  } else {
    pushBool(stack, memcmp(left, right, left_len) != 0);
  }
}

const SFunction neq_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &neq_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lt(uint64, uint64) -> bool
 * lt(timestamp64, timestamp4) -> bool
 */
void lt_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left < right);
}

const SFunction lt_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &lt_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction lt_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &lt_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lt(int64, int64) -> bool
 */
void lt_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left < right);
}

const SFunction lt_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &lt_int64_call);


/**
 * lt(float64, float64) -> bool
 */
void lt_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left < right);
}

const SFunction lt_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &lt_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lt(string, string) -> bool
 */
void lt_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  auto cmp = strncmp(left, right, std::min(left_len, right_len));
  pushBool(stack, cmp < 0 || (cmp == 0 && left_len < right_len));
}

const SFunction lt_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &lt_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lte(uint64, uint64) -> bool
 * lte(timestamp64, timestamp64) -> bool
 */
void lte_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left <= right);
}

const SFunction lte_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &lte_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction lte_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &lte_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lte(int64, int64) -> bool
 */
void lte_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left <= right);
}

const SFunction lte_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &lte_int64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lte(float64, float64) -> bool
 */
void lte_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left <= right);
}

const SFunction lte_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &lte_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * lte(string, string) -> bool
 */
void lte_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  auto cmp = strncmp(left, right, std::min(left_len, right_len));
  pushBool(stack, cmp < 0 || (cmp == 0 && left_len <= right_len));
}

const SFunction lte_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &lte_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gt(uint64, uint64) -> bool
 * gt(timestamp64, timestamp4) -> bool
 */
void gt_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left > right);
}

const SFunction gt_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &gt_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction gt_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &gt_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gt(int64, int64) -> bool
 */
void gt_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left > right);
}

const SFunction gt_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &gt_int64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gt(float64, float64) -> bool
 */
void gt_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left > right);
}

const SFunction gt_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &gt_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gt(string, string) -> bool
 */
void gt_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  auto cmp = strncmp(left, right, std::min(left_len, right_len));
  pushBool(stack, cmp > 0 || (cmp == 0 && left_len > right_len));
}

const SFunction gt_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &gt_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gte(uint64, uint64) -> bool
 * gte(timestamp64, timestamp64) -> bool
 */
void gte_uint64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popUInt64(stack);
  auto left = popUInt64(stack);
  pushBool(stack, left >= right);
}

const SFunction gte_uint64(
    { SType::UINT64, SType::UINT64 },
    SType::BOOL,
    &gte_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */

const SFunction gte_timestamp64(
    { SType::TIMESTAMP64, SType::TIMESTAMP64 },
    SType::BOOL,
    &gte_uint64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gte(int64, int64) -> bool
 */
void gte_int64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popInt64(stack);
  auto left = popInt64(stack);
  pushBool(stack, left >= right);
}

const SFunction gte_int64(
    { SType::INT64, SType::INT64 },
    SType::BOOL,
    &gte_int64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gte(float64, float64) -> bool
 */
void gte_float64_call(sql_txn* ctx, VMStack* stack) {
  auto right = popFloat64(stack);
  auto left = popFloat64(stack);
  pushBool(stack, left >= right);
}

const SFunction gte_float64(
    { SType::FLOAT64, SType::FLOAT64 },
    SType::BOOL,
    &gte_float64_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


/**
 * gte(string, string) -> bool
 */
void gte_string_call(sql_txn* ctx, VMStack* stack) {
  const char* right;
  size_t right_len;
  popString(stack, &right, &right_len);

  const char* left;
  size_t left_len;
  popString(stack, &left, &left_len);

  auto cmp = strncmp(left, right, std::min(left_len, right_len));
  pushBool(stack, cmp > 0 || (cmp == 0 && left_len >= right_len));
}

const SFunction gte_string(
    { SType::STRING, SType::STRING },
    SType::BOOL,
    &gte_string_call,
    false,  /* has_side_effects*/
    false); /* allow_argument_conversion */


} // namespace expressions
} // namespace csql

