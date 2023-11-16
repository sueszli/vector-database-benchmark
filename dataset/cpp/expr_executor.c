/*
  Copyright(C) 2010-2018  Brazil
  Copyright(C) 2020-2022  Sutou Kouhei <kou@clear-code.com>

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "grn.h"
#include "grn_bulk.h"
#include "grn_ctx_impl.h"
#include "grn_expr.h"
#include "grn_expr_code.h"
#include "grn_expr_executor.h"
#include "grn_geo.h"
#include "grn_normalizer.h"

#ifdef GRN_SUPPORT_REGEXP
# include "grn_onigmo.h"
#endif

static void
grn_expr_executor_init_general(grn_ctx *ctx,
                               grn_expr_executor *executor)
{
}

#define EXPRVP(x) ((x)->header.impl_flags & GRN_OBJ_EXPRVALUE)

#define WITH_SPSAVE(block) do {\
  ctx->impl->stack_curr = sp - ctx->impl->stack;\
  e->values_curr = vp - e->values;\
  block\
  vp = e->values + e->values_curr;\
  sp = ctx->impl->stack + ctx->impl->stack_curr;\
  s_ = ctx->impl->stack;\
  s0 = (sp > s_) ? sp[-1] : NULL;\
  s1 = (sp > s_ + 1) ? sp[-2] : NULL;\
} while (0)

#define GEO_RESOLUTION   3600000
#define GEO_RADIOUS      6357303
#define GEO_BES_C1       6334834
#define GEO_BES_C2       6377397
#define GEO_BES_C3       0.006674
#define GEO_GRS_C1       6335439
#define GEO_GRS_C2       6378137
#define GEO_GRS_C3       0.006694

#define VAR_SET_VALUE(ctx,var,value) do {\
  if (GRN_DB_OBJP(value)) {\
    (var)->header.type = GRN_PTR;\
    (var)->header.domain = DB_OBJ(value)->id;\
    GRN_PTR_SET(ctx, (var), (value));\
  } else {\
    (var)->header.type = (value)->header.type;\
    (var)->header.domain = (value)->header.domain;\
    GRN_TEXT_SET(ctx, (var), GRN_TEXT_VALUE(value), GRN_TEXT_LEN(value));\
  }\
} while (0)

#define PUSH1(v) do {\
  if (EXPRVP(v)) {\
    vp++;\
    if (vp - e->values > e->values_tail) { e->values_tail = vp - e->values; }\
  }\
  s1 = s0;\
  if (sp >= s_ + ctx->impl->stack_size) {\
    if (grn_ctx_expand_stack(ctx) != GRN_SUCCESS) {\
      goto exit;\
    }\
    sp += (ctx->impl->stack - s_);\
    s_ = ctx->impl->stack;\
  }\
  *sp++ = s0 = v;\
} while (0)

#define POP1(v) do {\
  if (EXPRVP(s0)) { vp--; }\
  v = s0;\
  s0 = s1;\
  sp--;\
  if (sp < s_) { ERR(GRN_INVALID_ARGUMENT, "stack underflow"); goto exit; }\
  s1 = (sp > s_ + 1) ? sp[-2] : NULL;\
} while (0)

#define ALLOC1(value) do {\
  s1 = s0;\
  if (sp >= s_ + ctx->impl->stack_size) {\
    if (grn_ctx_expand_stack(ctx) != GRN_SUCCESS) {\
      goto exit;\
    }\
    sp += (ctx->impl->stack - s_);\
    s_ = ctx->impl->stack;\
  }\
  *sp++ = s0 = value = vp++;\
  if (vp - e->values > e->values_tail) { e->values_tail = vp - e->values; }\
} while (0)

#define POP1ALLOC1(arg,value) do {\
  arg = s0;\
  if (EXPRVP(s0)) {\
    value = s0;\
  } else {\
    if (sp < s_ + 1) { ERR(GRN_INVALID_ARGUMENT, "stack underflow"); goto exit; }\
    sp[-1] = s0 = value = vp++;\
    if (vp - e->values > e->values_tail) { e->values_tail = vp - e->values; }\
    s0->header.impl_flags |= GRN_OBJ_EXPRVALUE;\
  }\
} while (0)

#define POP2ALLOC1(arg1,arg2,value) do {\
  if (EXPRVP(s0)) { vp--; }\
  if (EXPRVP(s1)) { vp--; }\
  arg2 = s0;\
  arg1 = s1;\
  sp--;\
  if (sp < s_ + 1) { ERR(GRN_INVALID_ARGUMENT, "stack underflow"); goto exit; }\
  s1 = (sp > s_ + 1) ? sp[-2] : NULL;\
  sp[-1] = s0 = value = vp++;\
  if (vp - e->values > e->values_tail) { e->values_tail = vp - e->values; }\
  s0->header.impl_flags |= GRN_OBJ_EXPRVALUE;\
} while (0)

#define INTEGER_ARITHMETIC_OPERATION_PLUS(x, y) ((x) + (y))
#define FLOAT_ARITHMETIC_OPERATION_PLUS(x, y) ((double)(x) + (double)(y))
#define INTEGER_ARITHMETIC_OPERATION_MINUS(x, y) ((x) - (y))
#define FLOAT_ARITHMETIC_OPERATION_MINUS(x, y) ((double)(x) - (double)(y))
#define INTEGER_ARITHMETIC_OPERATION_STAR(x, y) ((x) * (y))
#define FLOAT_ARITHMETIC_OPERATION_STAR(x, y) ((double)(x) * (double)(y))
#define INTEGER_ARITHMETIC_OPERATION_SLASH(x, y) ((x) / (y))
#define FLOAT_ARITHMETIC_OPERATION_SLASH(x, y) ((double)(x) / (double)(y))
#define INTEGER_ARITHMETIC_OPERATION_MOD(x, y) ((x) % (y))
#define FLOAT_ARITHMETIC_OPERATION_MOD(x, y) (fmod((x), (y)))
#define INTEGER_ARITHMETIC_OPERATION_SHIFTL(x, y) ((x) << (y))
#define FLOAT_ARITHMETIC_OPERATION_SHIFTL(x, y)                         \
  ((long long int)(x) << (long long int)(y))
#define INTEGER_ARITHMETIC_OPERATION_SHIFTR(x, y) ((x) >> (y))
#define FLOAT_ARITHMETIC_OPERATION_SHIFTR(x, y)                         \
  ((long long int)(x) >> (long long int)(y))
#define INTEGER8_ARITHMETIC_OPERATION_SHIFTRR(x, y)             \
  ((uint8_t)(x) >> (y))
#define INTEGER16_ARITHMETIC_OPERATION_SHIFTRR(x, y)            \
  ((uint16_t)(x) >> (y))
#define INTEGER32_ARITHMETIC_OPERATION_SHIFTRR(x, y)            \
  ((unsigned int)(x) >> (y))
#define INTEGER64_ARITHMETIC_OPERATION_SHIFTRR(x, y)            \
  ((long long unsigned int)(x) >> (y))
#define FLOAT_ARITHMETIC_OPERATION_SHIFTRR(x, y)                \
  ((long long unsigned int)(x) >> (long long unsigned int)(y))

#define INTEGER_ARITHMETIC_OPERATION_BITWISE_OR(x, y) ((x) | (y))
#define FLOAT_ARITHMETIC_OPERATION_BITWISE_OR(x, y)                \
  ((long long int)(x) | (long long int)(y))
#define INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR(x, y) ((x) ^ (y))
#define FLOAT_ARITHMETIC_OPERATION_BITWISE_XOR(x, y)                \
  ((long long int)(x) ^ (long long int)(y))
#define INTEGER_ARITHMETIC_OPERATION_BITWISE_AND(x, y) ((x) & (y))
#define FLOAT_ARITHMETIC_OPERATION_BITWISE_AND(x, y)                \
  ((long long int)(x) & (long long int)(y))

#define INTEGER_UNARY_ARITHMETIC_OPERATION_MINUS(x) (-(x))
#define FLOAT_UNARY_ARITHMETIC_OPERATION_MINUS(x) (-(x))
#define INTEGER_UNARY_ARITHMETIC_OPERATION_BITWISE_NOT(x) (~(x))
#define FLOAT_UNARY_ARITHMETIC_OPERATION_BITWISE_NOT(x) \
  (~((long long int)(x)))

#define TEXT_ARITHMETIC_OPERATION(operator) do {                        \
  long long int x_;                                                     \
  long long int y_;                                                     \
                                                                        \
  res->header.domain = GRN_DB_INT64;                                    \
                                                                        \
  GRN_INT64_SET(ctx, res, 0);                                           \
  grn_obj_cast(ctx, x, res, GRN_FALSE);                                 \
  x_ = GRN_INT64_VALUE(res);                                            \
                                                                        \
  GRN_INT64_SET(ctx, res, 0);                                           \
  grn_obj_cast(ctx, y, res, GRN_FALSE);                                 \
  y_ = GRN_INT64_VALUE(res);                                            \
                                                                        \
  GRN_INT64_SET(ctx, res, x_ operator y_);                              \
} while (0)

#define TEXT_UNARY_ARITHMETIC_OPERATION(unary_operator) do {    \
  grn_obj buffer;                                               \
  int64_t value;                                                \
  GRN_INT64_INIT(&buffer, 0);                                   \
  grn_obj_cast(ctx, x, &buffer, GRN_FALSE);                     \
  grn_obj_reinit(ctx, res, GRN_DB_INT64, 0);                    \
  value = GRN_INT64_VALUE(&buffer);                             \
  GRN_INT64_SET(ctx, res, unary_operator value);                \
  GRN_OBJ_FIN(ctx, &buffer);                                    \
} while (0)

#define ARITHMETIC_OPERATION_NO_CHECK(y) do {} while (0)
#define ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y) do {        \
  if ((int64_t)y == 0) {                                        \
    ERR(GRN_INVALID_ARGUMENT, "divisor should not be 0");       \
    goto exit;                                                  \
  }                                                             \
} while (0)


#define NUMERIC_ARITHMETIC_OPERATION_DISPATCH(set, get, x_, y,          \
                                              res, res_domain,          \
                                              integer_operation,        \
                                              float_operation,          \
                                              right_expression_check,   \
                                              invalid_type_error) do {  \
  switch (y->header.domain) {                                           \
  case GRN_DB_BOOL :                                                    \
    {                                                                   \
      uint8_t y_;                                                       \
      y_ = GRN_BOOL_VALUE(y) ? 1 : 0;                                   \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT8 :                                                    \
    {                                                                   \
      int8_t y_;                                                        \
      y_ = GRN_INT8_VALUE(y);                                           \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT8 :                                                   \
    {                                                                   \
      uint8_t y_;                                                       \
      y_ = GRN_UINT8_VALUE(y);                                          \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT16 :                                                   \
    {                                                                   \
      int16_t y_;                                                       \
      y_ = GRN_INT16_VALUE(y);                                          \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT16 :                                                  \
    {                                                                   \
      uint16_t y_;                                                      \
      y_ = GRN_UINT16_VALUE(y);                                         \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT32 :                                                   \
    {                                                                   \
      int y_;                                                           \
      y_ = GRN_INT32_VALUE(y);                                          \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    {                                                                   \
      unsigned int y_;                                                  \
      y_ = GRN_UINT32_VALUE(y);                                         \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    {                                                                   \
      long long int y_;                                                 \
      y_ = GRN_TIME_VALUE(y);                                           \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    {                                                                   \
      long long int y_;                                                 \
      y_ = GRN_INT64_VALUE(y);                                          \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    {                                                                   \
      long long unsigned int y_;                                        \
      y_ = GRN_UINT64_VALUE(y);                                         \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, res_domain, 0);                          \
      set(ctx, res, integer_operation(x_, y_));                         \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    {                                                                   \
      float y_;                                                         \
      y_ = GRN_FLOAT32_VALUE(y);                                        \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, GRN_DB_FLOAT32, 0);                      \
      GRN_FLOAT32_SET(ctx, res, float_operation(x_, y_));               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    {                                                                   \
      double y_;                                                        \
      y_ = GRN_FLOAT_VALUE(y);                                          \
      right_expression_check(y_);                                       \
      grn_obj_reinit(ctx, res, GRN_DB_FLOAT, 0);                        \
      GRN_FLOAT_SET(ctx, res, float_operation(x_, y_));                 \
    }                                                                   \
    break;                                                              \
  case GRN_DB_SHORT_TEXT :                                              \
  case GRN_DB_TEXT :                                                    \
  case GRN_DB_LONG_TEXT :                                               \
    grn_obj_reinit(ctx, res, res_domain, 0);                            \
    if (grn_obj_cast(ctx, y, res, GRN_FALSE)) {                         \
      ERR(GRN_INVALID_ARGUMENT,                                         \
          "not a numerical format: <%.*s>",                             \
          (int)GRN_TEXT_LEN(y), GRN_TEXT_VALUE(y));                     \
      goto exit;                                                        \
    }                                                                   \
    set(ctx, res, integer_operation(x_, get(res)));                     \
    break;                                                              \
  default :                                                             \
    invalid_type_error;                                                 \
    break;                                                              \
  }                                                                     \
} while (0)


#define ARITHMETIC_OPERATION_DISPATCH(x, y, res,                        \
                                      integer8_operation,               \
                                      integer16_operation,              \
                                      integer32_operation,              \
                                      integer64_operation,              \
                                      float_operation,                  \
                                      left_expression_check,            \
                                      right_expression_check,           \
                                      text_operation,                   \
                                      invalid_type_error) do {          \
  switch (x->header.domain) {                                           \
  case GRN_DB_BOOL :                                                    \
    {                                                                   \
      uint8_t x_;                                                       \
      x_ = GRN_BOOL_VALUE(x) ? 1 : 0;                                   \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_UINT8_SET,              \
                                            GRN_UINT8_VALUE,            \
                                            x_, y, res, GRN_DB_UINT8,   \
                                            integer8_operation,         \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT8 :                                                    \
    {                                                                   \
      int8_t x_;                                                        \
      x_ = GRN_INT8_VALUE(x);                                           \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_INT8_SET,               \
                                            GRN_INT8_VALUE,             \
                                            x_, y, res, GRN_DB_INT8,    \
                                            integer8_operation,         \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT8 :                                                   \
    {                                                                   \
      uint8_t x_;                                                       \
      x_ = GRN_UINT8_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_UINT8_SET,              \
                                            GRN_UINT8_VALUE,            \
                                            x_, y, res, GRN_DB_UINT8,   \
                                            integer8_operation,         \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT16 :                                                   \
    {                                                                   \
      int16_t x_;                                                       \
      x_ = GRN_INT16_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_INT16_SET,              \
                                            GRN_INT16_VALUE,            \
                                            x_, y, res, GRN_DB_INT16,   \
                                            integer16_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT16 :                                                  \
    {                                                                   \
      uint16_t x_;                                                      \
      x_ = GRN_UINT16_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_UINT16_SET,             \
                                            GRN_UINT16_VALUE,           \
                                            x_, y, res, GRN_DB_UINT16,  \
                                            integer16_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT32 :                                                   \
    {                                                                   \
      int x_;                                                           \
      x_ = GRN_INT32_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_INT32_SET,              \
                                            GRN_INT32_VALUE,            \
                                            x_, y, res, GRN_DB_INT32,   \
                                            integer32_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    {                                                                   \
      unsigned int x_;                                                  \
      x_ = GRN_UINT32_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_UINT32_SET,             \
                                            GRN_UINT32_VALUE,           \
                                            x_, y, res, GRN_DB_UINT32,  \
                                            integer32_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    {                                                                   \
      long long int x_;                                                 \
      x_ = GRN_INT64_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_INT64_SET,              \
                                            GRN_INT64_VALUE,            \
                                            x_, y, res, GRN_DB_INT64,   \
                                            integer64_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    {                                                                   \
      long long int x_;                                                 \
      x_ = GRN_TIME_VALUE(x);                                           \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_TIME_SET,               \
                                            GRN_TIME_VALUE,             \
                                            x_, y, res, GRN_DB_TIME,    \
                                            integer64_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    {                                                                   \
      long long unsigned int x_;                                        \
      x_ = GRN_UINT64_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_UINT64_SET,             \
                                            GRN_UINT64_VALUE,           \
                                            x_, y, res, GRN_DB_UINT64,  \
                                            integer64_operation,        \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    {                                                                   \
      float x_;                                                         \
      x_ = GRN_FLOAT32_VALUE(x);                                        \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_FLOAT32_SET,            \
                                            GRN_FLOAT32_VALUE,          \
                                            x_, y, res, GRN_DB_FLOAT32, \
                                            float_operation,            \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    {                                                                   \
      double x_;                                                        \
      x_ = GRN_FLOAT_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      NUMERIC_ARITHMETIC_OPERATION_DISPATCH(GRN_FLOAT_SET,              \
                                            GRN_FLOAT_VALUE,            \
                                            x_, y, res, GRN_DB_FLOAT,   \
                                            float_operation,            \
                                            float_operation,            \
                                            right_expression_check,     \
                                            invalid_type_error);        \
    }                                                                   \
    break;                                                              \
  case GRN_DB_SHORT_TEXT :                                              \
  case GRN_DB_TEXT :                                                    \
  case GRN_DB_LONG_TEXT :                                               \
    text_operation;                                                     \
    break;                                                              \
  default:                                                              \
    invalid_type_error;                                                 \
    break;                                                              \
  }                                                                     \
  code++;                                                               \
} while (0)

#define ARITHMETIC_BINARY_OPERATION_DISPATCH(operator,                  \
                                             integer8_operation,        \
                                             integer16_operation,       \
                                             integer32_operation,       \
                                             integer64_operation,       \
                                             float_operation,           \
                                             left_expression_check,     \
                                             right_expression_check,    \
                                             text_operation,            \
                                             invalid_type_error) do {   \
  grn_obj *x = NULL;                                                    \
  grn_obj *y = NULL;                                                    \
                                                                        \
  POP2ALLOC1(x, y, res);                                                \
  if (x->header.type == GRN_VECTOR || y->header.type == GRN_VECTOR) {   \
    grn_obj inspected_x;                                                \
    grn_obj inspected_y;                                                \
    GRN_TEXT_INIT(&inspected_x, 0);                                     \
    GRN_TEXT_INIT(&inspected_y, 0);                                     \
    grn_inspect(ctx, &inspected_x, x);                                  \
    grn_inspect(ctx, &inspected_y, y);                                  \
    ERR(GRN_INVALID_ARGUMENT,                                           \
        "<%s> doesn't support vector: <%.*s> %s <%.*s>",                \
        operator,                                                       \
        (int)GRN_TEXT_LEN(&inspected_x), GRN_TEXT_VALUE(&inspected_x),  \
        operator,                                                       \
        (int)GRN_TEXT_LEN(&inspected_y), GRN_TEXT_VALUE(&inspected_y)); \
    GRN_OBJ_FIN(ctx, &inspected_x);                                     \
    GRN_OBJ_FIN(ctx, &inspected_y);                                     \
    goto exit;                                                          \
  }                                                                     \
  ARITHMETIC_OPERATION_DISPATCH(x, y, res,                              \
                                integer8_operation,                     \
                                integer16_operation,                    \
                                integer32_operation,                    \
                                integer64_operation,                    \
                                float_operation,                        \
                                left_expression_check,                  \
                                right_expression_check,                 \
                                text_operation,                         \
                                invalid_type_error);                    \
} while (0)

#define SIGNED_INTEGER_DIVISION_OPERATION_SLASH(x, y)                   \
  ((y == -1) ? -(x) : (x) / (y))
#define UNSIGNED_INTEGER_DIVISION_OPERATION_SLASH(x, y) ((x) / (y))
#define FLOAT_DIVISION_OPERATION_SLASH(x, y) ((double)(x) / (double)(y))
#define SIGNED_INTEGER_DIVISION_OPERATION_MOD(x, y) ((y == -1) ? 0 : (x) % (y))
#define UNSIGNED_INTEGER_DIVISION_OPERATION_MOD(x, y) ((x) % (y))
#define FLOAT_DIVISION_OPERATION_MOD(x, y) (fmod((x), (y)))

#define DIVISION_OPERATION_DISPATCH_RIGHT(set, get, x_, y, res,         \
                                          signed_integer_operation,     \
                                          unsigned_integer_operation,   \
                                          float_operation) do {         \
  switch (y->header.domain) {                                           \
  case GRN_DB_INT8 :                                                    \
    {                                                                   \
      int8_t y_;                                                        \
      y_ = GRN_INT8_VALUE(y);                                           \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, signed_integer_operation(x_, y_));                  \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT8 :                                                   \
    {                                                                   \
      uint8_t y_;                                                       \
      y_ = GRN_UINT8_VALUE(y);                                          \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, unsigned_integer_operation(x_, y_));                \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT16 :                                                   \
    {                                                                   \
      int16_t y_;                                                       \
      y_ = GRN_INT16_VALUE(y);                                          \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, signed_integer_operation(x_, y_));                  \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT16 :                                                  \
    {                                                                   \
      uint16_t y_;                                                      \
      y_ = GRN_UINT16_VALUE(y);                                         \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, unsigned_integer_operation(x_, y_));                \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT32 :                                                   \
    {                                                                   \
      int32_t y_;                                                       \
      y_ = GRN_INT32_VALUE(y);                                          \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, signed_integer_operation(x_, y_));                  \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    {                                                                   \
      uint32_t y_;                                                      \
      y_ = GRN_UINT32_VALUE(y);                                         \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, unsigned_integer_operation(x_, y_));                \
    }                                                                   \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    {                                                                   \
      int64_t y_;                                                       \
      y_ = GRN_TIME_VALUE(y);                                           \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, signed_integer_operation(x_, y_));                  \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    {                                                                   \
      int64_t y_;                                                       \
      y_ = GRN_INT64_VALUE(y);                                          \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, signed_integer_operation(x_, y_));                  \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    {                                                                   \
      uint64_t y_;                                                      \
      y_ = GRN_UINT64_VALUE(y);                                         \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      set(ctx, res, unsigned_integer_operation(x_, y_));                \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    {                                                                   \
      float y_;                                                         \
      y_ = GRN_FLOAT32_VALUE(y);                                        \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      res->header.domain = GRN_DB_FLOAT32;                              \
      GRN_FLOAT32_SET(ctx, res, float_operation(x_, y_));               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    {                                                                   \
      double y_;                                                        \
      y_ = GRN_FLOAT_VALUE(y);                                          \
      ARITHMETIC_OPERATION_ZERO_DIVISION_CHECK(y_);                     \
      res->header.domain = GRN_DB_FLOAT;                                \
      GRN_FLOAT_SET(ctx, res, float_operation(x_, y_));                 \
    }                                                                   \
    break;                                                              \
  case GRN_DB_SHORT_TEXT :                                              \
  case GRN_DB_TEXT :                                                    \
  case GRN_DB_LONG_TEXT :                                               \
    set(ctx, res, 0);                                                   \
    if (grn_obj_cast(ctx, y, res, GRN_FALSE)) {                         \
      ERR(GRN_INVALID_ARGUMENT,                                         \
          "not a numerical format: <%.*s>",                             \
          (int)GRN_TEXT_LEN(y), GRN_TEXT_VALUE(y));                     \
      goto exit;                                                        \
    }                                                                   \
    /* The following "+ 0" is needed to suppress warnings that say */   \
    /* comparison is always false due to limited range of data type */  \
    set(ctx, res, signed_integer_operation(x_, (get(res) + 0)));        \
    break;                                                              \
  default :                                                             \
    break;                                                              \
  }                                                                     \
} while (0)

#define DIVISION_OPERATION_DISPATCH_LEFT(x, y, res,                     \
                                         signed_integer_operation,      \
                                         unsigned_integer_operation,    \
                                         float_operation,               \
                                         invalid_type_error) do {       \
  switch (x->header.domain) {                                           \
  case GRN_DB_INT8 :                                                    \
    {                                                                   \
      int8_t x_;                                                        \
      x_ = GRN_INT8_VALUE(x);                                           \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_INT8_SET,                   \
                                        GRN_INT8_VALUE,                 \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT8 :                                                   \
    {                                                                   \
      uint8_t x_;                                                       \
      x_ = GRN_UINT8_VALUE(x);                                          \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_UINT8_SET,                  \
                                        (int)GRN_UINT8_VALUE,           \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT16 :                                                   \
    {                                                                   \
      int16_t x_;                                                       \
      x_ = GRN_INT16_VALUE(x);                                          \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_INT16_SET,                  \
                                        GRN_INT16_VALUE,                \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT16 :                                                  \
    {                                                                   \
      uint16_t x_;                                                      \
      x_ = GRN_UINT16_VALUE(x);                                         \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_UINT16_SET,                 \
                                        (int)GRN_UINT16_VALUE,          \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT32 :                                                   \
    {                                                                   \
      int32_t x_;                                                       \
      x_ = GRN_INT32_VALUE(x);                                          \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_INT32_SET,                  \
                                        GRN_INT32_VALUE,                \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    {                                                                   \
      uint32_t x_;                                                      \
      x_ = GRN_UINT32_VALUE(x);                                         \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_UINT32_SET,                 \
                                        GRN_UINT32_VALUE,               \
                                        x_, y, res,                     \
                                        unsigned_integer_operation,     \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    {                                                                   \
      int64_t x_;                                                       \
      x_ = GRN_INT64_VALUE(x);                                          \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_INT64_SET,                  \
                                        GRN_INT64_VALUE,                \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    {                                                                   \
      int64_t x_;                                                       \
      x_ = GRN_TIME_VALUE(x);                                           \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_TIME_SET,                   \
                                        GRN_TIME_VALUE,                 \
                                        x_, y, res,                     \
                                        signed_integer_operation,       \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    {                                                                   \
      uint64_t x_;                                                      \
      x_ = GRN_UINT64_VALUE(x);                                         \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_UINT64_SET,                 \
                                        GRN_UINT64_VALUE,               \
                                        x_, y, res,                     \
                                        unsigned_integer_operation,     \
                                        unsigned_integer_operation,     \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    {                                                                   \
      float x_;                                                         \
      x_ = GRN_FLOAT32_VALUE(x);                                        \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_FLOAT32_SET,                \
                                        GRN_FLOAT32_VALUE,              \
                                        x_, y, res,                     \
                                        float_operation,                \
                                        float_operation,                \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    {                                                                   \
      double x_;                                                        \
      x_ = GRN_FLOAT_VALUE(x);                                          \
      DIVISION_OPERATION_DISPATCH_RIGHT(GRN_FLOAT_SET,                  \
                                        GRN_FLOAT_VALUE,                \
                                        x_, y, res,                     \
                                        float_operation,                \
                                        float_operation,                \
                                        float_operation);               \
    }                                                                   \
    break;                                                              \
  case GRN_DB_SHORT_TEXT :                                              \
  case GRN_DB_TEXT :                                                    \
  case GRN_DB_LONG_TEXT :                                               \
    invalid_type_error;                                                 \
    break;                                                              \
  default:                                                              \
    break;                                                              \
  }                                                                     \
  code++;                                                               \
} while (0)

#define DIVISION_OPERATION_DISPATCH(signed_integer_operation,           \
                                    unsigned_integer_operation,         \
                                    float_operation,                    \
                                    invalid_type_error) do {            \
  grn_obj *x = NULL;                                                    \
  grn_obj *y = NULL;                                                    \
                                                                        \
  POP2ALLOC1(x, y, res);                                                \
  if (y != res) {                                                       \
    res->header.domain = x->header.domain;                              \
  }                                                                     \
  DIVISION_OPERATION_DISPATCH_LEFT(x, y, res,                           \
                                   signed_integer_operation,            \
                                   unsigned_integer_operation,          \
                                   float_operation,                     \
                                   invalid_type_error);                 \
  if (y == res) {                                                       \
    res->header.domain = x->header.domain;                              \
  }                                                                     \
} while (0)

#define ARITHMETIC_UNARY_OPERATION_DISPATCH(integer_operation,          \
                                            float_operation,            \
                                            left_expression_check,      \
                                            right_expression_check,     \
                                            text_operation,             \
                                            invalid_type_error) do {    \
  grn_obj *x = NULL;                                                    \
  POP1ALLOC1(x, res);                                                   \
  switch (x->header.domain) {                                           \
  case GRN_DB_INT8 :                                                    \
    {                                                                   \
      int8_t x_;                                                        \
      x_ = GRN_INT8_VALUE(x);                                           \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT8, 0);                         \
      GRN_INT8_SET(ctx, res, integer_operation(x_));                    \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT8 :                                                   \
    {                                                                   \
      int16_t x_;                                                       \
      x_ = GRN_UINT8_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT16, 0);                        \
      GRN_INT16_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT16 :                                                   \
    {                                                                   \
      int16_t x_;                                                       \
      x_ = GRN_INT16_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT16, 0);                        \
      GRN_INT16_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT16 :                                                  \
    {                                                                   \
      int32_t x_;                                                       \
      x_ = GRN_UINT16_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT32, 0);                        \
      GRN_INT32_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT32 :                                                   \
    {                                                                   \
      int32_t x_;                                                       \
      x_ = GRN_INT32_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT32, 0);                        \
      GRN_INT32_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    {                                                                   \
      int64_t x_;                                                       \
      x_ = GRN_UINT32_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT64, 0);                        \
      GRN_INT64_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    {                                                                   \
      int64_t x_;                                                       \
      x_ = GRN_INT64_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_INT64, 0);                        \
      GRN_INT64_SET(ctx, res, integer_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    {                                                                   \
      int64_t x_;                                                       \
      x_ = GRN_TIME_VALUE(x);                                           \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_TIME, 0);                         \
      GRN_TIME_SET(ctx, res, integer_operation(x_));                    \
    }                                                                   \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    {                                                                   \
      uint64_t x_;                                                      \
      x_ = GRN_UINT64_VALUE(x);                                         \
      left_expression_check(x_);                                        \
      if (x_ > (uint64_t)INT64_MAX) {                                   \
        ERR(GRN_INVALID_ARGUMENT,                                       \
            "too large UInt64 value to inverse sign: "                  \
            "<%" GRN_FMT_INT64U ">",                                    \
            x_);                                                        \
        goto exit;                                                      \
      } else {                                                          \
        int64_t signed_x_;                                              \
        signed_x_ = x_;                                                 \
        grn_obj_reinit(ctx, res, GRN_DB_INT64, 0);                      \
        GRN_INT64_SET(ctx, res, integer_operation(signed_x_));          \
      }                                                                 \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    {                                                                   \
      float x_;                                                         \
      x_ = GRN_FLOAT32_VALUE(x);                                        \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_FLOAT32, 0);                      \
      GRN_FLOAT32_SET(ctx, res, float_operation(x_));                   \
    }                                                                   \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    {                                                                   \
      double x_;                                                        \
      x_ = GRN_FLOAT_VALUE(x);                                          \
      left_expression_check(x_);                                        \
      grn_obj_reinit(ctx, res, GRN_DB_FLOAT, 0);                        \
      GRN_FLOAT_SET(ctx, res, float_operation(x_));                     \
    }                                                                   \
    break;                                                              \
  case GRN_DB_SHORT_TEXT :                                              \
  case GRN_DB_TEXT :                                                    \
  case GRN_DB_LONG_TEXT :                                               \
    text_operation;                                                     \
    break;                                                              \
  default:                                                              \
    invalid_type_error;                                                 \
    break;                                                              \
  }                                                                     \
  code++;                                                               \
} while (0)

#define EXEC_OPERATE(operate_sentence, assign_sentence)   \
  operate_sentence                                        \
  assign_sentence

#define EXEC_OPERATE_POST(operate_sentence, assign_sentence)    \
  assign_sentence                                               \
  operate_sentence

#define UNARY_OPERATE_AND_ASSIGN_DISPATCH(exec_operate, delta,          \
                                          set_flags) do {               \
  grn_obj *var = NULL;                                                  \
  grn_obj *col = NULL;                                                  \
  grn_obj value;                                                        \
  grn_id rid;                                                           \
                                                                        \
  POP1ALLOC1(var, res);                                                 \
  if (var->header.type != GRN_PTR) {                                    \
    ERR(GRN_INVALID_ARGUMENT, "invalid variable type: 0x%0x",           \
        var->header.type);                                              \
    goto exit;                                                          \
  }                                                                     \
  if (GRN_BULK_VSIZE(var) != (sizeof(grn_obj *) + sizeof(grn_id))) {    \
    ERR(GRN_INVALID_ARGUMENT,                                           \
        "invalid variable size: "                                       \
        "expected: %" GRN_FMT_SIZE                                      \
        "actual: %" GRN_FMT_SIZE,                                       \
        (sizeof(grn_obj *) + sizeof(grn_id)), GRN_BULK_VSIZE(var));     \
    goto exit;                                                          \
  }                                                                     \
  col = GRN_PTR_VALUE(var);                                             \
  rid = *(grn_id *)(GRN_BULK_HEAD(var) + sizeof(grn_obj *));            \
  res->header.type = GRN_VOID;                                          \
  res->header.domain = DB_OBJ(col)->range;                              \
  switch (DB_OBJ(col)->range) {                                         \
  case GRN_DB_INT32 :                                                   \
    GRN_INT32_INIT(&value, 0);                                          \
    GRN_INT32_SET(ctx, &value, delta);                                  \
    break;                                                              \
  case GRN_DB_UINT32 :                                                  \
    GRN_UINT32_INIT(&value, 0);                                         \
    GRN_UINT32_SET(ctx, &value, delta);                                 \
    break;                                                              \
  case GRN_DB_INT64 :                                                   \
    GRN_INT64_INIT(&value, 0);                                          \
    GRN_INT64_SET(ctx, &value, delta);                                  \
    break;                                                              \
  case GRN_DB_UINT64 :                                                  \
    GRN_UINT64_INIT(&value, 0);                                         \
    GRN_UINT64_SET(ctx, &value, delta);                                 \
    break;                                                              \
  case GRN_DB_FLOAT32 :                                                 \
    GRN_FLOAT32_INIT(&value, 0);                                        \
    GRN_FLOAT32_SET(ctx, &value, delta);                                \
    break;                                                              \
  case GRN_DB_FLOAT :                                                   \
    GRN_FLOAT_INIT(&value, 0);                                          \
    GRN_FLOAT_SET(ctx, &value, delta);                                  \
    break;                                                              \
  case GRN_DB_TIME :                                                    \
    GRN_TIME_INIT(&value, 0);                                           \
    GRN_TIME_SET(ctx, &value, GRN_TIME_PACK(delta, 0));                 \
    break;                                                              \
  default:                                                              \
    ERR(GRN_INVALID_ARGUMENT,                                           \
        "invalid increment target type: %d "                            \
        "(FIXME: type name is needed)", DB_OBJ(col)->range);            \
    goto exit;                                                          \
    break;                                                              \
  }                                                                     \
  exec_operate(grn_obj_set_value(ctx, col, rid, &value, set_flags);,    \
               grn_obj_get_value(ctx, col, rid, res););                 \
  code++;                                                               \
} while (0)

#define ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(integer8_operation,    \
                                                 integer16_operation,   \
                                                 integer32_operation,   \
                                                 integer64_operation,   \
                                                 float_operation,       \
                                                 left_expression_check, \
                                                 right_expression_check,\
                                                 text_operation) do {   \
  grn_obj *value = NULL;                                                \
  grn_obj *var = NULL;                                                  \
  grn_obj *res = NULL;                                                  \
  if (code->value) {                                                    \
    value = code->value;                                                \
    POP1ALLOC1(var, res);                                               \
  } else {                                                              \
    POP2ALLOC1(var, value, res);                                        \
  }                                                                     \
  if (var->header.type == GRN_PTR &&                                    \
      GRN_BULK_VSIZE(var) == (sizeof(grn_obj *) + sizeof(grn_id))) {    \
    grn_obj *col = GRN_PTR_VALUE(var);                                  \
    grn_id rid = *(grn_id *)(GRN_BULK_HEAD(var) + sizeof(grn_obj *));   \
    grn_obj variable_value, casted_value;                               \
    grn_id domain;                                                      \
                                                                        \
    value = GRN_OBJ_RESOLVE(ctx, value);                                \
                                                                        \
    domain = grn_obj_get_range(ctx, col);                               \
    GRN_OBJ_INIT(&variable_value, GRN_BULK, 0, domain);                 \
    grn_obj_get_value(ctx, col, rid, &variable_value);                  \
                                                                        \
    GRN_OBJ_INIT(&casted_value, GRN_BULK, 0, domain);                   \
    if (grn_obj_cast(ctx, value, &casted_value, GRN_FALSE)) {           \
      ERR(GRN_INVALID_ARGUMENT, "invalid value: string");               \
      GRN_OBJ_FIN(ctx, &variable_value);                                \
      GRN_OBJ_FIN(ctx, &casted_value);                                  \
      POP1(res);                                                        \
      goto exit;                                                        \
    }                                                                   \
    ARITHMETIC_OPERATION_DISPATCH((&variable_value), (&casted_value),   \
                                  res,                                  \
                                  integer8_operation,                   \
                                  integer16_operation,                  \
                                  integer32_operation,                  \
                                  integer64_operation,                  \
                                  float_operation,                      \
                                  left_expression_check,                \
                                  right_expression_check,               \
                                  text_operation,);                     \
    grn_obj_set_value(ctx, col, rid, res, GRN_OBJ_SET);                 \
    GRN_OBJ_FIN(ctx, (&variable_value));                                \
    GRN_OBJ_FIN(ctx, (&casted_value));                                  \
  } else {                                                              \
    ERR(GRN_INVALID_ARGUMENT, "left hand expression isn't column.");    \
    POP1(res);                                                          \
  }                                                                     \
} while (0)

grn_inline static void
grn_expr_exec_get_member_vector(grn_ctx *ctx,
                                grn_obj *expr,
                                grn_obj *column_and_record_id,
                                grn_obj *index,
                                grn_obj *result)
{
  grn_obj *column;
  grn_id record_id;
  grn_obj values;
  uint32_t i;

  column = GRN_PTR_VALUE(column_and_record_id);
  record_id = *((grn_id *)(&(GRN_PTR_VALUE_AT(column_and_record_id, 1))));
  GRN_TEXT_INIT(&values, 0);
  grn_obj_get_value(ctx, column, record_id, &values);

  i = GRN_UINT32_VALUE(index);
  if (values.header.type == GRN_UVECTOR) {
    uint32_t n_elements = 0;
    grn_obj *range;
    grn_id range_id = DB_OBJ(column)->range;

    grn_obj_reinit(ctx, result, range_id, 0);
    range = grn_ctx_at(ctx, range_id);
    if (range) {
      switch (range->header.type) {
      case GRN_TYPE :
        n_elements = GRN_BULK_VSIZE(&values) / grn_type_size(ctx, range);
        break;
      case GRN_TABLE_HASH_KEY :
      case GRN_TABLE_PAT_KEY :
      case GRN_TABLE_DAT_KEY :
      case GRN_TABLE_NO_KEY :
        n_elements = GRN_BULK_VSIZE(&values) / sizeof(grn_id);
        break;
      }
    }
    if (n_elements > i) {
#define GET_UVECTOR_ELEMENT_AS(type) do {                               \
        GRN_ ## type ## _SET(ctx,                                       \
                             result,                                    \
                             GRN_ ## type ## _VALUE_AT(&values, i));    \
      } while (GRN_FALSE)
      switch (values.header.domain) {
      case GRN_DB_BOOL :
        GET_UVECTOR_ELEMENT_AS(BOOL);
        break;
      case GRN_DB_INT8 :
        GET_UVECTOR_ELEMENT_AS(INT8);
        break;
      case GRN_DB_UINT8 :
        GET_UVECTOR_ELEMENT_AS(UINT8);
        break;
      case GRN_DB_INT16 :
        GET_UVECTOR_ELEMENT_AS(INT16);
        break;
      case GRN_DB_UINT16 :
        GET_UVECTOR_ELEMENT_AS(UINT16);
        break;
      case GRN_DB_INT32 :
        GET_UVECTOR_ELEMENT_AS(INT32);
        break;
      case GRN_DB_UINT32 :
        GET_UVECTOR_ELEMENT_AS(UINT32);
        break;
      case GRN_DB_INT64 :
        GET_UVECTOR_ELEMENT_AS(INT64);
        break;
      case GRN_DB_UINT64 :
        GET_UVECTOR_ELEMENT_AS(UINT64);
        break;
      case GRN_DB_FLOAT32 :
        GET_UVECTOR_ELEMENT_AS(FLOAT32);
        break;
      case GRN_DB_FLOAT :
        GET_UVECTOR_ELEMENT_AS(FLOAT);
        break;
      case GRN_DB_TIME :
        GET_UVECTOR_ELEMENT_AS(TIME);
        break;
      default :
        GET_UVECTOR_ELEMENT_AS(RECORD);
        break;
      }
#undef GET_UVECTOR_ELEMENT_AS
    }
  } else {
    if (values.u.v.n_sections > i) {
      const char *content;
      unsigned int content_length;
      grn_id domain;

      content_length = grn_vector_get_element(ctx, &values, i,
                                              &content, NULL, &domain);
      grn_obj_reinit(ctx, result, domain, 0);
      grn_bulk_write(ctx, result, content, content_length);
    }
  }

  GRN_OBJ_FIN(ctx, &values);
}

grn_inline static void
grn_expr_exec_get_member_table(grn_ctx *ctx,
                               grn_obj *expr,
                               grn_obj *table,
                               grn_obj *key,
                               grn_obj *result)
{
  grn_id id;

  if (table->header.domain == key->header.domain) {
    id = grn_table_get(ctx, table, GRN_BULK_HEAD(key), GRN_BULK_VSIZE(key));
  } else {
    grn_obj casted_key;
    GRN_OBJ_INIT(&casted_key, GRN_BULK, 0, table->header.domain);
    if (grn_obj_cast(ctx, key, &casted_key, GRN_FALSE) == GRN_SUCCESS) {
      id = grn_table_get(ctx, table,
                         GRN_BULK_HEAD(&casted_key),
                         GRN_BULK_VSIZE(&casted_key));
    } else {
      id = GRN_ID_NIL;
    }
    GRN_OBJ_FIN(ctx, &casted_key);
  }

  grn_obj_reinit(ctx, result, DB_OBJ(table)->id, 0);
  GRN_RECORD_SET(ctx, result, id);
}

static grn_obj *
expr_exec(grn_ctx *ctx, grn_obj *expr)
{
  grn_obj *val = NULL;
  uint32_t stack_curr = ctx->impl->stack_curr;
  grn_expr *e = (grn_expr *)expr;
  register grn_obj **s_ = ctx->impl->stack;
  register grn_obj *s0 = NULL;
  register grn_obj *s1 = NULL;
  register grn_obj **sp;
  register grn_obj *vp = e->values;
  grn_obj *res = NULL, *v0 = grn_expr_get_var_by_offset(ctx, expr, 0);
  grn_expr_code *code = e->codes, *ce = &e->codes[e->codes_curr];
  sp = s_ + stack_curr;
  while (code < ce) {
    switch (code->op) {
    case GRN_OP_NOP :
      code++;
      break;
    case GRN_OP_PUSH :
      PUSH1(code->value);
      code++;
      break;
    case GRN_OP_POP :
      {
        grn_obj *obj = NULL;
        POP1(obj);
        code++;
      }
      break;
    case GRN_OP_GET_REF :
      {
        grn_obj *col = NULL;
        grn_obj *rec = NULL;
        if (code->nargs == 1) {
          rec = v0;
          if (code->value) {
            col = code->value;
            ALLOC1(res);
          } else {
            POP1ALLOC1(col, res);
          }
        } else {
          if (code->value) {
            col = code->value;
            POP1ALLOC1(rec, res);
          } else {
            POP2ALLOC1(rec, col, res);
          }
        }
        if (col->header.type == GRN_BULK) {
          grn_obj *table = grn_ctx_at(ctx, GRN_OBJ_GET_DOMAIN(rec));
          col = grn_obj_column(ctx, table, GRN_BULK_HEAD(col), GRN_BULK_VSIZE(col));
          if (col) { grn_expr_take_obj(ctx, (grn_obj *)e, col); }
        }
        if (col) {
          res->header.type = GRN_PTR;
          res->header.domain = GRN_ID_NIL;
          GRN_PTR_SET(ctx, res, col);
          GRN_UINT32_PUT(ctx, res, GRN_RECORD_VALUE(rec));
        } else {
          ERR(GRN_INVALID_ARGUMENT, "col resolve failed");
          goto exit;
        }
        code++;
      }
      break;
    case GRN_OP_CALL :
      {
        grn_obj *proc = NULL;
        if (code->value) {
          if (sp < s_ + code->nargs - 1) {
            ERR(GRN_INVALID_ARGUMENT, "stack error");
            goto exit;
          }
          proc = code->value;
          WITH_SPSAVE({
            grn_proc_call(ctx, proc, code->nargs - 1, expr);
          });
        } else {
          int offset = code->nargs;
          if (sp < s_ + offset) {
            ERR(GRN_INVALID_ARGUMENT, "stack error");
            goto exit;
          }
          proc = sp[-offset];
          if (grn_obj_is_window_function_proc(ctx, proc)) {
            grn_obj inspected;
            GRN_TEXT_INIT(&inspected, 0);
            grn_inspect(ctx, &inspected, proc);
            ERR(GRN_INVALID_ARGUMENT,
                "window function can't be executed for each record: %.*s",
                (int)GRN_TEXT_LEN(&inspected),
                GRN_TEXT_VALUE(&inspected));
            GRN_OBJ_FIN(ctx, &inspected);
            goto exit;
          } else {
            WITH_SPSAVE({
              grn_proc_call(ctx, proc, code->nargs - 1, expr);
            });
          }
          if (ctx->rc) {
            goto exit;
          }
          POP1(res);
          {
            grn_obj *proc_;
            POP1(proc_);
            if (proc != proc_) {
              GRN_LOG(ctx, GRN_LOG_WARNING, "stack may be corrupt");
            }
          }
          PUSH1(res);
        }
      }
      code++;
      break;
    case GRN_OP_INTERN :
      {
        grn_obj *obj = NULL;
        POP1(obj);
        obj = GRN_OBJ_RESOLVE(ctx, obj);
        res = grn_expr_get_var(ctx, expr, GRN_TEXT_VALUE(obj), GRN_TEXT_LEN(obj));
        if (!res) { res = grn_ctx_get(ctx, GRN_TEXT_VALUE(obj), GRN_TEXT_LEN(obj)); }
        if (!res) {
          ERR(GRN_INVALID_ARGUMENT, "intern failed");
          goto exit;
        }
        PUSH1(res);
      }
      code++;
      break;
    case GRN_OP_TABLE_CREATE :
      {
        grn_obj *value_type, *key_type, *flags, *name;
        POP1(value_type);
        value_type = GRN_OBJ_RESOLVE(ctx, value_type);
        POP1(key_type);
        key_type = GRN_OBJ_RESOLVE(ctx, key_type);
        POP1(flags);
        flags = GRN_OBJ_RESOLVE(ctx, flags);
        POP1(name);
        name = GRN_OBJ_RESOLVE(ctx, name);
        res = grn_table_create(ctx, GRN_TEXT_VALUE(name), GRN_TEXT_LEN(name),
                               NULL, GRN_UINT32_VALUE(flags),
                               key_type, value_type);
        PUSH1(res);
      }
      code++;
      break;
    case GRN_OP_EXPR_GET_VAR :
      {
        grn_obj *name = NULL;
        grn_obj *expr = NULL;
        POP1(name);
        name = GRN_OBJ_RESOLVE(ctx, name);
        POP1(expr);
        expr = GRN_OBJ_RESOLVE(ctx, expr);
        switch (name->header.domain) {
        case GRN_DB_INT32 :
          res = grn_expr_get_var_by_offset(ctx, expr, (unsigned int) GRN_INT32_VALUE(name));
          break;
        case GRN_DB_UINT32 :
          res = grn_expr_get_var_by_offset(ctx, expr, (unsigned int) GRN_UINT32_VALUE(name));
          break;
        case GRN_DB_INT64 :
          res = grn_expr_get_var_by_offset(ctx, expr, (unsigned int) GRN_INT64_VALUE(name));
          break;
        case GRN_DB_UINT64 :
          res = grn_expr_get_var_by_offset(ctx, expr, (unsigned int) GRN_UINT64_VALUE(name));
          break;
        case GRN_DB_SHORT_TEXT :
        case GRN_DB_TEXT :
        case GRN_DB_LONG_TEXT :
          res = grn_expr_get_var(ctx, expr, GRN_TEXT_VALUE(name), GRN_TEXT_LEN(name));
          break;
        default :
          ERR(GRN_INVALID_ARGUMENT, "invalid type");
          goto exit;
        }
        PUSH1(res);
      }
      code++;
      break;
    case GRN_OP_ASSIGN :
      {
        grn_obj *value = NULL;
        grn_obj *var = NULL;
        if (code->value) {
          value = code->value;
        } else {
          POP1(value);
        }
        value = GRN_OBJ_RESOLVE(ctx, value);
        POP1(var);
        // var = GRN_OBJ_RESOLVE(ctx, var);
        if (var->header.type == GRN_PTR &&
            GRN_BULK_VSIZE(var) == (sizeof(grn_obj *) + sizeof(grn_id))) {
          grn_obj *col = GRN_PTR_VALUE(var);
          grn_id rid = *(grn_id *)(GRN_BULK_HEAD(var) + sizeof(grn_obj *));
          grn_obj_set_value(ctx, col, rid, value, GRN_OBJ_SET);
        } else {
          VAR_SET_VALUE(ctx, var, value);
        }
        PUSH1(value);
      }
      code++;
      break;
    case GRN_OP_STAR_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        FLOAT_ARITHMETIC_OPERATION_STAR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable *= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_SLASH_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_SLASH,
        INTEGER_ARITHMETIC_OPERATION_SLASH,
        INTEGER_ARITHMETIC_OPERATION_SLASH,
        INTEGER_ARITHMETIC_OPERATION_SLASH,
        FLOAT_ARITHMETIC_OPERATION_SLASH,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable /= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_MOD_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_MOD,
        INTEGER_ARITHMETIC_OPERATION_MOD,
        INTEGER_ARITHMETIC_OPERATION_MOD,
        INTEGER_ARITHMETIC_OPERATION_MOD,
        FLOAT_ARITHMETIC_OPERATION_MOD,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable %%= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_PLUS_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_PLUS,
        INTEGER_ARITHMETIC_OPERATION_PLUS,
        INTEGER_ARITHMETIC_OPERATION_PLUS,
        INTEGER_ARITHMETIC_OPERATION_PLUS,
        FLOAT_ARITHMETIC_OPERATION_PLUS,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable += \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_MINUS_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_MINUS,
        INTEGER_ARITHMETIC_OPERATION_MINUS,
        INTEGER_ARITHMETIC_OPERATION_MINUS,
        INTEGER_ARITHMETIC_OPERATION_MINUS,
        FLOAT_ARITHMETIC_OPERATION_MINUS,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable -= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_SHIFTL_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        FLOAT_ARITHMETIC_OPERATION_SHIFTL,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable <<= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_SHIFTR_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        FLOAT_ARITHMETIC_OPERATION_SHIFTR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable >>= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_SHIFTRR_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER8_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER16_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER32_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER64_ARITHMETIC_OPERATION_SHIFTRR,
        FLOAT_ARITHMETIC_OPERATION_SHIFTRR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT,
              "variable >>>= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_AND_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_AND,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable &= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_OR_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_OR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable |= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_XOR_ASSIGN :
      ARITHMETIC_OPERATION_AND_ASSIGN_DISPATCH(
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_XOR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT, "variable ^= \"string\" isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_JUMP :
      code += code->nargs + 1;
      break;
    case GRN_OP_CJUMP :
      {
        grn_obj *v = NULL;
        POP1(v);
        if (!grn_obj_is_true(ctx, v)) {
          code += code->nargs;
        }
      }
      code++;
      break;
    case GRN_OP_GET_VALUE :
      {
        grn_obj *col = NULL;
        grn_obj *rec = NULL;
        do {
          if (code->nargs == 1) {
            rec = v0;
            if (code->value) {
              col = code->value;
              ALLOC1(res);
            } else {
              POP1ALLOC1(col, res);
            }
          } else {
            if (code->value) {
              col = code->value;
              POP1ALLOC1(rec, res);
            } else {
              POP2ALLOC1(rec, col, res);
            }
          }
          if (col->header.type == GRN_BULK) {
            grn_obj *table = grn_ctx_at(ctx, GRN_OBJ_GET_DOMAIN(rec));
            col = grn_obj_column(ctx, table, GRN_BULK_HEAD(col), GRN_BULK_VSIZE(col));
            if (col) { grn_expr_take_obj(ctx, (grn_obj *)expr, col); }
          }
          if (!col) {
            ERR(GRN_INVALID_ARGUMENT, "col resolve failed");
            goto exit;
          }
          grn_obj_reinit_for(ctx, res, col);
          grn_obj_get_value(ctx, col, GRN_RECORD_VALUE(rec), res);
          code++;
        } while (code < ce && code->op == GRN_OP_GET_VALUE);
      }
      break;
    case GRN_OP_OBJ_SEARCH :
      {
        grn_obj *op = NULL;
        grn_obj *query = NULL;
        grn_obj *index = NULL;
        // todo : grn_search_optarg optarg;
        POP1(op);
        op = GRN_OBJ_RESOLVE(ctx, op);
        POP1(res);
        res = GRN_OBJ_RESOLVE(ctx, res);
        POP1(query);
        query = GRN_OBJ_RESOLVE(ctx, query);
        POP1(index);
        index = GRN_OBJ_RESOLVE(ctx, index);
        grn_obj_search(ctx, index, query, res,
                       (grn_operator)GRN_UINT32_VALUE(op), NULL);
      }
      code++;
      break;
    case GRN_OP_TABLE_SELECT :
      {
        grn_obj *op = NULL;
        grn_obj *res = NULL;
        grn_obj *expr = NULL;
        grn_obj *table = NULL;
        POP1(op);
        op = GRN_OBJ_RESOLVE(ctx, op);
        POP1(res);
        res = GRN_OBJ_RESOLVE(ctx, res);
        POP1(expr);
        expr = GRN_OBJ_RESOLVE(ctx, expr);
        POP1(table);
        table = GRN_OBJ_RESOLVE(ctx, table);
        WITH_SPSAVE({
          grn_table_select(ctx, table, expr, res, (grn_operator)GRN_UINT32_VALUE(op));
        });
        PUSH1(res);
      }
      code++;
      break;
    case GRN_OP_TABLE_SORT :
      {
        grn_obj *keys_ = NULL;
        grn_obj *res = NULL;
        grn_obj *limit = NULL;
        grn_obj *table = NULL;
        POP1(keys_);
        keys_ = GRN_OBJ_RESOLVE(ctx, keys_);
        POP1(res);
        res = GRN_OBJ_RESOLVE(ctx, res);
        POP1(limit);
        limit = GRN_OBJ_RESOLVE(ctx, limit);
        POP1(table);
        table = GRN_OBJ_RESOLVE(ctx, table);
        {
          grn_table_sort_key *keys;
          const char *p = GRN_BULK_HEAD(keys_), *tokbuf[256];
          int n = grn_str_tok(p, GRN_BULK_VSIZE(keys_), ' ', tokbuf, 256, NULL);
          if ((keys = GRN_MALLOCN(grn_table_sort_key, n))) {
            int i, n_keys = 0;
            for (i = 0; i < n; i++) {
              uint32_t len = (uint32_t) (tokbuf[i] - p);
              grn_obj *col = grn_obj_column(ctx, table, p, len);
              if (col) {
                keys[n_keys].key = col;
                keys[n_keys].flags = GRN_TABLE_SORT_ASC;
                keys[n_keys].offset = 0;
                n_keys++;
              } else {
                if (p[0] == ':' && p[1] == 'd' && len == 2 && n_keys) {
                  keys[n_keys - 1].flags |= GRN_TABLE_SORT_DESC;
                }
              }
              p = tokbuf[i] + 1;
            }
            WITH_SPSAVE({
              grn_table_sort(ctx, table, 0, GRN_INT32_VALUE(limit), res, keys, n_keys);
            });
            for (i = 0; i < n_keys; i++) {
              grn_obj_unlink(ctx, keys[i].key);
            }
            GRN_FREE(keys);
          }
        }
      }
      code++;
      break;
    case GRN_OP_TABLE_GROUP :
      {
        grn_obj *res = NULL;
        grn_obj *keys_ = NULL;
        grn_obj *table = NULL;
        POP1(res);
        res = GRN_OBJ_RESOLVE(ctx, res);
        POP1(keys_);
        keys_ = GRN_OBJ_RESOLVE(ctx, keys_);
        POP1(table);
        table = GRN_OBJ_RESOLVE(ctx, table);
        {
          grn_table_sort_key *keys;
          grn_table_group_result results;
          const char *p = GRN_BULK_HEAD(keys_), *tokbuf[256];
          int n = grn_str_tok(p, GRN_BULK_VSIZE(keys_), ' ', tokbuf, 256, NULL);
          if ((keys = GRN_MALLOCN(grn_table_sort_key, n))) {
            int i, n_keys = 0;
            for (i = 0; i < n; i++) {
              uint32_t len = (uint32_t) (tokbuf[i] - p);
              grn_obj *col = grn_obj_column(ctx, table, p, len);
              if (col) {
                keys[n_keys].key = col;
                keys[n_keys].flags = GRN_TABLE_SORT_ASC;
                keys[n_keys].offset = 0;
                n_keys++;
              } else if (n_keys) {
                if (p[0] == ':' && p[1] == 'd' && len == 2) {
                  keys[n_keys - 1].flags |= GRN_TABLE_SORT_DESC;
                } else {
                  keys[n_keys - 1].offset = grn_atoi(p, p + len, NULL);
                }
              }
              p = tokbuf[i] + 1;
            }
            /* todo : support multi-results */
            results.table = res;
            results.key_begin = 0;
            results.key_end = 0;
            results.limit = 0;
            results.flags = 0;
            results.op = GRN_OP_OR;
            WITH_SPSAVE({
              grn_table_group(ctx, table, keys, n_keys, &results, 1);
            });
            for (i = 0; i < n_keys; i++) {
              grn_obj_unlink(ctx, keys[i].key);
            }
            GRN_FREE(keys);
          }
        }
      }
      code++;
      break;
    case GRN_OP_JSON_PUT :
      {
        grn_obj_format format;
        grn_obj *res = NULL;
        grn_obj *str = NULL;
        grn_obj *table = NULL;
        POP1(res);
        res = GRN_OBJ_RESOLVE(ctx, res);
        POP1(str);
        str = GRN_OBJ_RESOLVE(ctx, str);
        POP1(table);
        table = GRN_OBJ_RESOLVE(ctx, table);
        GRN_OBJ_FORMAT_INIT(&format, grn_table_size(ctx, table), 0, -1, 0);
        format.flags = 0;
        grn_obj_columns(ctx, table,
                        GRN_TEXT_VALUE(str), GRN_TEXT_LEN(str), &format.columns);
        grn_text_otoj(ctx, res, table, &format);
        grn_obj_format_fin(ctx, &format);
      }
      code++;
      break;
    case GRN_OP_AND :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_obj *result = NULL;
        POP2ALLOC1(x, y, res);
        if (grn_obj_is_true(ctx, x)) {
          if (grn_obj_is_true(ctx, y)) {
            result = y;
          }
        }
        if (result) {
          if (res != result) {
            grn_bulk_copy(ctx, result, res);
          }
        } else {
          grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
          GRN_BOOL_SET(ctx, res, GRN_FALSE);
        }
      }
      code++;
      break;
    case GRN_OP_OR :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_obj *result = NULL;
        POP2ALLOC1(x, y, res);
        if (grn_obj_is_true(ctx, x)) {
          result = x;
        } else {
          if (grn_obj_is_true(ctx, y)) {
            result = y;
          } else {
            result = NULL;
          }
        }
        if (result) {
          if (res != result) {
            grn_bulk_copy(ctx, result, res);
          }
        } else {
          grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
          GRN_BOOL_SET(ctx, res, GRN_FALSE);
        }
      }
      code++;
      break;
    case GRN_OP_AND_NOT :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_bool is_true;
        POP2ALLOC1(x, y, res);
        if (!grn_obj_is_true(ctx, x) || grn_obj_is_true(ctx, y)) {
          is_true = GRN_FALSE;
        } else {
          is_true = GRN_TRUE;
        }
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, is_true);
      }
      code++;
      break;
    case GRN_OP_ADJUST :
      {
        /* todo */
      }
      code++;
      break;
    case GRN_OP_MATCH :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_bool matched;
        POP1(y);
        POP1(x);
        WITH_SPSAVE({
          matched = grn_operator_exec_match(ctx, x, y);
        });
        ALLOC1(res);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, matched);
      }
      code++;
      break;
    case GRN_OP_EQUAL :
      {
        grn_bool is_equal;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        is_equal = grn_operator_exec_equal(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, is_equal);
      }
      code++;
      break;
    case GRN_OP_NOT_EQUAL :
      {
        grn_bool is_not_equal;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        is_not_equal = grn_operator_exec_not_equal(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, is_not_equal);
      }
      code++;
      break;
    case GRN_OP_PREFIX :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_bool matched;
        POP1(y);
        POP1(x);
        WITH_SPSAVE({
          matched = grn_operator_exec_prefix(ctx, x, y);
        });
        ALLOC1(res);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, matched);
      }
      code++;
      break;
    case GRN_OP_SUFFIX :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        grn_bool matched = GRN_FALSE;
        POP2ALLOC1(x, y, res);
        if (GRN_TEXT_LEN(x) >= GRN_TEXT_LEN(y) &&
            !memcmp(GRN_TEXT_VALUE(x) + GRN_TEXT_LEN(x) - GRN_TEXT_LEN(y),
                    GRN_TEXT_VALUE(y), GRN_TEXT_LEN(y))) {
          matched = GRN_TRUE;
        }
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, matched);
      }
      code++;
      break;
    case GRN_OP_LESS :
      {
        grn_bool r;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        r = grn_operator_exec_less(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, r);
      }
      code++;
      break;
    case GRN_OP_GREATER :
      {
        grn_bool r;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        r = grn_operator_exec_greater(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, r);
      }
      code++;
      break;
    case GRN_OP_LESS_EQUAL :
      {
        grn_bool r;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        r = grn_operator_exec_less_equal(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, r);
      }
      code++;
      break;
    case GRN_OP_GREATER_EQUAL :
      {
        grn_bool r;
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        r = grn_operator_exec_greater_equal(ctx, x, y);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, r);
      }
      code++;
      break;
    case GRN_OP_GEO_DISTANCE1 :
      {
        grn_obj *value = NULL;
        double lng1, lat1, lng2, lat2, x, y, d;
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        lat2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        x = (lng2 - lng1) * cos((lat1 + lat2) * 0.5);
        y = (lat2 - lat1);
        d = sqrt((x * x) + (y * y)) * GEO_RADIOUS;
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_FLOAT;
        GRN_FLOAT_SET(ctx, res, d);
      }
      code++;
      break;
    case GRN_OP_GEO_DISTANCE2 :
      {
        grn_obj *value = NULL;
        double lng1, lat1, lng2, lat2, x, y, d;
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        lat2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        x = sin(fabs(lng2 - lng1) * 0.5);
        y = sin(fabs(lat2 - lat1) * 0.5);
        d = asin(sqrt((y * y) + cos(lat1) * cos(lat2) * x * x)) * 2 * GEO_RADIOUS;
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_FLOAT;
        GRN_FLOAT_SET(ctx, res, d);
      }
      code++;
      break;
    case GRN_OP_GEO_DISTANCE3 :
      {
        grn_obj *value = NULL;
        double lng1, lat1, lng2, lat2, p, q, m, n, x, y, d;
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        lat2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        p = (lat1 + lat2) * 0.5;
        q = (1 - GEO_BES_C3 * sin(p) * sin(p));
        m = GEO_BES_C1 / sqrt(q * q * q);
        n = GEO_BES_C2 / sqrt(q);
        x = n * cos(p) * fabs(lng1 - lng2);
        y = m * fabs(lat1 - lat2);
        d = sqrt((x * x) + (y * y));
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_FLOAT;
        GRN_FLOAT_SET(ctx, res, d);
      }
      code++;
      break;
    case GRN_OP_GEO_DISTANCE4 :
      {
        grn_obj *value = NULL;
        double lng1, lat1, lng2, lat2, p, q, m, n, x, y, d;
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        lat2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        p = (lat1 + lat2) * 0.5;
        q = (1 - GEO_GRS_C3 * sin(p) * sin(p));
        m = GEO_GRS_C1 / sqrt(q * q * q);
        n = GEO_GRS_C2 / sqrt(q);
        x = n * cos(p) * fabs(lng1 - lng2);
        y = m * fabs(lat1 - lat2);
        d = sqrt((x * x) + (y * y));
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_FLOAT;
        GRN_FLOAT_SET(ctx, res, d);
      }
      code++;
      break;
    case GRN_OP_GEO_WITHINP5 :
      {
        int r;
        grn_obj *value = NULL;
        double lng0, lat0, lng1, lat1, x, y, d;
        POP1(value);
        lng0 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat0 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        x = (lng1 - lng0) * cos((lat0 + lat1) * 0.5);
        y = (lat1 - lat0);
        d = sqrt((x * x) + (y * y)) * GEO_RADIOUS;
        switch (value->header.domain) {
        case GRN_DB_INT32 :
          r = d <= GRN_INT32_VALUE(value);
          break;
        case GRN_DB_FLOAT :
          r = d <= GRN_FLOAT_VALUE(value);
          break;
        default :
          r = 0;
          break;
        }
        GRN_INT32_SET(ctx, res, r);
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_INT32;
      }
      code++;
      break;
    case GRN_OP_GEO_WITHINP6 :
      {
        int r;
        grn_obj *value = NULL;
        double lng0, lat0, lng1, lat1, lng2, lat2, x, y, d;
        POP1(value);
        lng0 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat0 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lat1 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1(value);
        lng2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        POP1ALLOC1(value, res);
        lat2 = GRN_GEO_INT2RAD(GRN_INT32_VALUE(value));
        x = (lng1 - lng0) * cos((lat0 + lat1) * 0.5);
        y = (lat1 - lat0);
        d = (x * x) + (y * y);
        x = (lng2 - lng1) * cos((lat1 + lat2) * 0.5);
        y = (lat2 - lat1);
        r = d <= (x * x) + (y * y);
        GRN_INT32_SET(ctx, res, r);
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_INT32;
      }
      code++;
      break;
    case GRN_OP_GEO_WITHINP8 :
      {
        int r;
        grn_obj *value = NULL;
        int64_t ln0, la0, ln1, la1, ln2, la2, ln3, la3;
        POP1(value);
        ln0 = GRN_INT32_VALUE(value);
        POP1(value);
        la0 = GRN_INT32_VALUE(value);
        POP1(value);
        ln1 = GRN_INT32_VALUE(value);
        POP1(value);
        la1 = GRN_INT32_VALUE(value);
        POP1(value);
        ln2 = GRN_INT32_VALUE(value);
        POP1(value);
        la2 = GRN_INT32_VALUE(value);
        POP1(value);
        ln3 = GRN_INT32_VALUE(value);
        POP1ALLOC1(value, res);
        la3 = GRN_INT32_VALUE(value);
        r = ((ln2 <= ln0) && (ln0 <= ln3) && (la2 <= la0) && (la0 <= la3));
        GRN_INT32_SET(ctx, res, r);
        res->header.type = GRN_BULK;
        res->header.domain = GRN_DB_INT32;
      }
      code++;
      break;
    case GRN_OP_PLUS :
      {
        grn_obj *x = NULL;
        grn_obj *y = NULL;
        POP2ALLOC1(x, y, res);
        if (x->header.type == GRN_VECTOR && y->header.type == GRN_VECTOR) {
          grn_id domain_id = x->header.domain;
          if (domain_id == GRN_ID_NIL) {
            domain_id = y->header.domain;
          }
          if (domain_id == GRN_ID_NIL && grn_vector_size(ctx, x) > 0) {
            const char *content;
            grn_vector_get_element(ctx, x, 0, &content, NULL, &domain_id);
          }
          if (domain_id == GRN_ID_NIL && grn_vector_size(ctx, y) > 0) {
            const char *content;
            grn_vector_get_element(ctx, y, 0, &content, NULL, &domain_id);
          }
          if (domain_id == GRN_ID_NIL) {
            /* TODO: Consider better default for GRN_VECTOR. */
            domain_id = GRN_DB_SHORT_TEXT;
          }
          if (res == y) {
            grn_obj y_copied;
            GRN_OBJ_INIT(&y_copied, GRN_VECTOR, 0, y->header.domain);
            grn_vector_copy(ctx, y, &y_copied);
            grn_obj_reinit(ctx, res, domain_id, GRN_OBJ_VECTOR);
            grn_vector_copy(ctx, x, res);
            grn_vector_copy(ctx, &y_copied, res);
            GRN_OBJ_FIN(ctx, &y_copied);
          } else {
            if (res != x) {
              grn_obj_reinit(ctx, res, domain_id, GRN_OBJ_VECTOR);
              grn_vector_copy(ctx, x, res);
            }
            grn_vector_copy(ctx, y, res);
          }
          code++;
        } else if (x->header.type == GRN_VECTOR ||
                   y->header.type == GRN_VECTOR) {
          grn_obj inspected_x;
          grn_obj inspected_y;
          GRN_TEXT_INIT(&inspected_x, 0);
          GRN_TEXT_INIT(&inspected_y, 0);
          grn_inspect(ctx, &inspected_x, x);
          grn_inspect(ctx, &inspected_y, y);
          ERR(GRN_INVALID_ARGUMENT,
              "<+> doesn't support %s + %s: <%.*s> + <%.*s>",
              x->header.type == GRN_VECTOR ? "vector" : "non-vector",
              y->header.type == GRN_VECTOR ? "vector" : "non-vector",
              (int)GRN_TEXT_LEN(&inspected_x), GRN_TEXT_VALUE(&inspected_x),
              (int)GRN_TEXT_LEN(&inspected_y), GRN_TEXT_VALUE(&inspected_y));
          GRN_OBJ_FIN(ctx, &inspected_x);
          GRN_OBJ_FIN(ctx, &inspected_y);
          goto exit;
        } else {
          ARITHMETIC_OPERATION_DISPATCH(
            x, y, res,
            INTEGER_ARITHMETIC_OPERATION_PLUS,
            INTEGER_ARITHMETIC_OPERATION_PLUS,
            INTEGER_ARITHMETIC_OPERATION_PLUS,
            INTEGER_ARITHMETIC_OPERATION_PLUS,
            FLOAT_ARITHMETIC_OPERATION_PLUS,
            ARITHMETIC_OPERATION_NO_CHECK,
            ARITHMETIC_OPERATION_NO_CHECK,
            {
              if (x == res) {
                grn_obj_cast(ctx, y, res, GRN_FALSE);
              } else if (y == res) {
                grn_obj buffer;
                GRN_TEXT_INIT(&buffer, 0);
                grn_obj_cast(ctx, x, &buffer, GRN_FALSE);
                grn_obj_cast(ctx, y, &buffer, GRN_FALSE);
                GRN_BULK_REWIND(res);
                grn_obj_reinit(ctx, res, GRN_DB_TEXT, 0);
                grn_obj_cast(ctx, &buffer, res, GRN_FALSE);
                GRN_OBJ_FIN(ctx, &buffer);
              } else {
                grn_obj_reinit(ctx, res, GRN_DB_TEXT, 0);
                grn_obj_cast(ctx, x, res, GRN_FALSE);
                grn_obj_cast(ctx, y, res, GRN_FALSE);
              }
            },
            );
        }
      }
      break;
    case GRN_OP_MINUS :
      if (code->nargs == 1) {
        ARITHMETIC_UNARY_OPERATION_DISPATCH(
          INTEGER_UNARY_ARITHMETIC_OPERATION_MINUS,
          FLOAT_UNARY_ARITHMETIC_OPERATION_MINUS,
          ARITHMETIC_OPERATION_NO_CHECK,
          ARITHMETIC_OPERATION_NO_CHECK,
          TEXT_UNARY_ARITHMETIC_OPERATION(-),);
      } else {
        ARITHMETIC_BINARY_OPERATION_DISPATCH(
          "-",
          INTEGER_ARITHMETIC_OPERATION_MINUS,
          INTEGER_ARITHMETIC_OPERATION_MINUS,
          INTEGER_ARITHMETIC_OPERATION_MINUS,
          INTEGER_ARITHMETIC_OPERATION_MINUS,
          FLOAT_ARITHMETIC_OPERATION_MINUS,
          ARITHMETIC_OPERATION_NO_CHECK,
          ARITHMETIC_OPERATION_NO_CHECK,
          {
            ERR(GRN_INVALID_ARGUMENT,
                "\"string\" - \"string\" "
                "isn't supported");
            goto exit;
          }
          ,);
      }
      break;
    case GRN_OP_STAR :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        "*",
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        INTEGER_ARITHMETIC_OPERATION_STAR,
        FLOAT_ARITHMETIC_OPERATION_STAR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          ERR(GRN_INVALID_ARGUMENT,
              "\"string\" * \"string\" "
              "isn't supported");
          goto exit;
        }
        ,);
      break;
    case GRN_OP_SLASH :
      DIVISION_OPERATION_DISPATCH(
        SIGNED_INTEGER_DIVISION_OPERATION_SLASH,
        UNSIGNED_INTEGER_DIVISION_OPERATION_SLASH,
        FLOAT_DIVISION_OPERATION_SLASH,
        {
          ERR(GRN_INVALID_ARGUMENT,
              "\"string\" / \"string\" "
              "isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_MOD :
      DIVISION_OPERATION_DISPATCH(
        SIGNED_INTEGER_DIVISION_OPERATION_MOD,
        UNSIGNED_INTEGER_DIVISION_OPERATION_MOD,
        FLOAT_DIVISION_OPERATION_MOD,
        {
          ERR(GRN_INVALID_ARGUMENT,
              "\"string\" %% \"string\" "
              "isn't supported");
          goto exit;
        });
      break;
    case GRN_OP_BITWISE_NOT :
      ARITHMETIC_UNARY_OPERATION_DISPATCH(
        INTEGER_UNARY_ARITHMETIC_OPERATION_BITWISE_NOT,
        FLOAT_UNARY_ARITHMETIC_OPERATION_BITWISE_NOT,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_UNARY_ARITHMETIC_OPERATION(~),);
      break;
    case GRN_OP_BITWISE_OR :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        "|",
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_OR,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_OR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_ARITHMETIC_OPERATION(|),);
      break;
    case GRN_OP_BITWISE_XOR :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        "^",
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_XOR,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_XOR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_ARITHMETIC_OPERATION(^),);
      break;
    case GRN_OP_BITWISE_AND :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        "&",
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        INTEGER_ARITHMETIC_OPERATION_BITWISE_AND,
        FLOAT_ARITHMETIC_OPERATION_BITWISE_AND,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_ARITHMETIC_OPERATION(&),);
      break;
    case GRN_OP_SHIFTL :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        "<<",
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        INTEGER_ARITHMETIC_OPERATION_SHIFTL,
        FLOAT_ARITHMETIC_OPERATION_SHIFTL,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_ARITHMETIC_OPERATION(<<),);
      break;
    case GRN_OP_SHIFTR :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        ">>",
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        INTEGER_ARITHMETIC_OPERATION_SHIFTR,
        FLOAT_ARITHMETIC_OPERATION_SHIFTR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        TEXT_ARITHMETIC_OPERATION(>>),);
      break;
    case GRN_OP_SHIFTRR :
      ARITHMETIC_BINARY_OPERATION_DISPATCH(
        ">>>",
        INTEGER8_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER16_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER32_ARITHMETIC_OPERATION_SHIFTRR,
        INTEGER64_ARITHMETIC_OPERATION_SHIFTRR,
        FLOAT_ARITHMETIC_OPERATION_SHIFTRR,
        ARITHMETIC_OPERATION_NO_CHECK,
        ARITHMETIC_OPERATION_NO_CHECK,
        {
          long long unsigned int x_;
          long long unsigned int y_;

          res->header.type = GRN_BULK;
          res->header.domain = GRN_DB_INT64;

          GRN_INT64_SET(ctx, res, 0);
          grn_obj_cast(ctx, x, res, GRN_FALSE);
          x_ = GRN_INT64_VALUE(res);

          GRN_INT64_SET(ctx, res, 0);
          grn_obj_cast(ctx, y, res, GRN_FALSE);
          y_ = GRN_INT64_VALUE(res);

          GRN_INT64_SET(ctx, res, x_ >> y_);
        }
        ,);
      break;
    case GRN_OP_INCR :
      UNARY_OPERATE_AND_ASSIGN_DISPATCH(EXEC_OPERATE, 1, GRN_OBJ_INCR);
      break;
    case GRN_OP_DECR :
      UNARY_OPERATE_AND_ASSIGN_DISPATCH(EXEC_OPERATE, 1, GRN_OBJ_DECR);
      break;
    case GRN_OP_INCR_POST :
      UNARY_OPERATE_AND_ASSIGN_DISPATCH(EXEC_OPERATE_POST, 1, GRN_OBJ_INCR);
      break;
    case GRN_OP_DECR_POST :
      UNARY_OPERATE_AND_ASSIGN_DISPATCH(EXEC_OPERATE_POST, 1, GRN_OBJ_DECR);
      break;
    case GRN_OP_NOT :
      {
        grn_obj *value = NULL;
        grn_bool value_boolean;
        POP1ALLOC1(value, res);
        GRN_OBJ_IS_TRUE(ctx, value, value_boolean);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, !value_boolean);
      }
      code++;
      break;
    case GRN_OP_GET_MEMBER :
      {
        grn_obj *receiver = NULL;
        grn_obj *index_or_key = NULL;
        POP2ALLOC1(receiver, index_or_key, res);
        if (receiver->header.type == GRN_PTR) {
          grn_obj *index = index_or_key;
          grn_expr_exec_get_member_vector(ctx, expr, receiver, index, res);
        } else {
          grn_obj *key = index_or_key;
          grn_expr_exec_get_member_table(ctx, expr, receiver, key, res);
        }
        code++;
      }
      break;
    case GRN_OP_REGEXP :
      {
        grn_obj *target = NULL;
        grn_obj *pattern = NULL;
        grn_bool matched;
        POP1(pattern);
        POP1(target);
        WITH_SPSAVE({
          matched = grn_operator_exec_regexp(ctx, target, pattern);
        });
        ALLOC1(res);
        grn_obj_reinit(ctx, res, GRN_DB_BOOL, 0);
        GRN_BOOL_SET(ctx, res, matched);
      }
      code++;
      break;
    default :
      ERR(GRN_FUNCTION_NOT_IMPLEMENTED, "not implemented operator assigned");
      goto exit;
      break;
    }
  }
  ctx->impl->stack_curr = sp - s_;
  if (ctx->impl->stack_curr > stack_curr) {
    val = grn_ctx_pop(ctx);
  }
exit :
  if (ctx->impl->stack_curr > stack_curr) {
    /*
      GRN_LOG(ctx, GRN_LOG_WARNING, "stack balance=%d",
      stack_curr - ctx->impl->stack_curr);
    */
    ctx->impl->stack_curr = stack_curr;
  }
  return val;
}

static grn_obj *
grn_expr_executor_exec_general(grn_ctx *ctx,
                               grn_expr_executor *executor,
                               grn_id id)
{
  if (!executor->variable && id != GRN_ID_NIL) {
    ERR(GRN_INVALID_ARGUMENT,
        "[expr-executor][exec][general] can't specify ID for no variable expr");
    return NULL;
  }
  if (executor->variable) {
    GRN_RECORD_SET(ctx, executor->variable, id);
  }
  return expr_exec(ctx, executor->expr);
}

static void
grn_expr_executor_fin_general(grn_ctx *ctx,
                              grn_expr_executor *executor)
{
}

static void
grn_expr_executor_init_constant(grn_ctx *ctx,
                                grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  executor->data.constant.value = e->codes[0].value;
}

static grn_bool
grn_expr_executor_is_constant(grn_ctx *ctx,
                              grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;

  if (e->codes_curr != 1) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);

  if (target->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (!target->value) {
    return GRN_FALSE;
  }

  grn_expr_executor_init_constant(ctx, executor);

  return GRN_TRUE;
}

static grn_obj *
grn_expr_executor_exec_constant(grn_ctx *ctx,
                                grn_expr_executor *executor,
                                grn_id id)
{
  return executor->data.constant.value;
}

static void
grn_expr_executor_fin_constant(grn_ctx *ctx,
                               grn_expr_executor *executor)
{
}

static void
grn_expr_executor_init_value(grn_ctx *ctx,
                             grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  executor->data.value.column = e->codes[0].value;
  GRN_VOID_INIT(&(executor->data.value.value_buffer));
}

static grn_bool
grn_expr_executor_is_value(grn_ctx *ctx,
                           grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;

  if (e->codes_curr != 1) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (!target->value) {
    return GRN_FALSE;
  }

  grn_expr_executor_init_value(ctx, executor);

  return GRN_TRUE;
}

static grn_obj *
grn_expr_executor_exec_value(grn_ctx *ctx,
                             grn_expr_executor *executor,
                             grn_id id)
{
  grn_obj *value_buffer = &(executor->data.value.value_buffer);

  GRN_BULK_REWIND(value_buffer);
  grn_obj_get_value(ctx, executor->data.value.column, id, value_buffer);

  return value_buffer;
}

static void
grn_expr_executor_fin_value(grn_ctx *ctx,
                            grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.value.value_buffer));
}

#ifdef GRN_SUPPORT_REGEXP
static void
grn_expr_executor_init_simple_regexp(grn_ctx *ctx,
                                     grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *result_buffer = &(executor->data.simple_regexp.result_buffer);
  grn_obj *pattern;

  GRN_BOOL_INIT(result_buffer, 0);
  GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);

  pattern = e->codes[1].value;
  executor->data.simple_regexp.regex =
    grn_onigmo_new(ctx,
                   GRN_TEXT_VALUE(pattern),
                   GRN_TEXT_LEN(pattern),
                   GRN_ONIGMO_OPTION_DEFAULT,
                   GRN_ONIGMO_SYNTAX_DEFAULT,
                   "[expr-executor][regexp]");
  if (!executor->data.simple_regexp.regex) {
    return;
  }

  GRN_VOID_INIT(&(executor->data.simple_regexp.value_buffer));

  executor->data.simple_regexp.normalizer =
    grn_ctx_get(ctx, GRN_NORMALIZER_AUTO_NAME, -1);
}

static grn_bool
grn_expr_executor_is_simple_regexp(grn_ctx *ctx,
                                   grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;
  grn_expr_code *pattern;
  grn_expr_code *operator;

  if (e->codes_curr != 3) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);
  pattern = &(e->codes[1]);
  operator = &(e->codes[2]);

  if (operator->op != GRN_OP_REGEXP) {
    return GRN_FALSE;
  }
  if (operator->nargs != 2) {
    return GRN_FALSE;
  }

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (target->nargs != 1) {
    return GRN_FALSE;
  }
  if (!target->value) {
    return GRN_FALSE;
  }
  if (target->value->header.type != GRN_COLUMN_VAR_SIZE) {
    return GRN_FALSE;
  }
  if ((target->value->header.flags & GRN_OBJ_COLUMN_TYPE_MASK) !=
      GRN_OBJ_COLUMN_SCALAR) {
    return GRN_FALSE;
  }
  switch (grn_obj_get_range(ctx, target->value)) {
  case GRN_DB_SHORT_TEXT :
  case GRN_DB_TEXT :
  case GRN_DB_LONG_TEXT :
    break;
  default :
    return GRN_FALSE;
  }

  if (pattern->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (pattern->nargs != 1) {
    return GRN_FALSE;
  }
  if (!pattern->value) {
    return GRN_FALSE;
  }
  if (pattern->value->header.type != GRN_BULK) {
    return GRN_FALSE;
  }
  switch (pattern->value->header.domain) {
  case GRN_DB_SHORT_TEXT :
  case GRN_DB_TEXT :
  case GRN_DB_LONG_TEXT :
    break;
  default :
    return GRN_FALSE;
  }

  if (!grn_onigmo_is_valid_encoding(ctx)) {
    return GRN_FALSE;
  }

  grn_expr_executor_init_simple_regexp(ctx, executor);

  return GRN_TRUE;
}

static grn_obj *
grn_expr_executor_exec_simple_regexp(grn_ctx *ctx,
                                     grn_expr_executor *executor,
                                     grn_id id)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  OnigRegex regex = executor->data.simple_regexp.regex;
  grn_obj *value_buffer = &(executor->data.simple_regexp.value_buffer);
  grn_obj *result_buffer = &(executor->data.simple_regexp.result_buffer);

  if (ctx->rc) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
    return result_buffer;
  }

  if (!regex) {
    return result_buffer;
  }

  grn_obj_reinit_for(ctx, value_buffer, e->codes[0].value);
  grn_obj_get_value(ctx, e->codes[0].value, id, value_buffer);
  {
    grn_obj *norm_target;
    const char *norm_target_raw;
    unsigned int norm_target_raw_length_in_bytes;

    norm_target = grn_string_open(ctx,
                                  GRN_TEXT_VALUE(value_buffer),
                                  GRN_TEXT_LEN(value_buffer),
                                  executor->data.simple_regexp.normalizer,
                                  0);
    grn_string_get_normalized(ctx, norm_target,
                              &norm_target_raw,
                              &norm_target_raw_length_in_bytes,
                              NULL);

    {
      OnigPosition position;
      position = onig_search(regex,
                             norm_target_raw,
                             norm_target_raw + norm_target_raw_length_in_bytes,
                             norm_target_raw,
                             norm_target_raw + norm_target_raw_length_in_bytes,
                             NULL,
                             ONIG_OPTION_NONE);
      grn_obj_close(ctx, norm_target);
      GRN_BOOL_SET(ctx, result_buffer, (position != ONIG_MISMATCH));
      return result_buffer;
    }
  }
}

static void
grn_expr_executor_fin_simple_regexp(grn_ctx *ctx,
                                    grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.simple_regexp.result_buffer));

  if (!executor->data.simple_regexp.regex) {
    return;
  }

  onig_free(executor->data.simple_regexp.regex);
  GRN_OBJ_FIN(ctx, &(executor->data.simple_regexp.value_buffer));
}
#endif /* GRN_SUPPORT_REGEXP */

static void
grn_expr_executor_init_simple_match(grn_ctx *ctx,
                                    grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *result_buffer = &(executor->data.simple_match.result_buffer);
  grn_obj *sub_text;

  GRN_BOOL_INIT(result_buffer, 0);
  GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);

  executor->data.simple_match.normalizer =
    grn_ctx_get(ctx, GRN_NORMALIZER_AUTO_NAME, -1);

  sub_text = e->codes[1].value;
  executor->data.simple_match.normalized_sub_text =
    grn_string_open(ctx,
                    GRN_TEXT_VALUE(sub_text),
                    GRN_TEXT_LEN(sub_text),
                    executor->data.simple_match.normalizer,
                    0);
  grn_string_get_normalized(
    ctx,
    executor->data.simple_match.normalized_sub_text,
    &(executor->data.simple_match.normalized_sub_text_raw),
    &(executor->data.simple_match.normalized_sub_text_raw_length_in_bytes),
    NULL);
#ifdef GRN_SUPPORT_REGEXP
  executor->data.simple_match.regex =
    grn_onigmo_new(
      ctx,
      executor->data.simple_match.normalized_sub_text_raw,
      executor->data.simple_match.normalized_sub_text_raw_length_in_bytes,
      GRN_ONIGMO_OPTION_DEFAULT,
      ONIG_SYNTAX_ASIS,
      "[expr-executor][match]");
#endif

  grn_obj *target = e->codes[0].value;
  grn_obj *value_buffer = &(executor->data.simple_match.value_buffer);
  GRN_VOID_INIT(value_buffer);
  grn_obj_reinit_for(ctx, value_buffer, target);
}

static grn_bool
grn_expr_executor_is_simple_match(grn_ctx *ctx,
                                  grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;
  grn_expr_code *sub_text;
  grn_expr_code *operator;

  if (e->codes_curr != 3) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);
  sub_text = &(e->codes[1]);
  operator = &(e->codes[2]);

  if (operator->op != GRN_OP_MATCH) {
    return GRN_FALSE;
  }
  if (operator->nargs != 2) {
    return GRN_FALSE;
  }

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (target->nargs != 1) {
    return GRN_FALSE;
  }
  if (!target->value) {
    return GRN_FALSE;
  }
  if (!(grn_obj_is_key_accessor(ctx, target->value) ||
        grn_obj_is_text_family_scalar_column(ctx, target->value))) {
    return false;
  }

  if (sub_text->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (sub_text->nargs != 1) {
    return GRN_FALSE;
  }
  if (!grn_obj_is_text_family_bulk(ctx, sub_text->value)) {
    return false;
  }

#ifdef GRN_SUPPORT_REGEXP
  if (!grn_onigmo_is_valid_encoding(ctx)) {
    return GRN_FALSE;
  }
#endif

  grn_expr_executor_init_simple_match(ctx, executor);

  return GRN_TRUE;
}

static bool
grn_expr_executor_exec_simple_match_have_sub_text(grn_ctx *ctx,
                                                  grn_expr_executor *executor,
                                                  const char *text,
                                                  unsigned int text_len)
{
  unsigned int sub_text_len =
    executor->data.simple_match.normalized_sub_text_raw_length_in_bytes;

  if (sub_text_len > text_len) {
    return false;
  }

#ifdef GRN_SUPPORT_REGEXP
  {
    OnigRegex regex = executor->data.simple_match.regex;
    OnigPosition position;
    position = onig_search(regex,
                           text,
                           text + text_len,
                           text,
                           text + text_len,
                           NULL,
                           ONIG_OPTION_NONE);
    return (position != ONIG_MISMATCH);
  }
#else /* GRN_SUPPORT_REGEXP */
  {
    grn_raw_string string;
    string.value = text;
    string.length = text_len;
    grn_raw_string sub_string;
    sub_string.value = executor->data.simple_match.normalized_sub_text_raw;
    sub_string.length = sub_text_len;
    return grn_raw_string_have_sub_string(ctx, &string, &sub_string);
  }
#endif /* GRN_SUPPORT_REGEXP */
}

static grn_obj *
grn_expr_executor_exec_simple_match(grn_ctx *ctx,
                                    grn_expr_executor *executor,
                                    grn_id id)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *value_buffer = &(executor->data.simple_match.value_buffer);
  grn_obj *result_buffer = &(executor->data.simple_match.result_buffer);

  if (ctx->rc != GRN_SUCCESS) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
    return result_buffer;
  }

  if (executor->data.simple_match.normalized_sub_text_raw_length_in_bytes == 0) {
    return result_buffer;
  }

  GRN_BULK_REWIND(value_buffer);
  grn_obj_get_value(ctx, e->codes[0].value, id, value_buffer);
  {
    grn_obj *norm_target;
    norm_target = grn_string_open(ctx,
                                  GRN_TEXT_VALUE(value_buffer),
                                  GRN_TEXT_LEN(value_buffer),
                                  executor->data.simple_match.normalizer,
                                  0);
    const char *norm_target_raw;
    unsigned int norm_target_raw_length_in_bytes;
    grn_string_get_normalized(ctx,
                              norm_target,
                              &norm_target_raw,
                              &norm_target_raw_length_in_bytes,
                              NULL);
    bool have_sub_text =
      grn_expr_executor_exec_simple_match_have_sub_text(
        ctx,
        executor,
        norm_target_raw,
        norm_target_raw_length_in_bytes);
    GRN_BOOL_SET(ctx, result_buffer, have_sub_text);
    grn_obj_close(ctx, norm_target);
  }

  return result_buffer;
}

static void
grn_expr_executor_fin_simple_match(grn_ctx *ctx,
                                   grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.simple_match.result_buffer));

#ifdef GRN_SUPPORT_REGEXP
  if (executor->data.simple_match.regex) {
    onig_free(executor->data.simple_match.regex);
  }
#endif /* GRN_SUPPORT_REGEXP */

  grn_obj_close(ctx, executor->data.simple_match.normalized_sub_text);
  GRN_OBJ_FIN(ctx, &(executor->data.simple_match.value_buffer));
}

static bool
grn_expr_executor_data_simple_proc_init(grn_ctx *ctx,
                                        grn_expr_executor *executor,
                                        grn_expr_executor_data_simple_proc *data)
{
  grn_proc_ctx *proc_ctx = &(data->proc_ctx);
  grn_expr *expr;
  grn_proc *proc;

  expr = (grn_expr *)(executor->expr);
  proc = (grn_proc *)(expr->codes[data->codes_start_offset].value);
  proc_ctx->proc = proc;
  proc_ctx->caller = executor->expr;

  int n_args = 0;
  int n_buffers = 0;
  {
    uint32_t i;
    for (i = data->codes_start_offset + 1;
         i < expr->codes_curr - data->codes_end_offset - 1;
         i++) {
      switch (expr->codes[i].op) {
      case GRN_OP_GET_VALUE :
        n_args++;
        n_buffers++;
        break;
      case GRN_OP_PUSH :
        n_args++;
        break;
      default :
        return false;
      }
    }
  }


  grn_obj **args = GRN_MALLOCN(grn_obj *, n_args);
  if (!args) {
    return false;
  }
  data->args = args;
  data->n_args = n_args;

  if (n_buffers > 0) {
    grn_obj *buffers = GRN_MALLOCN(grn_obj, n_buffers);
    if (!buffers) {
      GRN_FREE(args);
      return false;
    }
    int i;
    for (i = 0; i < n_buffers; i++) {
      GRN_VOID_INIT(&(buffers[i]));
    }
    data->buffers = buffers;
  } else {
    data->buffers = NULL;
  }
  data->n_buffers = n_buffers;

  return true;
}

static void
grn_expr_executor_data_simple_proc_fin(grn_ctx *ctx,
                                       grn_expr_executor *executor,
                                       grn_expr_executor_data_simple_proc *data)
{
  GRN_FREE(data->args);
  int n_buffers = data->n_buffers;
  if (n_buffers > 0) {
    grn_obj *buffers = data->buffers;
    int i;
    for (i = 0; i < n_buffers; i++) {
      GRN_OBJ_FIN(ctx, &(buffers[i]));
    }
    GRN_FREE(buffers);
  }
}

static grn_obj *
grn_expr_executor_data_simple_proc_exec(grn_ctx *ctx,
                                        grn_expr_executor *executor,
                                        grn_expr_executor_data_simple_proc *data,
                                        grn_id id)
{
  grn_proc_ctx *proc_ctx = &(data->proc_ctx);
  grn_obj **args = data->args;
  grn_obj *buffers = data->buffers;
  int n_args = data->n_args;
  grn_proc *proc;
  grn_expr *expr;
  grn_obj *result = NULL;

  proc = proc_ctx->proc;
  if (executor->variable) {
    GRN_RECORD_SET(ctx, executor->variable, id);
  }

  expr = (grn_expr *)(executor->expr);
  {
    uint32_t i_expr;
    uint32_t i_buffer;
    uint32_t i_arg;
    for (i_expr = data->codes_start_offset + 1, i_buffer = 0, i_arg = 0;
         i_expr < expr->codes_curr - data->codes_end_offset - 1;
         i_expr++, i_arg++) {
      switch (expr->codes[i_expr].op) {
      case GRN_OP_GET_VALUE :
        {
          grn_obj *buffer = &(buffers[i_buffer++]);
          GRN_BULK_REWIND(buffer);
          grn_obj_get_value(ctx, expr->codes[i_expr].value, id, buffer);
          args[i_arg] = buffer;
          break;
        }
      case GRN_OP_PUSH :
        args[i_arg] = expr->codes[i_expr].value;
        break;
      default :
        break;
      }
    }
  }

  uint32_t values_curr = expr->values_curr;
  if (proc->funcs[PROC_INIT]) {
    proc_ctx->phase = PROC_INIT;
    grn_obj *sub_result = proc->funcs[PROC_INIT](ctx,
                                                 n_args,
                                                 args,
                                                 &(proc_ctx->user_data));
    if (sub_result) {
      result = sub_result;
    }
  }
  if (proc->funcs[PROC_NEXT]) {
    proc_ctx->phase = PROC_NEXT;
    grn_obj *sub_result = proc->funcs[PROC_NEXT](ctx,
                                                 n_args,
                                                 args,
                                                 &(proc_ctx->user_data));
    if (sub_result) {
      result = sub_result;
    }
  }
  if (proc->funcs[PROC_FIN]) {
    proc_ctx->phase = PROC_FIN;
    grn_obj *sub_result = proc->funcs[PROC_FIN](ctx,
                                                 n_args,
                                                 args,
                                                 &(proc_ctx->user_data));
    if (sub_result) {
      result = sub_result;
    }
  }
  expr->values_curr = values_curr;

  return result;
}

static bool
grn_expr_executor_init_simple_proc(grn_ctx *ctx,
                                   grn_expr_executor *executor)
{
  grn_expr_executor_data_simple_proc *data = &(executor->data.simple_proc.data);
  data->codes_start_offset = 0;
  data->codes_end_offset = 0;
  return grn_expr_executor_data_simple_proc_init(ctx, executor, data);
}

static bool
grn_expr_executor_is_simple_proc(grn_ctx *ctx,
                                 grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  /*
   * 0: GRN_OP_PUSH(func)
   * n-1: GRN_OP_CALL(func)
   */
  if (e->codes_curr < 2) {
    return false;
  }

  grn_expr_code *func = &(e->codes[0]);
  if (func->op != GRN_OP_PUSH) {
    return false;
  }
  if (!grn_obj_is_function_proc(ctx, func->value)) {
    return false;
  }

  grn_proc *proc = (grn_proc *)(func->value);
  if (!proc->funcs[PROC_INIT] &&
      !proc->funcs[PROC_NEXT] &&
      !proc->funcs[PROC_FIN]) {
    return false;
  }

  if (func->modify != (int32_t)(e->codes_curr - 1)) {
    return false;
  }
  if (e->codes[func->modify].op != GRN_OP_CALL) {
    return false;
  }

  return grn_expr_executor_init_simple_proc(ctx, executor);
}

static grn_obj *
grn_expr_executor_exec_simple_proc(grn_ctx *ctx,
                                   grn_expr_executor *executor,
                                   grn_id id)
{
  grn_expr_executor_data_simple_proc *data = &(executor->data.simple_proc.data);
  return grn_expr_executor_data_simple_proc_exec(ctx, executor, data, id);
}

static void
grn_expr_executor_fin_simple_proc(grn_ctx *ctx,
                                  grn_expr_executor *executor)
{
  grn_expr_executor_data_simple_proc *data = &(executor->data.simple_proc.data);
  grn_expr_executor_data_simple_proc_fin(ctx, executor, data);
}

static grn_bool
grn_expr_executor_is_simple_condition_constant(grn_ctx *ctx,
                                               grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;
  grn_expr_code *constant;
  grn_expr_code *operator;
  grn_id target_range;
  grn_id constant_range;
  grn_bool constant_value_is_int = GRN_TRUE;
  int64_t constant_value_int = 0;
  uint64_t constant_value_uint = 0;
  grn_bool constant_always_over = GRN_FALSE;
  grn_bool constant_always_under = GRN_FALSE;

  if (e->codes_curr != 3) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);
  constant = &(e->codes[1]);
  operator = &(e->codes[2]);

  switch (operator->op) {
  case GRN_OP_EQUAL :
  case GRN_OP_NOT_EQUAL :
  case GRN_OP_LESS :
  case GRN_OP_GREATER :
  case GRN_OP_LESS_EQUAL :
  case GRN_OP_GREATER_EQUAL :
    break;
  default :
    return GRN_FALSE;
  }
  if (operator->nargs != 2) {
    return GRN_FALSE;
  }

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (target->nargs != 1) {
    return GRN_FALSE;
  }
  if (target->value->header.type != GRN_COLUMN_FIX_SIZE) {
    return GRN_FALSE;
  }

  target_range = grn_obj_get_range(ctx, target->value);
  if (!(GRN_DB_INT8 <= target_range && target_range <= GRN_DB_UINT64)) {
    return GRN_FALSE;
  }

  if (constant->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (constant->nargs != 1) {
    return GRN_FALSE;
  }
  if (!constant->value) {
    return GRN_FALSE;
  }
  if (constant->value->header.type != GRN_BULK) {
    return GRN_FALSE;
  }
  constant_range = constant->value->header.domain;
  if (!(GRN_DB_INT8 <= constant_range && constant_range <= GRN_DB_UINT64)) {
    return GRN_FALSE;
  }

#define CASE_INT(N)                                                     \
  GRN_DB_INT ## N :                                                     \
    constant_value_is_int = GRN_TRUE;                                   \
    constant_value_int = GRN_INT ## N ##_VALUE(constant->value);        \
    break
#define CASE_UINT(N)                                                    \
  GRN_DB_UINT ## N :                                                    \
    constant_value_is_int = GRN_FALSE;                                  \
    constant_value_uint = GRN_UINT ## N ##_VALUE(constant->value);      \
    break

  switch (constant_range) {
  case CASE_INT(8);
  case CASE_UINT(8);
  case CASE_INT(16);
  case CASE_UINT(16);
  case CASE_INT(32);
  case CASE_UINT(32);
  case CASE_INT(64);
  case CASE_UINT(64);
  default :
    return GRN_FALSE;
  }

#undef CASE_INT
#undef CASE_UINT

#define CASE_INT(N)                                                     \
  GRN_DB_INT ## N :                                                     \
    if (constant_value_is_int) {                                        \
      if (constant_value_int > INT ## N ## _MAX) {                      \
        constant_always_over = GRN_TRUE;                                \
      } else if (constant_value_int < INT ## N ## _MIN) {               \
        constant_always_under = GRN_TRUE;                               \
      }                                                                 \
    } else {                                                            \
      if (constant_value_uint > INT ## N ## _MAX) {                     \
        constant_always_over = GRN_TRUE;                                \
      }                                                                 \
    }                                                                   \
    break
#define CASE_UINT(N)                                                    \
  GRN_DB_UINT ## N :                                                    \
    if (constant_value_is_int) {                                        \
      if (constant_value_int > UINT ## N ## _MAX) {                     \
        constant_always_over = GRN_TRUE;                                \
      } else if (constant_value_int < 0) {                              \
        constant_always_under = GRN_TRUE;                               \
      }                                                                 \
    } else {                                                            \
      if (constant_value_uint > UINT ## N ## _MAX) {                    \
        constant_always_over = GRN_TRUE;                                \
      }                                                                 \
    }                                                                   \
    break

  switch (target_range) {
  case CASE_INT(8);
  case CASE_UINT(8);
  case CASE_INT(16);
  case CASE_UINT(16);
  case CASE_INT(32);
  case CASE_UINT(32);
  case CASE_INT(64);
  case GRN_DB_UINT64 :
    if (constant_value_is_int && constant_value_int < 0) {
      constant_always_under = GRN_TRUE;
    }
    break;
  default :
    return GRN_FALSE;
  }

  if (!constant_always_over && !constant_always_under) {
    return GRN_FALSE;
  }

  {
    grn_bool result;
    grn_obj *result_buffer =
      &(executor->data.simple_condition_constant.result_buffer);

    switch (operator->op) {
    case GRN_OP_EQUAL :
      result = GRN_FALSE;
      break;
    case GRN_OP_NOT_EQUAL :
      result = GRN_TRUE;
      break;
    case GRN_OP_LESS :
    case GRN_OP_LESS_EQUAL :
      result = constant_always_over;
      break;
    case GRN_OP_GREATER :
    case GRN_OP_GREATER_EQUAL :
      result = constant_always_under;
      break;
    default :
      return GRN_FALSE;
    }

    GRN_BOOL_INIT(result_buffer, 0);
    GRN_BOOL_SET(ctx, result_buffer, result);

    return GRN_TRUE;
  }
}

static grn_obj *
grn_expr_executor_exec_simple_condition_constant(grn_ctx *ctx,
                                                 grn_expr_executor *executor,
                                                 grn_id id)
{
  grn_obj *result_buffer =
    &(executor->data.simple_condition_constant.result_buffer);
  return result_buffer;
}

static void
grn_expr_executor_fin_simple_condition_constant(grn_ctx *ctx,
                                                grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx,
              &(executor->data.simple_condition_constant.result_buffer));
}

static void
grn_expr_executor_init_simple_condition_ra(grn_ctx *ctx,
                                           grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *target;
  grn_obj *constant;
  grn_operator op;
  grn_obj *result_buffer;
  grn_obj *value_buffer;
  grn_obj *constant_buffer;

  target = e->codes[0].value;
  constant = e->codes[1].value;
  op = e->codes[2].op;

  result_buffer = &(executor->data.simple_condition_ra.result_buffer);
  GRN_BOOL_INIT(result_buffer, 0);
  GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);

  value_buffer = &(executor->data.simple_condition_ra.value_buffer);
  GRN_VOID_INIT(value_buffer);
  grn_obj_reinit_for(ctx, value_buffer, target);

  executor->data.simple_condition_ra.ra = (grn_ra *)target;
  GRN_RA_CACHE_INIT(executor->data.simple_condition_ra.ra,
                    &(executor->data.simple_condition_ra.ra_cache));
  grn_ra_info(ctx,
              executor->data.simple_condition_ra.ra,
              &(executor->data.simple_condition_ra.ra_element_size));

  executor->data.simple_condition_ra.exec = grn_operator_to_exec_func(op);

  constant_buffer = &(executor->data.simple_condition_ra.constant_buffer);
  if (grn_obj_is_reference_column(ctx, target)) {
    GRN_OBJ_INIT(constant_buffer, GRN_BULK, 0, constant->header.domain);
    grn_obj_cast(ctx, constant, constant_buffer, GRN_FALSE);
  } else {
    GRN_VOID_INIT(constant_buffer);
    grn_obj_reinit_for(ctx, constant_buffer, target);
    grn_obj_cast(ctx, constant, constant_buffer, GRN_FALSE);
  }
}

static grn_bool
grn_expr_executor_is_simple_condition_ra(grn_ctx *ctx,
                                         grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;
  grn_expr_code *constant;
  grn_expr_code *operator;

  if (e->codes_curr != 3) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);
  constant = &(e->codes[1]);
  operator = &(e->codes[2]);

  switch (operator->op) {
  case GRN_OP_EQUAL :
  case GRN_OP_NOT_EQUAL :
  case GRN_OP_LESS :
  case GRN_OP_GREATER :
  case GRN_OP_LESS_EQUAL :
  case GRN_OP_GREATER_EQUAL :
    break;
  default :
    return GRN_FALSE;
  }
  if (operator->nargs != 2) {
    return GRN_FALSE;
  }

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (target->nargs != 1) {
    return GRN_FALSE;
  }
  if (target->value->header.type != GRN_COLUMN_FIX_SIZE) {
    return GRN_FALSE;
  }

  if (constant->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (constant->nargs != 1) {
    return GRN_FALSE;
  }
  if (!constant->value) {
    return GRN_FALSE;
  }
  if (constant->value->header.type != GRN_BULK) {
    return GRN_FALSE;
  }

  if (!grn_obj_is_reference_column(ctx, target->value)) {
    grn_obj constant_buffer;
    grn_rc rc;
    GRN_VOID_INIT(&constant_buffer);
    grn_obj_reinit_for(ctx, &constant_buffer, target->value);
    rc = grn_obj_cast(ctx, constant->value, &constant_buffer, GRN_FALSE);
    GRN_OBJ_FIN(ctx, &constant_buffer);
    if (rc != GRN_SUCCESS) {
      return GRN_FALSE;
    }
  }

  grn_expr_executor_init_simple_condition_ra(ctx, executor);

  return GRN_TRUE;
}

static grn_obj *
grn_expr_executor_exec_simple_condition_ra(grn_ctx *ctx,
                                           grn_expr_executor *executor,
                                           grn_id id)
{
  grn_obj *result_buffer = &(executor->data.simple_condition_ra.result_buffer);
  grn_obj *value_buffer = &(executor->data.simple_condition_ra.value_buffer);
  grn_obj *constant_buffer =
    &(executor->data.simple_condition_ra.constant_buffer);

  if (ctx->rc) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
    return result_buffer;
  }

  {
    grn_ra *ra = executor->data.simple_condition_ra.ra;
    grn_ra_cache *ra_cache = &(executor->data.simple_condition_ra.ra_cache);
    unsigned int ra_element_size =
      executor->data.simple_condition_ra.ra_element_size;
    void *raw_value;
    raw_value = grn_ra_ref_cache(ctx, ra, id, ra_cache);
    GRN_BULK_REWIND(value_buffer);
    grn_bulk_write(ctx, value_buffer, raw_value, ra_element_size);
  }

  if (executor->data.simple_condition_ra.exec(ctx,
                                              value_buffer,
                                              constant_buffer)) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_TRUE);
  } else {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
  }
  return result_buffer;
}

static void
grn_expr_executor_fin_simple_condition_ra(grn_ctx *ctx,
                                          grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition_ra.result_buffer));
  GRN_RA_CACHE_FIN(ctx,
                   executor->data.simple_condition_ra.ra,
                   &(executor->data.simple_condition_ra.ra_cache));
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition_ra.value_buffer));
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition_ra.constant_buffer));
}

static void
grn_expr_executor_init_simple_condition(grn_ctx *ctx,
                                        grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *target;
  grn_obj *constant;
  grn_operator op;
  grn_obj *result_buffer;
  grn_obj *value_buffer;
  grn_obj *constant_buffer;
  grn_rc rc;

  target = e->codes[0].value;
  constant = e->codes[1].value;
  op = e->codes[2].op;

  executor->data.simple_condition.need_exec = GRN_TRUE;

  result_buffer = &(executor->data.simple_condition.result_buffer);
  GRN_BOOL_INIT(result_buffer, 0);
  GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);

  value_buffer = &(executor->data.simple_condition.value_buffer);
  GRN_VOID_INIT(value_buffer);
  grn_obj_reinit_for(ctx, value_buffer, target);

  executor->data.simple_condition.exec = grn_operator_to_exec_func(op);

  constant_buffer = &(executor->data.simple_condition.constant_buffer);
  GRN_VOID_INIT(constant_buffer);
  grn_obj_reinit_for(ctx, constant_buffer, target);
  rc = grn_obj_cast(ctx, constant, constant_buffer, GRN_FALSE);
  if (rc != GRN_SUCCESS) {
    grn_obj *type;

    type = grn_ctx_at(ctx, constant_buffer->header.domain);
    if (grn_obj_is_table(ctx, type)) {
      GRN_BOOL_SET(ctx, result_buffer, (op == GRN_OP_NOT_EQUAL));
      executor->data.simple_condition.need_exec = GRN_FALSE;
    } else {
      int type_name_size;
      char type_name[GRN_TABLE_MAX_KEY_SIZE];
      grn_obj inspected;

      type_name_size = grn_obj_name(ctx, type, type_name,
                                    GRN_TABLE_MAX_KEY_SIZE);
      GRN_TEXT_INIT(&inspected, 0);
      grn_inspect(ctx, &inspected, constant);
      ERR(rc,
          "[expr-executor][condition] "
          "failed to cast to <%.*s>: <%.*s>",
          type_name_size, type_name,
          (int)GRN_TEXT_LEN(&inspected), GRN_TEXT_VALUE(&inspected));
    }
  }
}

static grn_bool
grn_expr_executor_is_simple_condition(grn_ctx *ctx,
                                      grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_expr_code *target;
  grn_expr_code *constant;
  grn_expr_code *operator;

  if (e->codes_curr != 3) {
    return GRN_FALSE;
  }

  target = &(e->codes[0]);
  constant = &(e->codes[1]);
  operator = &(e->codes[2]);

  switch (operator->op) {
  case GRN_OP_EQUAL :
  case GRN_OP_NOT_EQUAL :
  case GRN_OP_LESS :
  case GRN_OP_GREATER :
  case GRN_OP_LESS_EQUAL :
  case GRN_OP_GREATER_EQUAL :
    break;
  default :
    return GRN_FALSE;
  }
  if (operator->nargs != 2) {
    return GRN_FALSE;
  }

  if (target->op != GRN_OP_GET_VALUE) {
    return GRN_FALSE;
  }
  if (target->nargs != 1) {
    return GRN_FALSE;
  }
  if (!grn_obj_is_scalar_column(ctx, target->value)) {
    return GRN_FALSE;
  }

  if (constant->op != GRN_OP_PUSH) {
    return GRN_FALSE;
  }
  if (constant->nargs != 1) {
    return GRN_FALSE;
  }
  if (!constant->value) {
    return GRN_FALSE;
  }
  if (constant->value->header.type != GRN_BULK) {
    return GRN_FALSE;
  }

  grn_expr_executor_init_simple_condition(ctx, executor);

  return GRN_TRUE;
}

static grn_obj *
grn_expr_executor_exec_simple_condition(grn_ctx *ctx,
                                        grn_expr_executor *executor,
                                        grn_id id)
{
  grn_expr *e = (grn_expr *)(executor->expr);
  grn_obj *target;
  grn_obj *result_buffer = &(executor->data.simple_condition.result_buffer);
  grn_obj *value_buffer = &(executor->data.simple_condition.value_buffer);
  grn_obj *constant_buffer = &(executor->data.simple_condition.constant_buffer);

  if (ctx->rc) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
    return result_buffer;
  }

  if (!executor->data.simple_condition.need_exec) {
    return result_buffer;
  }

  target = e->codes[0].value;
  GRN_BULK_REWIND(value_buffer);
  grn_obj_get_value(ctx, target, id, value_buffer);

  if (executor->data.simple_condition.exec(ctx, value_buffer, constant_buffer)) {
    GRN_BOOL_SET(ctx, result_buffer, GRN_TRUE);
  } else {
    GRN_BOOL_SET(ctx, result_buffer, GRN_FALSE);
  }
  return result_buffer;
}

static void
grn_expr_executor_fin_simple_condition(grn_ctx *ctx,
                                       grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition.result_buffer));
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition.value_buffer));
  GRN_OBJ_FIN(ctx, &(executor->data.simple_condition.constant_buffer));
}

static bool
grn_expr_executor_init_simple_proc_scorer(grn_ctx *ctx,
                                          grn_expr_executor *executor)
{
  grn_expr_executor_data_simple_proc *data =
    &(executor->data.simple_proc_scorer.data);
  /* The first GRN_OP_GET_REF for _score */
  data->codes_start_offset = 1;
  /* The last GRN_OP_ASSIGN */
  data->codes_end_offset = 1;
  return grn_expr_executor_data_simple_proc_init(ctx, executor, data);
}

static void
grn_expr_executor_fin_simple_proc_scorer(grn_ctx *ctx,
                                         grn_expr_executor *executor)
{
  grn_expr_executor_data_simple_proc *data =
    &(executor->data.simple_proc_scorer.data);
  grn_expr_executor_data_simple_proc_fin(ctx, executor, data);
}

static bool
grn_expr_executor_is_simple_proc_scorer(grn_ctx *ctx,
                                        grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  /*
   * 0: GRN_OP_GET_REF(_score)
   * 1: GRN_OP_PUSH(func)
   * n-2: GRN_OP_CALL(func)
   * n-1: GRN_OP_ASSIGN(_score)
   */
  if (e->codes_curr < 4) {
    return false;
  }

  grn_expr_code *score = &(e->codes[0]);
  if (score->op != GRN_OP_GET_REF) {
    return false;
  }
  if (!grn_obj_is_score_accessor(ctx, score->value)) {
    return false;
  }
  executor->data.simple_proc_scorer.score_column = score->value;

  if (score->modify != (int32_t)(e->codes_curr - 1)) {
    return false;
  }
  grn_expr_code *operation = &(e->codes[score->modify]);
  if (operation->op != GRN_OP_ASSIGN) {
    return false;
  }

  const size_t func_index = 1;
  grn_expr_code *func = &(e->codes[func_index]);
  if (func->op != GRN_OP_PUSH) {
    return false;
  }
  if (!grn_obj_is_function_proc(ctx, func->value)) {
    return false;
  }
  grn_proc *proc = (grn_proc *)(func->value);
  if (!proc->funcs[PROC_INIT] &&
      !proc->funcs[PROC_NEXT] &&
      !proc->funcs[PROC_FIN]) {
    return false;
  }

  if ((func->modify + func_index) != e->codes_curr - 2) {
    return false;
  }
  if (e->codes[func->modify + func_index].op != GRN_OP_CALL) {
    return false;
  }

  if (!grn_expr_executor_init_simple_proc_scorer(ctx, executor)) {
    grn_expr_executor_fin_simple_proc_scorer(ctx, executor);
    return false;
  }

  return true;
}

static grn_obj *
grn_expr_executor_exec_simple_proc_scorer(grn_ctx *ctx,
                                          grn_expr_executor *executor,
                                          grn_id id)
{
  grn_expr_executor_data_simple_proc *data =
    &(executor->data.simple_proc_scorer.data);
  grn_obj *score =
    grn_expr_executor_data_simple_proc_exec(ctx, executor, data, id);
  if (score) {
    grn_obj *score_column = executor->data.simple_proc_scorer.score_column;
    grn_obj_set_value(ctx, score_column, id, score, GRN_OBJ_SET);
  }
  return score;
}

static double
grn_expr_executor_scorer_get_value(grn_ctx *ctx,
                                   grn_expr_executor *executor,
                                   grn_id id,
                                   grn_obj *args,
                                   grn_expr_executor_scorer_data *data)
{
  grn_obj *source_buffer = &(data->source_buffer);
  grn_obj *score_buffer = &(executor->data.scorer.score_buffer);
  if (data->cache) {
    size_t size;
    void *value = grn_column_cache_ref(ctx, data->cache, id, &size);
    if (size == 0) {
      return 0.0;
    }
    if (source_buffer->header.domain == GRN_DB_FLOAT) {
      return *((double *)value);
    }
    GRN_TEXT_SET(ctx, source_buffer, value, size);
  } else {
    GRN_BULK_REWIND(source_buffer);
    grn_obj_get_value(ctx, data->source, id, source_buffer);
    if (GRN_BULK_VSIZE(source_buffer) == 0) {
      return 0.0;
    }
    if (source_buffer->header.domain == GRN_DB_FLOAT) {
      return GRN_FLOAT_VALUE(source_buffer);
    }
  }
  GRN_BULK_REWIND(score_buffer);
  grn_rc rc = grn_obj_cast(ctx, source_buffer, score_buffer, true);
  if (rc == GRN_SUCCESS) {
    return GRN_FLOAT_VALUE(score_buffer);
  } else {
    return 0.0;
  }
}

static double
grn_expr_executor_scorer_push(grn_ctx *ctx,
                              grn_expr_executor *executor,
                              grn_id id,
                              grn_obj *args,
                              grn_expr_executor_scorer_data *data)
{
  grn_obj *source = data->source;
  grn_obj *score_buffer = &(executor->data.scorer.score_buffer);
  if (GRN_BULK_VSIZE(source) == 0) {
    return 0.0;
  }
  if (source->header.domain == GRN_DB_FLOAT) {
    return GRN_FLOAT_VALUE(source);
  }
  GRN_BULK_REWIND(score_buffer);
  grn_rc rc = grn_obj_cast(ctx, source, score_buffer, true);
  if (rc == GRN_SUCCESS) {
    return GRN_FLOAT_VALUE(score_buffer);
  } else {
    return 0.0;
  }
}

static double
grn_expr_executor_scorer_star(grn_ctx *ctx,
                              grn_expr_executor *executor,
                              grn_id id,
                              grn_obj *args,
                              grn_expr_executor_scorer_data *data)
{
  double right;
  GRN_FLOAT_POP(args, right);
  double left;
  GRN_FLOAT_POP(args, left);
  return left * right;
}

static double
grn_expr_executor_scorer_plus(grn_ctx *ctx,
                              grn_expr_executor *executor,
                              grn_id id,
                              grn_obj *args,
                              grn_expr_executor_scorer_data *data)
{
  double right;
  GRN_FLOAT_POP(args, right);
  double left;
  GRN_FLOAT_POP(args, left);
  return left + right;
}

static double
grn_expr_executor_scorer_minus(grn_ctx *ctx,
                               grn_expr_executor *executor,
                               grn_id id,
                               grn_obj *args,
                               grn_expr_executor_scorer_data *data)
{
  double right;
  GRN_FLOAT_POP(args, right);
  double left;
  GRN_FLOAT_POP(args, left);
  return left - right;
}

static bool
grn_expr_executor_init_scorer(grn_ctx *ctx,
                              grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  GRN_FLOAT_INIT(&(executor->data.scorer.args), GRN_OBJ_VECTOR);
  GRN_FLOAT_INIT(&(executor->data.scorer.score_buffer), 0);

  executor->data.scorer.n_funcs = 0;
  /* The first GRN_OP_GET_REF and the last GRN_OP_ASSIGN are needless. */
  size_t n_allocated_funcs = e->codes_curr - 2;
  executor->data.scorer.funcs = GRN_MALLOCN(grn_expr_executor_scorer_func,
                                            n_allocated_funcs);
  if (!executor->data.scorer.funcs) {
    return false;
  }
  executor->data.scorer.datas = GRN_MALLOCN(grn_expr_executor_scorer_data,
                                            n_allocated_funcs);
  if (!executor->data.scorer.datas) {
    GRN_FREE(executor->data.scorer.funcs);
    return false;
  }
  executor->data.scorer.n_allocated_funcs = n_allocated_funcs;
  {
    size_t i;
    for (i = 0; i < executor->data.scorer.n_allocated_funcs; i++) {
      executor->data.scorer.funcs[i] = NULL;
      GRN_VOID_INIT(&(executor->data.scorer.datas[i].source_buffer));
      executor->data.scorer.datas[i].source = NULL;
      executor->data.scorer.datas[i].cache = NULL;
      executor->data.scorer.datas[i].n_required_args = 0;
    }
  }

  {
    uint32_t i;
    for (i = 1; i < (e->codes_curr - 1); i++) {
      grn_expr_code *code = &(e->codes[i]);
      size_t nth_func = executor->data.scorer.n_funcs;
      grn_expr_executor_scorer_data *data =
        &(executor->data.scorer.datas[nth_func]);
      switch (code->op) {
      case GRN_OP_GET_VALUE :
        data->source = code->value;
        data->cache = grn_column_cache_open(ctx, data->source);
        grn_id range = grn_obj_get_range(ctx, data->source);
        grn_obj_reinit(ctx, &(data->source_buffer), range, 0);
        if (data->cache) {
          data->source_buffer.header.impl_flags = GRN_OBJ_DO_SHALLOW_COPY;
        }
        executor->data.scorer.funcs[nth_func] =
          grn_expr_executor_scorer_get_value;
        break;
      case GRN_OP_PUSH :
        data->source = code->value;
        executor->data.scorer.funcs[nth_func] =
          grn_expr_executor_scorer_push;
        break;
      case GRN_OP_STAR :
        data->n_required_args = 2;
        executor->data.scorer.funcs[nth_func] =
          grn_expr_executor_scorer_star;
        break;
      case GRN_OP_PLUS :
        data->n_required_args = 2;
        executor->data.scorer.funcs[nth_func] =
          grn_expr_executor_scorer_plus;
        break;
      case GRN_OP_MINUS :
        data->n_required_args = 2;
        executor->data.scorer.funcs[nth_func] =
          grn_expr_executor_scorer_minus;
        break;
      default :
        return false;
      }
      executor->data.scorer.n_funcs++;
    }
  }

  return true;
}

static void
grn_expr_executor_fin_scorer(grn_ctx *ctx,
                             grn_expr_executor *executor)
{
  GRN_OBJ_FIN(ctx, &(executor->data.scorer.args));
  GRN_OBJ_FIN(ctx, &(executor->data.scorer.score_buffer));

  size_t n_allocated_funcs = executor->data.scorer.n_allocated_funcs;
  if (n_allocated_funcs == 0) {
    return;
  }

  size_t i;
  for (i = 0; i < n_allocated_funcs; i++) {
    grn_expr_executor_scorer_data *data = &(executor->data.scorer.datas[i]);
    GRN_OBJ_FIN(ctx, &(data->source_buffer));
    if (data->cache) {
      grn_column_cache_close(ctx, data->cache);
    }
  }
  GRN_FREE(executor->data.scorer.funcs);
  GRN_FREE(executor->data.scorer.datas);
}

static bool
grn_expr_executor_is_scorer(grn_ctx *ctx,
                            grn_expr_executor *executor)
{
  grn_expr *e = (grn_expr *)(executor->expr);

  if (e->codes_curr < 2) {
    return false;
  }

  grn_expr_code *score = &(e->codes[0]);
  if (score->op != GRN_OP_GET_REF) {
    return false;
  }
  if (!grn_obj_is_score_accessor(ctx, score->value)) {
    return false;
  }
  executor->data.scorer.score_column = score->value;

  grn_expr_code *operation = &(e->codes[score->modify]);
  if (operation->op != GRN_OP_ASSIGN) {
    return false;
  }

  if (!grn_expr_executor_init_scorer(ctx, executor)) {
    grn_expr_executor_fin_scorer(ctx, executor);
    return false;
  }

  return true;
}

static grn_obj *
grn_expr_executor_exec_scorer(grn_ctx *ctx,
                              grn_expr_executor *executor,
                              grn_id id)
{
  double score = 0.0;
  grn_obj *args = &(executor->data.scorer.args);
  GRN_BULK_REWIND(args);
  size_t n_funcs = executor->data.scorer.n_funcs;
  size_t i;
  for (i = 0; i < n_funcs; i++) {
    grn_expr_executor_scorer_func func = executor->data.scorer.funcs[i];
    grn_expr_executor_scorer_data *data = &(executor->data.scorer.datas[i]);
    double value = func(ctx, executor, id, args, data);
    GRN_FLOAT_PUT(ctx, args, value);
    score = value;
  }
  grn_obj *score_column = executor->data.scorer.score_column;
  grn_obj *score_buffer = &(executor->data.scorer.score_buffer);
  GRN_FLOAT_SET(ctx, score_buffer, score);
  grn_obj_set_value(ctx, score_column, id, score_buffer, GRN_OBJ_SET);

  return score_buffer;
}

grn_rc
grn_expr_executor_init(grn_ctx *ctx,
                       grn_expr_executor *executor,
                       grn_obj *expr)
{
  GRN_API_ENTER;

  if (!grn_obj_is_expr(ctx, expr)) {
    grn_rc rc = ctx->rc;
    grn_obj inspected;
    GRN_TEXT_INIT(&inspected, 0);
    grn_inspect(ctx, &inspected, expr);
    if (rc == GRN_SUCCESS) {
      rc = GRN_INVALID_ARGUMENT;
    }
    ERR(rc,
        "[expr-executor][init] invalid expression: %.*s",
        (int)GRN_TEXT_LEN(&inspected),
        GRN_TEXT_VALUE(&inspected));
    GRN_OBJ_FIN(ctx, &inspected);
    GRN_API_RETURN(ctx->rc);
  }

  executor->expr = expr;
  executor->variable = grn_expr_get_var_by_offset(ctx, expr, 0);
  if (grn_expr_executor_is_constant(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_constant;
    executor->fin = grn_expr_executor_fin_constant;
  } else if (grn_expr_executor_is_value(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_value;
    executor->fin = grn_expr_executor_fin_value;
#ifdef GRN_SUPPORT_REGEXP
  } else if (grn_expr_executor_is_simple_regexp(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_regexp;
    executor->fin = grn_expr_executor_fin_simple_regexp;
#endif /* GRN_SUPPORT_REGEXP */
  } else if (grn_expr_executor_is_simple_match(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_match;
    executor->fin = grn_expr_executor_fin_simple_match;
  } else if (grn_expr_executor_is_simple_proc(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_proc;
    executor->fin = grn_expr_executor_fin_simple_proc;
  } else if (grn_expr_executor_is_simple_condition_constant(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_condition_constant;
    executor->fin = grn_expr_executor_fin_simple_condition_constant;
  } else if (grn_expr_executor_is_simple_condition_ra(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_condition_ra;
    executor->fin = grn_expr_executor_fin_simple_condition_ra;
  } else if (grn_expr_executor_is_simple_condition(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_condition;
    executor->fin = grn_expr_executor_fin_simple_condition;
  } else if (grn_expr_executor_is_simple_proc_scorer(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_simple_proc_scorer;
    executor->fin = grn_expr_executor_fin_simple_proc_scorer;
  } else if (grn_expr_executor_is_scorer(ctx, executor)) {
    executor->exec = grn_expr_executor_exec_scorer;
    executor->fin = grn_expr_executor_fin_scorer;
  } else {
    grn_expr_executor_init_general(ctx, executor);
    executor->exec = grn_expr_executor_exec_general;
    executor->fin = grn_expr_executor_fin_general;
  }

  GRN_API_RETURN(ctx->rc);
}

grn_rc
grn_expr_executor_fin(grn_ctx *ctx,
                      grn_expr_executor *executor)
{
  GRN_API_ENTER;

  if (!executor) {
    GRN_API_RETURN(GRN_SUCCESS);
  }

  executor->fin(ctx, executor);

  GRN_API_RETURN(GRN_SUCCESS);
}

grn_expr_executor *
grn_expr_executor_open(grn_ctx *ctx, grn_obj *expr)
{
  grn_expr_executor *executor;

  GRN_API_ENTER;

  executor = GRN_CALLOC(sizeof(grn_expr_executor));
  if (!executor) {
    char errbuf[GRN_CTX_MSGSIZE];
    grn_strcpy(errbuf, GRN_CTX_MSGSIZE, ctx->errbuf);
    ERR(ctx->rc,
        "[expr-executor][open] failed to allocate: %s",
        errbuf);
    GRN_API_RETURN(NULL);
  }

  grn_expr_executor_init(ctx, executor, expr);

  if (ctx->rc != GRN_SUCCESS) {
    GRN_FREE(executor);
    executor = NULL;
  }

  GRN_API_RETURN(executor);
}

grn_rc
grn_expr_executor_close(grn_ctx *ctx, grn_expr_executor *executor)
{
  GRN_API_ENTER;

  if (!executor) {
    GRN_API_RETURN(GRN_SUCCESS);
  }

  grn_expr_executor_fin(ctx, executor);
  GRN_FREE(executor);

  GRN_API_RETURN(GRN_SUCCESS);
}

grn_obj *
grn_expr_executor_exec(grn_ctx *ctx, grn_expr_executor *executor, grn_id id)
{
  grn_obj *value;

  GRN_API_ENTER;

  if (!executor) {
    GRN_API_RETURN(NULL);
  }

  value = executor->exec(ctx, executor, id);
  if (ctx->rc != GRN_SUCCESS) {
    value = NULL;
  }

  GRN_API_RETURN(value);
}
