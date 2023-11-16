/*-------------------------------------------------------------------------
 *
 * ybgate_api.c
 *	  YbGate interface functions.
 *	  YbGate allows to execute Postgres code from DocDB
 *
 * Copyright (c) Yugabyte, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  You may obtain a copy
 * of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * IDENTIFICATION
 *	  src/backend/ybgate/ybgate_api.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ybgate/ybgate_api.h"

#include "access/htup_details.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "catalog/yb_type.h"
#include "common/int.h"
#include "executor/execExpr.h"
#include "executor/executor.h"
#include "mb/pg_wchar.h"
#include "nodes/execnodes.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/primnodes.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/rowtypes.h"
#include "utils/sampling.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "funcapi.h"
#include "pg_yb_utils.h"

YbgStatus YbgInit()
{
	PG_SETUP_ERROR_REPORTING();

	SetDatabaseEncoding(PG_UTF8);

	PG_STATUS_OK();
}

//-----------------------------------------------------------------------------
// Memory Context
//-----------------------------------------------------------------------------


YbgMemoryContext YbgGetCurrentMemoryContext()
{
	return GetThreadLocalCurrentMemoryContext();
}

YbgMemoryContext YbgSetCurrentMemoryContext(YbgMemoryContext memctx)
{
	return SetThreadLocalCurrentMemoryContext(memctx);
}

YbgStatus YbgCreateMemoryContext(YbgMemoryContext parent,
								 const char *name,
								 YbgMemoryContext *memctx)
{
	PG_SETUP_ERROR_REPORTING();

	*memctx = CreateThreadLocalMemoryContext(parent, name);

	PG_STATUS_OK();
}

YbgStatus YbgPrepareMemoryContext()
{
	PG_SETUP_ERROR_REPORTING();

	PrepareThreadLocalCurrentMemoryContext();

	PG_STATUS_OK();
}

YbgStatus YbgResetMemoryContext()
{
	PG_SETUP_ERROR_REPORTING();

	ResetThreadLocalCurrentMemoryContext();

	PG_STATUS_OK();
}

YbgStatus YbgDeleteMemoryContext()
{
	PG_SETUP_ERROR_REPORTING();

	DeleteThreadLocalCurrentMemoryContext();

	PG_STATUS_OK();
}

//-----------------------------------------------------------------------------
// Types
//-----------------------------------------------------------------------------

YbgStatus YbgGetTypeTable(const YBCPgTypeEntity **type_table, int *count)
{
	PG_SETUP_ERROR_REPORTING();

	YbGetTypeTable(type_table, count);

	PG_STATUS_OK();
}

YbgStatus
YbgGetPrimitiveTypeOid(uint32_t type_oid, char typtype, uint32_t typbasetype,
					   uint32_t *primitive_type_oid)
{
	PG_SETUP_ERROR_REPORTING();
	*primitive_type_oid = YbGetPrimitiveTypeOid(type_oid, typtype, typbasetype);
	PG_STATUS_OK();
}

//-----------------------------------------------------------------------------
// Expression Evaluation
//-----------------------------------------------------------------------------

/*
 * Expression context for evaluating a YSQL expression from DocDB.
 * Currently includes the table row values to resolve scan variables.
 * TODO Eventually this should probably also have schema/type information.
 */
struct YbgExprContextData
{
	// Values from table row.
	int32_t min_attno;
	int32_t max_attno;
	Datum *attr_vals;
	Bitmapset *attr_nulls;
};

/*
 * Evaluate an expression against an expression context.
 * Currently assumes the expression has been checked by the planner to only
 * allow immutable functions and the node types handled below.
 * TODO: this should use the general YSQL/PG expression evaluation framework, but
 * that requires syscaches and other dependencies to be fully initialized.
 */
static Datum evalExpr(YbgExprContext ctx, Expr* expr, bool *is_null)
{
	switch (expr->type)
	{
		case T_FuncExpr:
		case T_OpExpr:
		{
			Oid			funcid;
			Oid			inputcollid;
			List	   *args;
			ListCell   *lc;

			/* Get the (underlying) function info. */
			if (IsA(expr, FuncExpr))
			{
				FuncExpr *func_expr = castNode(FuncExpr, expr);
				args = func_expr->args;
				funcid = func_expr->funcid;
				inputcollid = func_expr->inputcollid;
			}
			else /* (IsA(expr, OpExpr)) */
			{
				OpExpr *op_expr = castNode(OpExpr, expr);
				args = op_expr->args;
				funcid = op_expr->opfuncid;
				inputcollid = op_expr->inputcollid;
			}

			FmgrInfo *flinfo = palloc0(sizeof(FmgrInfo));
			FunctionCallInfoData fcinfo;

			fmgr_info(funcid, flinfo);
			InitFunctionCallInfoData(fcinfo,
									 flinfo,
									 list_length(args),
									 inputcollid,
									 NULL,
									 NULL);
			int i = 0;
			foreach(lc, args)
			{
				Expr *arg = (Expr *) lfirst(lc);
				fcinfo.arg[i] = evalExpr(ctx, arg, &fcinfo.argnull[i]);
				/*
				 * Strict functions are guaranteed to return NULL if any of
				 * their arguments are NULL.
				 */
				if (flinfo->fn_strict && fcinfo.argnull[i]) {
					*is_null = true;
					return (Datum) 0;
				}
				i++;
			}
			Datum result = FunctionCallInvoke(&fcinfo);
			*is_null = fcinfo.isnull;
			return result;
		}
		case T_RelabelType:
		{
			RelabelType *rt = castNode(RelabelType, expr);
			return evalExpr(ctx, rt->arg, is_null);
		}
		case T_NullTest:
		{
			NullTest   *nt = castNode(NullTest, expr);
			bool		arg_is_null;
			evalExpr(ctx, nt->arg, &arg_is_null);
			*is_null = false;
			return (Datum) (nt->nulltesttype == IS_NULL) == arg_is_null;
		}
		case T_BoolExpr:
		{
			BoolExpr   *be = castNode(BoolExpr, expr);
			ListCell   *lc;
			Expr	   *arg;
			Datum		arg_value;
			bool		arg_is_null;
			switch (be->boolop)
			{
				case AND_EXPR:
					*is_null = false;
					foreach(lc, be->args)
					{
						arg = (Expr *) lfirst(lc);
						arg_value = evalExpr(ctx, arg, &arg_is_null);
						if (arg_is_null)
						{
							*is_null = true;
						}
						else if (!arg_value)
						{
							*is_null = false;
							return (Datum) false;
						}
					}
					return *is_null ? (Datum) 0 : (Datum) true;
				case OR_EXPR:
					*is_null = false;
					foreach(lc, be->args)
					{
						arg = (Expr *) lfirst(lc);
						arg_value = evalExpr(ctx, arg, &arg_is_null);
						if (arg_is_null)
						{
							*is_null = true;
						}
						else if (arg_value)
						{
							*is_null = false;
							return (Datum) true;
						}
					}
					return *is_null ? (Datum) 0 : (Datum) false;
				case NOT_EXPR:
					arg = (Expr *) linitial(be->args);
					arg_value = evalExpr(ctx, arg, is_null);
					return *is_null ? (Datum) 0 : (Datum) (!arg_value);
				default:
					/* Planner should ensure we never get here. */
					ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR), errmsg(
						"Unsupported boolop received by DocDB")));
					break;
			}
			return true;
		}
		case T_CaseExpr:
		{
			CaseExpr   *ce = castNode(CaseExpr, expr);
			ListCell   *lc;
			/*
			 * Support for implicit equality comparison would require catalog
			 * lookup to find equality operation for the argument data type.
			 */
			if (ce->arg)
				ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR), errmsg(
					"Unsupported CASE expression received by DocDB")));
			/*
			 * Evaluate WHEN clause expressions one by one, if any evaluation
			 * result is true, evaluate and return respective result expression
			 */
			foreach(lc, ce->args)
			{
				CaseWhen *cw = castNode(CaseWhen, lfirst(lc));
				bool arg_is_null;
				if (evalExpr(ctx, cw->expr, &arg_is_null))
					return evalExpr(ctx, cw->result, is_null);
			}
			/* None of the exprerssions was true, so evaluate the default. */
			if (ce->defresult)
				return evalExpr(ctx, ce->defresult, is_null);
			/* If default is not specified, return NULL */
			*is_null = true;
			return (Datum) 0;
		}
		case T_Const:
		{
			Const* const_expr = castNode(Const, expr);
			*is_null = const_expr->constisnull;
			return const_expr->constvalue;
		}
		case T_Var:
		{
			Var* var_expr = castNode(Var, expr);
			int32_t att_idx = var_expr->varattno - ctx->min_attno;
			*is_null = bms_is_member(att_idx, ctx->attr_nulls);
			return ctx->attr_vals[att_idx];
		}
		default:
			/* Planner should ensure we never get here. */
			ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg(
				"Unsupported YSQL expression received by DocDB")));
			break;
	}
	*is_null = true;
	return (Datum) 0;
}

YbgStatus YbgExprContextCreate(int32_t min_attno, int32_t max_attno, YbgExprContext *expr_ctx)
{
	PG_SETUP_ERROR_REPORTING();

	YbgExprContext ctx = (YbgExprContext) palloc0(sizeof(struct YbgExprContextData));
	ctx->min_attno = min_attno;
	ctx->max_attno = max_attno;
	int32_t num_attrs = max_attno - min_attno + 1;
	ctx->attr_vals = (Datum *) palloc0(sizeof(Datum) * num_attrs);
	ctx->attr_nulls = NULL;

	*expr_ctx = ctx;
	PG_STATUS_OK();
}

YbgStatus YbgExprContextReset(YbgExprContext expr_ctx)
{
	PG_SETUP_ERROR_REPORTING();

	int32_t num_attrs = expr_ctx->max_attno - expr_ctx->min_attno + 1;
	memset(expr_ctx->attr_vals, 0, sizeof(Datum) * num_attrs);
	expr_ctx->attr_nulls = NULL;

	PG_STATUS_OK();
}

YbgStatus YbgExprContextAddColValue(YbgExprContext expr_ctx,
                                    int32_t attno,
                                    uint64_t datum,
                                    bool is_null)
{
	PG_SETUP_ERROR_REPORTING();

	if (is_null)
	{
		expr_ctx->attr_nulls = bms_add_member(expr_ctx->attr_nulls, attno - expr_ctx->min_attno);
	}
	else
	{
		expr_ctx->attr_vals[attno - expr_ctx->min_attno] = (Datum) datum;
	}

	PG_STATUS_OK();
}

YbgStatus YbgPrepareExpr(char* expr_cstring, YbgPreparedExpr *expr)
{
	PG_SETUP_ERROR_REPORTING();
	*expr = (YbgPreparedExpr) stringToNode(expr_cstring);
	PG_STATUS_OK();
}

YbgStatus YbgExprType(const YbgPreparedExpr expr, int32_t *typid)
{
	PG_SETUP_ERROR_REPORTING();
	*typid = exprType((Node *) expr);
	PG_STATUS_OK();
}

YbgStatus YbgExprTypmod(const YbgPreparedExpr expr, int32_t *typmod)
{
	PG_SETUP_ERROR_REPORTING();
	*typmod = exprTypmod((Node *) expr);
	PG_STATUS_OK();
}

YbgStatus YbgExprCollation(const YbgPreparedExpr expr, int32_t *collid)
{
	PG_SETUP_ERROR_REPORTING();
	*collid = exprCollation((Node *) expr);
	PG_STATUS_OK();
}

YbgStatus YbgEvalExpr(YbgPreparedExpr expr, YbgExprContext expr_ctx, uint64_t *datum, bool *is_null)
{
	PG_SETUP_ERROR_REPORTING();
	*datum = (uint64_t) evalExpr(expr_ctx, expr, is_null);
	PG_STATUS_OK();
}

YbgStatus YbgSplitArrayDatum(uint64_t datum,
			     const int type,
			     uint64_t **result_datum_array,
			     int *const nelems)
{
	PG_SETUP_ERROR_REPORTING();
	ArrayType  *arr = DatumGetArrayTypeP((Datum)datum);

	if (ARR_NDIM(arr) != 1 || ARR_HASNULL(arr) || ARR_ELEMTYPE(arr) != type)
		return YbgStatusCreateError(
				"Type of given datum array does not match the given type",
				__FILE__, __LINE__);

	int32 elmlen;
	bool elmbyval;
	char elmalign;
	/*
	 * Ideally this information should come from pg_type or from caller instead of hardcoding
	 * here. However this could be okay as PG also has this harcoded in few places.
	 */
	switch (type)
	{
		case TEXTOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case XMLOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case LINEOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case CIRCLEOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case CASHOID:
			elmlen = sizeof(int64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case BOOLOID:
			elmlen = sizeof(bool);
			elmbyval = true;
			elmalign = 'i';
			break;
		case BYTEAOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case CHAROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case NAMEOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case INT2OID:
			elmlen = 2;
			elmbyval = true;
			elmalign = 's';
			break;
		case INT2VECTOROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case INT4OID:
			elmlen = sizeof(int32);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGPROCOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case OIDOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case TIDOID:
			elmlen = sizeof(ItemPointerData);
			elmbyval = true;
			elmalign = 'i';
			break;
		case XIDOID:
			elmlen = sizeof(TransactionId);
			elmbyval = true;
			elmalign = 'i';
			break;
		case CIDOID:
			elmlen = sizeof(CommandId);
			elmbyval = true;
			elmalign = 'i';
			break;
		case OIDVECTOROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case BPCHAROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case VARCHAROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case INT8OID:
			elmlen = sizeof(int64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case POINTOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case LSEGOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case PATHOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case BOXOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case FLOAT4OID:
			elmlen = sizeof(int64);
			elmbyval = false;
			elmalign = 'i';
			break;
		case FLOAT8OID:
			elmlen = 8;
			elmbyval = FLOAT8PASSBYVAL;
			elmalign = 'd';
			break;
		case ABSTIMEOID:
			elmlen = sizeof(int32);
			elmbyval = true;
			elmalign = 'i';
			break;
		case RELTIMEOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case TINTERVALOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case ACLITEMOID:
			elmlen = sizeof(AclItem);
			elmbyval = true;
			elmalign = 'i';
			break;
		case MACADDROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case MACADDR8OID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case INETOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case CSTRINGOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'c';
			break;
		case TIMESTAMPOID:
			elmlen = sizeof(int64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case DATEOID:
			elmlen = sizeof(int32);
			elmbyval = true;
			elmalign = 'i';
			break;
		case TIMEOID:
			elmlen = sizeof(int64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case TIMESTAMPTZOID:
			elmlen = sizeof(int64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case INTERVALOID:
			elmlen = sizeof(Interval);
			elmbyval = false;
			elmalign = 'd';
			break;
		case NUMERICOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case TIMETZOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case BITOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case VARBITOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case REGPROCEDUREOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGOPEROID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGOPERATOROID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGCLASSOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGTYPEOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGROLEOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGNAMESPACEOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case UUIDOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case LSNOID:
			elmlen = sizeof(uint64);
			elmbyval = true;
			elmalign = 'i';
			break;
		case TSVECTOROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case GTSVECTOROID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case TSQUERYOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case REGCONFIGOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case REGDICTIONARYOID:
			elmlen = sizeof(Oid);
			elmbyval = true;
			elmalign = 'i';
			break;
		case JSONBOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case TXID_SNAPSHOTOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case RECORDOID:
			elmlen = -1;
			elmbyval = false;
			elmalign = 'i';
			break;
		case ANYOID:
			elmlen = sizeof(int32);
			elmbyval = true;
			elmalign = 'i';
			break;
		/* TODO: Extend support to other types as well. */
		default:
			return YbgStatusCreateError(
					"Only Text type supported for split of datum of array types",
					__FILE__, __LINE__);
	}
	deconstruct_array(arr, type, elmlen, elmbyval, elmalign,
			  (Datum**)result_datum_array, NULL /* nullsp */, nelems);
	PG_STATUS_OK();
}

//-----------------------------------------------------------------------------
// Relation sampling
//-----------------------------------------------------------------------------

struct YbgReservoirStateData {
	ReservoirStateData rs;
};

YbgStatus YbgSamplerCreate(double rstate_w, uint64_t randstate, YbgReservoirState *yb_rs)
{
	PG_SETUP_ERROR_REPORTING();
	YbgReservoirState rstate = (YbgReservoirState) palloc0(sizeof(struct YbgReservoirStateData));
	rstate->rs.W = rstate_w;
	Uint64ToSamplerRandomState(rstate->rs.randstate, randstate);
	*yb_rs = rstate;
	PG_STATUS_OK();
}

YbgStatus YbgSamplerGetState(YbgReservoirState yb_rs, double *rstate_w, uint64_t *randstate)
{
	PG_SETUP_ERROR_REPORTING();
	*rstate_w = yb_rs->rs.W;
	*randstate = SamplerRandomStateToUint64(yb_rs->rs.randstate);
	PG_STATUS_OK();
}

YbgStatus YbgSamplerRandomFract(YbgReservoirState yb_rs, double *value)
{
	PG_SETUP_ERROR_REPORTING();
	ReservoirState rs = &yb_rs->rs;
	*value = sampler_random_fract(rs->randstate);
	PG_STATUS_OK();
}

YbgStatus YbgReservoirGetNextS(YbgReservoirState yb_rs, double t, int n, double *s)
{
	PG_SETUP_ERROR_REPORTING();
	*s = reservoir_get_next_S(&yb_rs->rs, t, n);
	PG_STATUS_OK();
}

char* DecodeDatum(char const* fn_name, uintptr_t datum)
{
	FmgrInfo   *finfo;
	finfo = palloc0(sizeof(FmgrInfo));
	Oid id = fmgr_internal_function(fn_name);
	fmgr_info(id, finfo);
	char* tmp = OutputFunctionCall(finfo, (uintptr_t)datum);
	return tmp;
}

char* DecodeTZDatum(char const* fn_name, uintptr_t datum, const char *timezone, bool from_YB)
{
	FmgrInfo   *finfo;
	finfo = palloc0(sizeof(FmgrInfo));
	Oid id = fmgr_internal_function(fn_name);
	fmgr_info(id, finfo);

	DatumDecodeOptions decodeOptions;
	decodeOptions.timezone = timezone;
	decodeOptions.from_YB = from_YB;
	decodeOptions.range_datum_decode_options = NULL;
	return DatumGetCString(FunctionCall2(finfo, (uintptr_t)datum,
				PointerGetDatum(&decodeOptions)));
}

char* DecodeArrayDatum(char const* arr_fn_name, uintptr_t datum,
		int16_t elem_len, bool elem_by_val, char elem_align, char elem_delim, bool from_YB,
		char const* fn_name, const char *timezone, char option)
{
	FmgrInfo   *arr_finfo;
	arr_finfo = palloc0(sizeof(FmgrInfo));
	Oid arr_id = fmgr_internal_function(arr_fn_name);
	fmgr_info(arr_id, arr_finfo);

	FmgrInfo   *elem_finfo;
	elem_finfo = palloc0(sizeof(FmgrInfo));
	Oid elem_id = fmgr_internal_function(fn_name);
	fmgr_info(elem_id, elem_finfo);

	DatumDecodeOptions decodeOptions;
	decodeOptions.is_array = true;
	decodeOptions.elem_by_val = elem_by_val;
	decodeOptions.from_YB = from_YB;
	decodeOptions.elem_align = elem_align;
	decodeOptions.elem_delim = elem_delim;
	decodeOptions.option = option;
	decodeOptions.elem_len = elem_len;
	//decodeOptions.datum = datum;
	decodeOptions.elem_finfo = elem_finfo;
	decodeOptions.timezone = timezone;
	decodeOptions.range_datum_decode_options = NULL;

	char* tmp = DatumGetCString(FunctionCall2(arr_finfo, (uintptr_t)datum,
				PointerGetDatum(&decodeOptions)));
	return tmp;
}

char* DecodeRangeDatum(char const* range_fn_name, uintptr_t datum,
		int16_t elem_len, bool elem_by_val, char elem_align, char option, bool from_YB,
		char const* elem_fn_name, int range_type, const char *timezone)
{
	FmgrInfo   *range_finfo;
	range_finfo = palloc0(sizeof(FmgrInfo));
	Oid range_id = fmgr_internal_function(range_fn_name);
	fmgr_info(range_id, range_finfo);

	FmgrInfo   *elem_finfo;
	elem_finfo = palloc0(sizeof(FmgrInfo));
	Oid elem_id = fmgr_internal_function(elem_fn_name);
	fmgr_info(elem_id, elem_finfo);

	DatumDecodeOptions decodeOptions;
	decodeOptions.is_array = false;
	decodeOptions.elem_by_val = elem_by_val;
	decodeOptions.from_YB = from_YB;
	decodeOptions.elem_align = elem_align;
	decodeOptions.option = option;
	decodeOptions.elem_len = elem_len;
	decodeOptions.range_type = range_type;
	//decodeOptions.datum = datum;
	decodeOptions.elem_finfo = elem_finfo;
	decodeOptions.timezone = timezone;
	decodeOptions.range_datum_decode_options = NULL;

	char* tmp = DatumGetCString(FunctionCall2(range_finfo, (uintptr_t)datum,
				PointerGetDatum(&decodeOptions)));
	return tmp;
}

char* DecodeRangeArrayDatum(char const* arr_fn_name, uintptr_t datum,
		int16_t elem_len, int16_t range_len, bool elem_by_val, bool range_by_val,
		char elem_align, char range_align, char elem_delim, char option, char range_option,
		bool from_YB, char const* elem_fn_name, char const* range_fn_name, int range_type,
		const char *timezone)
{
	FmgrInfo   *arr_finfo;
	arr_finfo = palloc0(sizeof(FmgrInfo));
	Oid arr_id = fmgr_internal_function(arr_fn_name);
	fmgr_info(arr_id, arr_finfo);

	FmgrInfo   *range_finfo;
	range_finfo = palloc0(sizeof(FmgrInfo));
	Oid range_id = fmgr_internal_function(range_fn_name);
	fmgr_info(range_id, range_finfo);

	FmgrInfo   *elem_finfo;
	elem_finfo = palloc0(sizeof(FmgrInfo));
	Oid elem_id = fmgr_internal_function(elem_fn_name);
	fmgr_info(elem_id, elem_finfo);

	DatumDecodeOptions range_decodeOptions;
	range_decodeOptions.is_array = false;
	range_decodeOptions.elem_by_val = range_by_val;
	range_decodeOptions.from_YB = from_YB;
	range_decodeOptions.elem_align = range_align;
	range_decodeOptions.option = range_option;
	range_decodeOptions.elem_len = range_len;
	range_decodeOptions.range_type = range_type;
	range_decodeOptions.elem_finfo = range_finfo;
	range_decodeOptions.timezone = timezone;
	range_decodeOptions.range_datum_decode_options = NULL;

	DatumDecodeOptions arr_decodeOptions;
	arr_decodeOptions.is_array = true;
	arr_decodeOptions.elem_by_val = elem_by_val;
	arr_decodeOptions.from_YB = from_YB;
	arr_decodeOptions.elem_align = elem_align;
	arr_decodeOptions.elem_delim = elem_delim;
	arr_decodeOptions.option = option;
	arr_decodeOptions.elem_len = elem_len;
	arr_decodeOptions.elem_finfo = elem_finfo;
	arr_decodeOptions.timezone = timezone;
	arr_decodeOptions.range_datum_decode_options = &range_decodeOptions;

	char* tmp = DatumGetCString(FunctionCall2(arr_finfo, (uintptr_t)datum,
				PointerGetDatum(&arr_decodeOptions)));
	return tmp;
}

char *
DecodeRecordDatum(uintptr_t datum, void *attrs, size_t natts)
{
	FmgrInfo *finfo = palloc0(sizeof(FmgrInfo));

	HeapTupleHeader rec = DatumGetHeapTupleHeader(datum);
	Oid				tupType = HeapTupleHeaderGetTypeId(rec);
	int32			tupTypmod = HeapTupleHeaderGetTypMod(rec);
	TupleDesc		tupdesc = CreateTupleDesc(natts, true, attrs);
	finfo->fn_extra = MemoryContextAlloc(GetCurrentMemoryContext(),
										 offsetof(RecordIOData, columns) +
											 natts * sizeof(ColumnIOData));
	RecordIOData *my_extra = (RecordIOData *) finfo->fn_extra;
	my_extra->record_type = tupType;
	my_extra->record_typmod = tupTypmod;
	my_extra->ncolumns = natts;
	for (size_t i = 0; i < natts; i++)
	{
		ColumnIOData	 *column_info = &my_extra->columns[i];
		Form_pg_attribute att = TupleDescAttr(tupdesc, i);
		column_info->typiofunc =
			fmgr_internal_function(GetOutFuncName(att->atttypid));
		fmgr_info(column_info->typiofunc, &column_info->proc);
		column_info->column_type = att->atttypid;
	}
	return DatumGetCString(record_out_internal(rec, &tupdesc, finfo));
}

char *
GetOutFuncName(const int pg_data_type)
{
	char *func_name;
	switch (pg_data_type)
	{
		case BOOLOID:
			func_name = "boolout";
			break;
		case BYTEAOID:
			func_name = "byteaout";
			break;
		case CHAROID:
			func_name = "charout";
			break;
		case NAMEOID:
			func_name = "nameout";
			break;
		case INT8OID:
			func_name = "int8out";
			break;
		case INT2OID:
			func_name = "int2out";
			break;
		case INT4OID:
			func_name = "int4out";
			break;
		case REGPROCOID:
			func_name = "regprocout";
			break;
		case TEXTOID:
			func_name = "textout";
			break;
		case OIDOID:
			func_name = "oidout";
			break;
		case TIDOID:
			func_name = "tidout";
			break;
		case XIDOID:
			func_name = "xidout";
			break;
		case CIDOID:
			func_name = "cidout";
			break;
		case JSONOID:
			func_name = "json_out";
			break;
		case XMLOID:
			func_name = "xml_out";
			break;
		case PGNODETREEOID:
			func_name = "pg_node_tree_out";
			break;
		case PGNDISTINCTOID:
			func_name = "pg_ndistinct_out";
			break;
		case PGDEPENDENCIESOID:
			func_name = "pg_dependencies_out";
			break;
		case PGDDLCOMMANDOID:
			func_name = "pg_ddl_command_out";
			break;
		case SMGROID:
			func_name = "smgrout";
			break;
		case POINTOID:
			func_name = "point_out";
			break;
		case LSEGOID:
			func_name = "lseg_out";
			break;
		case PATHOID:
			func_name = "path_out";
			break;
		case BOXOID:
			func_name = "box_out";
			break;
		case LINEOID:
			func_name = "line_out";
			break;
		case FLOAT4OID:
			func_name = "float4out";
			break;
		case FLOAT8OID:
			func_name = "float8out";
			break;
		case CIRCLEOID:
			func_name = "circle_out";
			break;
		case CASHOID:
			func_name = "cash_out";
			break;
		case MACADDROID:
			func_name = "macaddr_out";
			break;
		case INETOID:
			func_name = "inet_out";
			break;
		case CIDROID:
			func_name = "cidr_out";
			break;
		case MACADDR8OID:
			func_name = "macaddr8_out";
			break;
		case ACLITEMOID:
			func_name = "aclitemout";
			break;
		case BPCHAROID:
			func_name = "bpcharout";
			break;
		case VARCHAROID:
			func_name = "varcharout";
			break;
		case DATEOID:
			func_name = "date_out";
			break;
		case TIMEOID:
			func_name = "time_out";
			break;
		case TIMESTAMPOID:
			func_name = "timestamp_out";
			break;
		case TIMESTAMPTZOID:
			func_name = "timestamptz_out";
			break;
		case INTERVALOID:
			func_name = "interval_out";
			break;
		case TIMETZOID:
			func_name = "timetz_out";
			break;
		case BITOID:
			func_name = "bit_out";
			break;
		case VARBITOID:
			func_name = "varbit_out";
			break;
		case NUMERICOID:
			func_name = "numeric_out";
			break;
		case REGPROCEDUREOID:
			func_name = "regprocedureout";
			break;
		case REGOPEROID:
			func_name = "regoperout";
			break;
		case REGOPERATOROID:
			func_name = "regoperatorout";
			break;
		case REGCLASSOID:
			func_name = "regclassout";
			break;
		case REGTYPEOID:
			func_name = "regtypeout";
			break;
		case REGROLEOID:
			func_name = "regroleout";
			break;
		case REGNAMESPACEOID:
			func_name = "regnamespaceout";
			break;
		case UUIDOID:
			func_name = "uuid_out";
			break;
		case LSNOID:
			func_name = "pg_lsn_out";
			break;
		case TSQUERYOID:
			func_name = "tsqueryout";
			break;
		case REGCONFIGOID:
			func_name = "regconfigout";
			break;
		case REGDICTIONARYOID:
			func_name = "regdictionaryout";
			break;
		case JSONBOID:
			func_name = "jsonb_out";
			break;
		case TXID_SNAPSHOTOID:
			func_name = "txid_snapshot_out";
			break;
		case RECORDOID:
			func_name = "record_out";
			break;
		case CSTRINGOID:
			func_name = "cstring_out";
			break;
		case ANYOID:
			func_name = "any_out";
			break;
		case VOIDOID:
			func_name = "void_out";
			break;
		case TRIGGEROID:
			func_name = "trigger_out";
			break;
		case EVTTRIGGEROID:
			func_name = "event_trigger_out";
			break;
		case LANGUAGE_HANDLEROID:
			func_name = "language_handler_out";
			break;
		case INTERNALOID:
			func_name = "internal_out";
			break;
		case OPAQUEOID:
			func_name = "opaque_out";
			break;
		case ANYELEMENTOID:
			func_name = "anyelement_out";
			break;
		case ANYNONARRAYOID:
			func_name = "anynonarray_out";
			break;
		case ANYENUMOID:
			func_name = "anyenum_out";
			break;
		case FDW_HANDLEROID:
			func_name = "fdw_handler_out";
			break;
		case INDEX_AM_HANDLEROID:
			func_name = "index_am_handler_out";
			break;
		case TSM_HANDLEROID:
			func_name = "tsm_handler_out";
			break;
		case ANYRANGEOID:
			func_name = "anyrange_out";
			break;
		case INT2VECTOROID:
			func_name = "int2vectorout";
			break;
		case OIDVECTOROID:
			func_name = "oidvectorout";
			break;
		case TSVECTOROID:
			func_name = "tsvectorout";
			break;
		case GTSVECTOROID:
			func_name = "gtsvectorout";
			break;
		case POLYGONOID:
			func_name = "poly_out";
			break;
		case INT4RANGEOID:
			func_name = "int4out";
			break;
		case NUMRANGEOID:
			func_name = "numeric_out";
			break;
		case TSRANGEOID:
			func_name = "timestamp_out";
			break;
		case TSTZRANGEOID:
			func_name = "timestamptz_out";
			break;
		case DATERANGEOID:
			func_name = "date_out";
			break;
		case INT8RANGEOID:
			func_name = "int8out";
			break;
		case XMLARRAYOID:
			func_name = "xml_out";
			break;
		case LINEARRAYOID:
			func_name = "line_out";
			break;
		case CIRCLEARRAYOID:
			func_name = "circle_out";
			break;
		case MONEYARRAYOID:
			func_name = "cash_out";
			break;
		case BOOLARRAYOID:
			func_name = "boolout";
			break;
		case BYTEAARRAYOID:
			func_name = "byteaout";
			break;
		case CHARARRAYOID:
			func_name = "charout";
			break;
		case NAMEARRAYOID:
			func_name = "nameout";
			break;
		case INT2ARRAYOID:
			func_name = "int2out";
			break;
		case INT2VECTORARRAYOID:
			func_name = "int2vectorout";
			break;
		case INT4ARRAYOID:
			func_name = "int4out";
			break;
		case REGPROCARRAYOID:
			func_name = "regprocout";
			break;
		case TEXTARRAYOID:
			func_name = "textout";
			break;
		case OIDARRAYOID:
			func_name = "oidout";
			break;
		case CIDRARRAYOID:
			func_name = "cidr_out";
			break;
		case TIDARRAYOID:
			func_name = "tidout";
			break;
		case XIDARRAYOID:
			func_name = "xidout";
			break;
		case CIDARRAYOID:
			func_name = "cidout";
			break;
		case OIDVECTORARRAYOID:
			func_name = "oidvectorout";
			break;
		case BPCHARARRAYOID:
			func_name = "bpcharout";
			break;
		case VARCHARARRAYOID:
			func_name = "varcharout";
			break;
		case INT8ARRAYOID:
			func_name = "int8out";
			break;
		case POINTARRAYOID:
			func_name = "point_out";
			break;
		case LSEGARRAYOID:
			func_name = "lseg_out";
			break;
		case PATHARRAYOID:
			func_name = "path_out";
			break;
		case BOXARRAYOID:
			func_name = "box_out";
			break;
		case FLOAT4ARRAYOID:
			func_name = "float4out";
			break;
		case FLOAT8ARRAYOID:
			func_name = "float8out";
			break;
		case ABSTIMEARRAYOID:
			func_name = "abstimeout";
			break;
		case RELTIMEARRAYOID:
			func_name = "reltimeout";
			break;
		case TINTERVALARRAYOID:
			func_name = "tintervalout";
			break;
		case ACLITEMARRAYOID:
			func_name = "aclitemout";
			break;
		case MACADDRARRAYOID:
			func_name = "macaddr_out";
			break;
		case MACADDR8ARRAYOID:
			func_name = "macaddr8_out";
			break;
		case INETARRAYOID:
			func_name = "inet_out";
			break;
		case CSTRINGARRAYOID:
			func_name = "cstring_out";
			break;
		case TIMESTAMPARRAYOID:
			func_name = "timestamp_out";
			break;
		case DATEARRAYOID:
			func_name = "date_out";
			break;
		case TIMEARRAYOID:
			func_name = "time_out";
			break;
		case TIMESTAMPTZARRAYOID:
			func_name = "timestamptz_out";
			break;
		case INTERVALARRAYOID:
			func_name = "interval_out";
			break;
		case NUMERICARRAYOID:
			func_name = "numeric_out";
			break;
		case TIMETZARRAYOID:
			func_name = "timetz_out";
			break;
		case BITARRAYOID:
			func_name = "bit_out";
			break;
		case VARBITARRAYOID:
			func_name = "varbit_out";
			break;
		case REGPROCEDUREARRAYOID:
			func_name = "regprocedureout";
			break;
		case REGOPERARRAYOID:
			func_name = "regoperout";
			break;
		case REGOPERATORARRAYOID:
			func_name = "regoperatorout";
			break;
		case REGCLASSARRAYOID:
			func_name = "regclassout";
			break;
		case REGTYPEARRAYOID:
			func_name = "regtypeout";
			break;
		case REGROLEARRAYOID:
			func_name = "regroleout";
			break;
		case REGNAMESPACEARRAYOID:
			func_name = "regnamespaceout";
			break;
		case UUIDARRAYOID:
			func_name = "uuid_out";
			break;
		case PG_LSNARRAYOID:
			func_name = "pg_lsn_out";
			break;
		case TSVECTORARRAYOID:
			func_name = "tsvectorout";
			break;
		case GTSVECTORARRAYOID:
			func_name = "gtsvectorout";
			break;
		case TSQUERYARRAYOID:
			func_name = "tsqueryout";
			break;
		case REGCONFIGARRAYOID:
			func_name = "regconfigout";
			break;
		case REGDICTIONARYARRAYOID:
			func_name = "regdictionaryout";
			break;
		case JSONARRAYOID:
			func_name = "json_out";
			break;
		case JSONBARRAYOID:
			func_name = "jsonb_out";
			break;
		case TXID_SNAPSHOTARRAYOID:
			func_name = "txid_snapshot_out";
			break;
		case RECORDARRAYOID:
			func_name = "record_out";
			break;
		case ANYARRAYOID:
			func_name = "any_out";
			break;
		case POLYGONARRAYOID:
			func_name = "poly_out";
			break;
		case INT4RANGEARRAYOID:
			func_name = "int4out";
			break;
		case NUMRANGEARRAYOID:
			func_name = "numeric_out";
			break;
		case TSRANGEARRAYOID:
			func_name = "timestamp_out";
			break;
		case TSTZRANGEARRAYOID:
			func_name = "timestamptz_out";
			break;
		case DATERANGEARRAYOID:
			func_name = "date_out";
			break;
		case INT8RANGEARRAYOID:
			func_name = "int8out";
			break;
	}
	return func_name;
}

uint32_t
GetRecordTypeId(uintptr_t datum)
{
	HeapTupleHeader rec = DatumGetHeapTupleHeader(datum);
	return HeapTupleHeaderGetTypeId(rec);
}

uintptr_t
HeapFormTuple(void *attrs, size_t natts, uintptr_t *values, bool *nulls)
{
	TupleDesc tupdesc = CreateTupleDesc(natts, true, attrs);
	PG_RETURN_HEAPTUPLEHEADER(heap_form_tuple(tupdesc, values, nulls)->t_data);
}

void
HeapDeformTuple(uintptr_t datum, void *attrs, size_t natts, uintptr_t *values,
				bool *nulls)
{
	HeapTupleHeader rec = DatumGetHeapTupleHeader(datum);
	HeapTupleData	tuple;
	tuple.t_len = HeapTupleHeaderGetDatumLength(rec);
	ItemPointerSetInvalid(&(tuple.t_self));
	tuple.t_tableOid = InvalidOid;
	tuple.t_data = rec;
	TupleDesc tupdesc = CreateTupleDesc(natts, true, attrs);
	/* Break down the tuple into fields */
	heap_deform_tuple(&tuple, tupdesc, values, nulls);
}
