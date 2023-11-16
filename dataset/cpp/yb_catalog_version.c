/*-------------------------------------------------------------------------
 *
 * yb_catalog_version.c
 *	  utility functions related to the ysql catalog version table.
 *
 * Portions Copyright (c) YugaByte, Inc.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "miscadmin.h"

#include <inttypes.h>

#include "access/htup_details.h"
#include "access/sysattr.h"
#include "access/yb_scan.h"
#include "catalog/catalog.h"
#include "catalog/namespace.h"
#include "catalog/pg_authid_d.h"
#include "catalog/pg_database.h"
#include "catalog/pg_namespace_d.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_type.h"
#include "catalog/pg_yb_catalog_version.h"
#include "catalog/schemapg.h"
#include "catalog/yb_catalog_version.h"
#include "executor/ybcExpr.h"
#include "executor/ybcModifyTable.h"
#include "nodes/makefuncs.h"
#include "utils/catcache.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

#include "yb/yql/pggate/ybc_pggate.h"
#include "yb/yql/pggate/ybc_pg_typedefs.h"
#include "pg_yb_utils.h"

YbCatalogVersionType yb_catalog_version_type = CATALOG_VERSION_UNSET;

static FormData_pg_attribute Desc_pg_yb_catalog_version[Natts_pg_yb_catalog_version] = {
	Schema_pg_yb_catalog_version
};

static bool YbGetMasterCatalogVersionFromTable(Oid db_oid, uint64_t *version);
static Datum YbGetMasterCatalogVersionTableEntryYbctid(
	Relation catalog_version_rel, Oid db_oid);

/* Retrieve Catalog Version */

uint64_t YbGetMasterCatalogVersion()
{
	uint64_t version = YB_CATCACHE_VERSION_UNINITIALIZED;
	switch (YbGetCatalogVersionType())
	{
		case CATALOG_VERSION_CATALOG_TABLE:
			if (YbGetMasterCatalogVersionFromTable(
			    	YbMasterCatalogVersionTableDBOid(), &version))
				return version;
			/*
			 * In spite of the fact the pg_yb_catalog_version table exists it has no actual
			 * version (this could happen during YSQL upgrade),
			 * fallback to an old protobuf mechanism until the next cache refresh.
			 */
			yb_catalog_version_type = CATALOG_VERSION_PROTOBUF_ENTRY;
			switch_fallthrough();
		case CATALOG_VERSION_PROTOBUF_ENTRY:
			/* deprecated (kept for compatibility with old clusters). */
			HandleYBStatus(YBCPgGetCatalogMasterVersion(&version));
			return version;

		case CATALOG_VERSION_UNSET: /* should not happen. */
			break;
	}
	ereport(FATAL,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("catalog version type was not set, cannot load system catalog.")));
	return version;
}

/* Modify Catalog Version */

static void
YbCallSQLIncrementCatalogVersions(bool is_breaking_change)
{
	List* names =
		list_make2(makeString("pg_catalog"),
				   makeString("yb_increment_all_db_catalog_versions"));
	FuncCandidateList clist = FuncnameGetCandidates(
		names,
		-1 /* nargs */,
		NIL /* argnames */,
		false /* expand_variadic */,
		false /* expand_defaults */,
		false /* missing_ok */);
	/* We expect exactly one candidate. */
	Assert(clist && clist->next == NULL);
	Oid functionId = clist->oid;
	FmgrInfo    flinfo;
	FunctionCallInfoData fcinfo;
	fmgr_info(functionId, &flinfo);
	InitFunctionCallInfoData(fcinfo, &flinfo, 1, InvalidOid, NULL, NULL);
	fcinfo.arg[0] = BoolGetDatum(is_breaking_change);
	fcinfo.argnull[0] = false;

	// Save old values and set new values to enable the call.
	bool saved = yb_non_ddl_txn_for_sys_tables_allowed;
	yb_non_ddl_txn_for_sys_tables_allowed = true;
	Oid save_userid;
	int save_sec_context;
	GetUserIdAndSecContext(&save_userid, &save_sec_context);
	SetUserIdAndSecContext(BOOTSTRAP_SUPERUSERID,
						   SECURITY_RESTRICTED_OPERATION);
	PG_TRY();
	{
		FunctionCallInvoke(&fcinfo);
		/* Restore old values. */
		yb_non_ddl_txn_for_sys_tables_allowed = saved;
		SetUserIdAndSecContext(save_userid, save_sec_context);
	}
	PG_CATCH();
	{
		/* Restore old values. */
		yb_non_ddl_txn_for_sys_tables_allowed = saved;
		SetUserIdAndSecContext(save_userid, save_sec_context);
		PG_RE_THROW();
	}
	PG_END_TRY();
}

static void
YbIncrementMasterDBCatalogVersionTableEntryImpl(
	Oid db_oid, bool is_breaking_change, bool is_global_ddl)
{
	Assert(YbGetCatalogVersionType() == CATALOG_VERSION_CATALOG_TABLE);

	if (is_global_ddl)
	{
		Assert(YBIsDBCatalogVersionMode());
		/* Call yb_increment_all_db_catalog_versions(is_breaking_change). */
		YbCallSQLIncrementCatalogVersions(is_breaking_change);
		return;
	}

	YBCPgStatement update_stmt    = NULL;
	YBCPgTypeAttrs type_attrs = { 0 };
	YBCPgExpr yb_expr;

	/* The table pg_yb_catalog_version is in template1. */
	HandleYBStatus(YBCPgNewUpdate(TemplateDbOid,
								  YBCatalogVersionRelationId,
								  false /* is_region_local */,
								  &update_stmt,
									YB_TRANSACTIONAL));

	Relation rel = RelationIdGetRelation(YBCatalogVersionRelationId);
	Datum ybctid = YbGetMasterCatalogVersionTableEntryYbctid(rel, db_oid);

	/* Bind ybctid to identify the current row. */
	YBCPgExpr ybctid_expr = YBCNewConstant(update_stmt, BYTEAOID, InvalidOid,
										   ybctid, false /* is_null */);
	HandleYBStatus(YBCPgDmlBindColumn(update_stmt, YBTupleIdAttributeNumber,
									  ybctid_expr));

	/* Set expression c = c + 1 for current version attribute. */
	AttrNumber attnum = Anum_pg_yb_catalog_version_current_version;
	Var *arg1 = makeVar(1,
						attnum,
						INT8OID,
						0,
						InvalidOid,
						0);

	Const *arg2 = makeConst(INT8OID,
							0,
							InvalidOid,
							sizeof(int64),
							(Datum) 1,
							false,
							true);

	List *args = list_make2(arg1, arg2);

	FuncExpr *expr = makeFuncExpr(F_INT8PL,
								  INT8OID,
								  args,
								  InvalidOid,
								  InvalidOid,
								  COERCE_EXPLICIT_CALL);

	/* INT8 OID. */
	YBCPgExpr ybc_expr = YBCNewEvalExprCall(update_stmt, (Expr *) expr);

	HandleYBStatus(YBCPgDmlAssignColumn(update_stmt, attnum, ybc_expr));
	yb_expr = YBCNewColumnRef(update_stmt,
							  attnum,
							  INT8OID,
							  InvalidOid,
							  &type_attrs);
	HandleYBStatus(YbPgDmlAppendColumnRef(update_stmt, yb_expr, true));

	/* If breaking change set the latest breaking version to the same expression. */
	if (is_breaking_change)
	{
		ybc_expr = YBCNewEvalExprCall(update_stmt, (Expr *) expr);
		HandleYBStatus(YBCPgDmlAssignColumn(update_stmt, attnum + 1, ybc_expr));
	}

	int rows_affected_count = 0;
	if (*YBCGetGFlags()->log_ysql_catalog_versions)
	{
		char tmpbuf[30] = "";
		if (YBIsDBCatalogVersionMode())
			snprintf(tmpbuf, sizeof(tmpbuf), " for database %u", db_oid);
		ereport(LOG,
				(errmsg("%s: incrementing master catalog version (%sbreaking)%s",
						__func__, is_breaking_change ? "" : "non", tmpbuf)));
	}
	HandleYBStatus(YBCPgDmlExecWriteOp(update_stmt, &rows_affected_count));
	/*
	 * Under normal situation rows_affected_count should be exactly 1. However
	 * when a connection is established in per-database catalog version mode,
	 * if the table pg_yb_catalog_version is updated to only have one row
	 * for database template1 which is part of the process of converting to
	 * global catalog version mode, this connection remains in per-database
	 * catalog version mode. In this case rows_affected_count will be 0 unless
	 * MyDatabaseId is template1 because in pg_yb_catalog_version the row for
	 * MyDatabaseId no longer exists.
	 */
	if (rows_affected_count == 0)
		Assert(YBIsDBCatalogVersionMode());
	else
		Assert(rows_affected_count == 1);

	/* Cleanup. */
	update_stmt = NULL;
	RelationClose(rel);
}

bool YbIncrementMasterCatalogVersionTableEntry(bool is_breaking_change,
											   bool is_global_ddl)
{
	if (YbGetCatalogVersionType() != CATALOG_VERSION_CATALOG_TABLE)
		return false;
	/*
	 * TemplateDbOid row is for global catalog version when not in per-db mode.
	 */
	YbIncrementMasterDBCatalogVersionTableEntryImpl(
		YBIsDBCatalogVersionMode() ? MyDatabaseId : TemplateDbOid,
		is_breaking_change, is_global_ddl);
	return true;
}

bool YbMarkStatementIfCatalogVersionIncrement(YBCPgStatement ybc_stmt,
											  Relation rel) {
	if (YbGetCatalogVersionType() != CATALOG_VERSION_PROTOBUF_ENTRY)
	{
		/*
		 * Nothing to do -- only need to maintain this for the (old)
		 * protobuf-based way of storing the version.
		 */
		return false;
	}

	bool is_syscatalog_change = YbIsSystemCatalogChange(rel);
	bool modifies_row = false;
	HandleYBStatus(YBCPgDmlModifiesRow(ybc_stmt, &modifies_row));

	/*
	 * If this write may invalidate catalog cache tuples (i.e. UPDATE or DELETE),
	 * or this write may insert into a cached list, we must increment the
	 * cache version so other sessions can invalidate their caches.
	 * NOTE: If this relation caches lists, an INSERT could effectively be
	 * UPDATINGing the list object.
	 */
	bool is_syscatalog_version_change = is_syscatalog_change
			&& (modifies_row || RelationHasCachedLists(rel));

	/* Let the master know if this should increment the catalog version. */
	if (is_syscatalog_version_change)
	{
		HandleYBStatus(YBCPgSetIsSysCatalogVersionChange(ybc_stmt));
	}

	return is_syscatalog_version_change;
}

void YbCreateMasterDBCatalogVersionTableEntry(Oid db_oid)
{
	Assert(YbGetCatalogVersionType() == CATALOG_VERSION_CATALOG_TABLE);
	Assert(db_oid != MyDatabaseId);

	/*
	 * The table pg_yb_catalog_version is a shared relation in template1 and
	 * db_oid is the primary key. There is no separate docdb index table for
	 * primary key and therefore only one insert statement is needed to insert
	 * the row for db_oid.
	 */
	YBCPgStatement insert_stmt = NULL;
	HandleYBStatus(YBCPgNewInsert(TemplateDbOid,
								  YBCatalogVersionRelationId,
								  false /* is_region_local */,
								  &insert_stmt,
								  YB_SINGLE_SHARD_TRANSACTION));

	Relation rel = RelationIdGetRelation(YBCatalogVersionRelationId);
	Datum ybctid = YbGetMasterCatalogVersionTableEntryYbctid(rel, db_oid);

	YBCPgExpr ybctid_expr = YBCNewConstant(insert_stmt, BYTEAOID, InvalidOid,
										   ybctid, false /* is_null */);
	HandleYBStatus(YBCPgDmlBindColumn(insert_stmt, YBTupleIdAttributeNumber,
									  ybctid_expr));

	AttrNumber attnum = Anum_pg_yb_catalog_version_current_version;
	Datum		initial_version = 1;
	YBCPgExpr initial_version_expr = YBCNewConstant(insert_stmt, INT8OID,
													InvalidOid,
													initial_version,
													false /* is_null */);
	HandleYBStatus(YBCPgDmlBindColumn(insert_stmt, attnum,
									  initial_version_expr));
	HandleYBStatus(YBCPgDmlBindColumn(insert_stmt, attnum + 1,
									  initial_version_expr));

	int rows_affected_count = 0;
	if (*YBCGetGFlags()->log_ysql_catalog_versions)
		ereport(LOG,
				(errmsg("%s: creating master catalog version for database %u",
						__func__, db_oid)));
	HandleYBStatus(YBCPgDmlExecWriteOp(insert_stmt, &rows_affected_count));
	/* Insert a new row does not affect any existing rows. */
	Assert(rows_affected_count == 0);

	/* Cleanup. */
	RelationClose(rel);
}

void YbDeleteMasterDBCatalogVersionTableEntry(Oid db_oid)
{
	Assert(YbGetCatalogVersionType() == CATALOG_VERSION_CATALOG_TABLE);
	Assert(db_oid != MyDatabaseId);

	/*
	 * The table pg_yb_catalog_version is a shared relation in template1 and
	 * db_oid is the primary key. There is no separate docdb index table for
	 * primary key and therefore only one delete statement is needed to delete
	 * the row for db_oid.
	 */
	YBCPgStatement delete_stmt = NULL;
	HandleYBStatus(YBCPgNewDelete(TemplateDbOid,
								  YBCatalogVersionRelationId,
								  false /* is_region_local */,
								  &delete_stmt,
									YB_SINGLE_SHARD_TRANSACTION));

	Relation rel = RelationIdGetRelation(YBCatalogVersionRelationId);
	Datum ybctid = YbGetMasterCatalogVersionTableEntryYbctid(rel, db_oid);

	YBCPgExpr ybctid_expr = YBCNewConstant(delete_stmt, BYTEAOID, InvalidOid,
										   ybctid, false /* is_null */);
	HandleYBStatus(YBCPgDmlBindColumn(delete_stmt, YBTupleIdAttributeNumber,
									  ybctid_expr));

	int rows_affected_count = 0;
	if (*YBCGetGFlags()->log_ysql_catalog_versions)
		ereport(LOG,
				(errmsg("%s: deleting master catalog version for database %u",
						__func__, db_oid)));
	HandleYBStatus(YBCPgDmlExecWriteOp(delete_stmt, &rows_affected_count));
	Assert(rows_affected_count == 1);

	RelationClose(rel);
}

YbCatalogVersionType YbGetCatalogVersionType()
{
	if (IsBootstrapProcessingMode())
	{
		/*
		 * We don't have the catalog version table at the start of initdb,
		 * and there's no point in switching later on.
		 */
		yb_catalog_version_type = CATALOG_VERSION_PROTOBUF_ENTRY;
	}
	else if (yb_catalog_version_type == CATALOG_VERSION_UNSET)
	{
		bool catalog_version_table_exists = false;
		HandleYBStatus(YBCPgTableExists(
			TemplateDbOid, YBCatalogVersionRelationId,
			&catalog_version_table_exists));
		yb_catalog_version_type = catalog_version_table_exists
		    ? CATALOG_VERSION_CATALOG_TABLE
		    : CATALOG_VERSION_PROTOBUF_ENTRY;
	}
	return yb_catalog_version_type;
}


/*
 * Check if operation changes a system table, ignore changes during
 * initialization (bootstrap mode).
 */
bool YbIsSystemCatalogChange(Relation rel)
{
	return IsCatalogRelation(rel) && !IsBootstrapProcessingMode();
}


bool YbGetMasterCatalogVersionFromTable(Oid db_oid, uint64_t *version)
{
	*version = 0; /* unset; */

	int natts = Natts_pg_yb_catalog_version;
	/*
	 * pg_yb_catalog_version is a shared catalog table, so as per DocDB store,
	 * it belongs to the template1 database.
	 */
	int oid_attnum = Anum_pg_yb_catalog_version_db_oid;
	int current_version_attnum = Anum_pg_yb_catalog_version_current_version;
	Form_pg_attribute oid_attrdesc = &Desc_pg_yb_catalog_version[oid_attnum - 1];

	YBCPgStatement ybc_stmt;

	HandleYBStatus(YBCPgNewSelect(TemplateDbOid,
	                              YBCatalogVersionRelationId,
	                              NULL /* prepare_params */,
	                              false /* is_region_local */,
	                              &ybc_stmt));

	Datum oid_datum = Int32GetDatum(db_oid);
	YBCPgExpr pkey_expr = YBCNewConstant(ybc_stmt,
	                                     oid_attrdesc->atttypid,
	                                     oid_attrdesc->attcollation,
	                                     oid_datum,
	                                     false /* is_null */);

	HandleYBStatus(YBCPgDmlBindColumn(ybc_stmt, 1, pkey_expr));

	/* Add scan targets */
	for (AttrNumber attnum = 1; attnum <= natts; attnum++)
	{
		/*
		 * Before copying the following code, see if YbDmlAppendTargetRegular
		 * or similar could be used instead.  Reason this doesn't use
		 * YbDmlAppendTargetRegular is that it doesn't have access to
		 * TupleDesc.  YbDmlAppendTargetRegular could be changed to take
		 * Form_pg_attribute instead, but that would make it inconvenient for
		 * other callers.
		 */
		Form_pg_attribute att = &Desc_pg_yb_catalog_version[attnum - 1];
		YBCPgTypeAttrs type_attrs = { att->atttypmod };
		YBCPgExpr   expr = YBCNewColumnRef(ybc_stmt, attnum, att->atttypid,
										   att->attcollation, &type_attrs);
		HandleYBStatus(YBCPgDmlAppendTarget(ybc_stmt, expr));
	}

	/*
	 * Fetching of the YBTupleIdAttributeNumber attribute is required for
	 * the ability to prefetch data from the pb_yb_catalog_version table via
	 * PgSysTablePrefetcher.
	 */
	YbDmlAppendTargetSystem(YBTupleIdAttributeNumber, ybc_stmt);

	HandleYBStatus(YBCPgExecSelect(ybc_stmt, NULL /* exec_params */));

	bool      has_data = false;

	Datum           *values = (Datum *) palloc0(natts * sizeof(Datum));
	bool            *nulls  = (bool *) palloc(natts * sizeof(bool));
	YBCPgSysColumns syscols;
	bool result = false;

	if (!YBIsDBCatalogVersionMode())
	{
		/* Fetch one row. */
		HandleYBStatus(YBCPgDmlFetch(ybc_stmt,
									 natts,
									 (uint64_t *) values,
									 nulls,
									 &syscols,
									 &has_data));

		if (has_data)
		{
			*version = (uint64_t) DatumGetInt64(values[current_version_attnum - 1]);
			result = true;
		}
	}
	else
	{
		/*
		 * When prefetching is enabled we always load all the rows even though
		 * we bind to the row matching given db_oid. This is a work around to
		 * pick the row that matches db_oid. This work around should be removed
		 * when prefetching is enhanced to support filtering.
		 */
		while (true) {
			/* Fetch one row. */
			HandleYBStatus(YBCPgDmlFetch(ybc_stmt,
										 natts,
										 (uint64_t *) values,
										 nulls,
										 &syscols,
										 &has_data));

			if (!has_data)
				ereport(ERROR,
					(errcode(ERRCODE_DATABASE_DROPPED),
					 errmsg("catalog version for database %u was not found.", db_oid),
					 errhint("Database might have been dropped by another user")));

			uint32_t oid = (uint32_t) DatumGetInt32(values[oid_attnum - 1]);
			if (oid == db_oid)
			{
				*version = (uint64_t) DatumGetInt64(values[current_version_attnum - 1]);
				result = true;
				break;
			}
 		}
	}

	pfree(values);
	pfree(nulls);
	return result;
}

Datum YbGetMasterCatalogVersionTableEntryYbctid(Relation catalog_version_rel,
												Oid db_oid)
{
	/*
	 * Construct HeapTuple (db_oid, null, null) for computing ybctid using
	 * YBCGetYBTupleIdFromTuple which requires a tuple. Note that db_oid
	 * is the primary key so we can use null for other columns for simplicity.
	 */
	Datum		values[3];
	bool		nulls[3];

	values[0] = db_oid;
	nulls[0] = false;
	values[1] = 0;
	nulls[1] = true;
	values[2] = 0;
	nulls[2] = true;

	HeapTuple tuple = heap_form_tuple(RelationGetDescr(catalog_version_rel),
									  values, nulls);
	return YBCGetYBTupleIdFromTuple(catalog_version_rel, tuple,
									RelationGetDescr(catalog_version_rel));
}

Oid YbMasterCatalogVersionTableDBOid()
{
	/*
	 * MyDatabaseId is 0 during connection setup time before
	 * MyDatabaseId is resolved. In per-db mode, we use TemplateDbOid
	 * during this period to find the catalog version in order to load
	 * initial catalog cache (needed for resolving MyDatabaseId, auth
	 * check etc.). Once MyDatabaseId is resolved from then on we'll
	 * use its catalog version.
	 */

	return YBIsDBCatalogVersionMode() && OidIsValid(MyDatabaseId)
		? MyDatabaseId : TemplateDbOid;
}
