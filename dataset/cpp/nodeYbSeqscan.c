/*-------------------------------------------------------------------------
 *
 * nodeYbSeqscan.c
 *	  Support routines for sequential scans of Yugabyte relations.
 *
 * Copyright (c) YugaByte, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.  See the License for the specific language governing permissions and limitations
 * under the License.
 *
 * Portions Copyright (c) 1996-2018, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeYbSeqscan.c
 *
 *-------------------------------------------------------------------------
 */
/*
 * INTERFACE ROUTINES
 *		ExecYbSeqScan				sequentially scans a relation.
 *		ExecYbSeqNext				retrieve next tuple in sequential order.
 *		ExecInitYbSeqScan			creates and initializes a seqscan node.
 *		ExecEndYbSeqScan			releases any storage allocated.
 *		ExecReScanYbSeqScan			rescans the relation
 */
#include "postgres.h"

#include "access/relscan.h"
#include "access/xact.h"
#include "executor/execdebug.h"
#include "executor/nodeYbSeqscan.h"
#include "utils/rel.h"

static TupleTableSlot *YbSeqNext(YbSeqScanState *node);

/* ----------------------------------------------------------------
 *						Scan Support
 * ----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 *		YbSeqNext
 *
 *		This is a workhorse for ExecYbSeqScan
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
YbSeqNext(YbSeqScanState *node)
{
	HeapScanDesc scandesc;
	EState	   *estate;
	TupleTableSlot *slot;
	ExprContext *econtext;
	MemoryContext oldcontext;
	YbScanDesc ybScan;

	/*
	 * get information from the estate and scan state
	 */
	scandesc = node->ss.ss_currentScanDesc;
	estate = node->ss.ps.state;
	econtext = node->ss.ps.ps_ExprContext;
	slot = node->ss.ss_ScanTupleSlot;

	/*
	 * Initialize the scandesc upon the first invocation.
	 * The scan only needs its ybscan field, so eventually we may use it
	 * directly and ignore the ss_currentScanDesc.
	 */
	if (scandesc == NULL)
	{
		if (node->aggrefs)
		{
			/*
			 * For aggregate pushdown, we read just the aggregates from DocDB
			 * and pass that up to the aggregate node (agg pushdown wouldn't be
			 * enabled if we needed to read more than that).  Set up a dummy
			 * scan slot to hold that as many attributes as there are pushed
			 * aggregates.
			 */
			TupleDesc tupdesc =
				CreateTemplateTupleDesc(list_length(node->aggrefs),
										false /* hasoid */);
			ExecInitScanTupleSlot(estate, &node->ss, tupdesc);
			/* Refresh the local pointer. */
			slot = node->ss.ss_ScanTupleSlot;
		}

		YbSeqScan *plan = (YbSeqScan *) node->ss.ps.plan;
		PushdownExprs *yb_pushdown =
			YbInstantiatePushdownParams(&plan->yb_pushdown, estate);
		scandesc = ybc_remote_beginscan(node->ss.ss_currentRelation,
										estate->es_snapshot,
										(Scan *) plan,
										yb_pushdown,
										node->aggrefs,
										&estate->yb_exec_params);
		node->ss.ss_currentScanDesc = scandesc;
	}

	/*
	 * Set up any locking that happens at the time of the scan.
	 */
	if (IsYugaByteEnabled() && IsolationIsSerializable())
	{
		/*
		 * In case of SERIALIZABLE isolation level we have to take prefix range
		 * locks to disallow INSERTion of new rows that satisfy the query
		 * predicate. So, we set the rowmark on all read requests sent to
		 * tserver instead of locking each tuple one by one in LockRows node.
		 */
		ListCell   *l;
		foreach(l, estate->es_rowMarks)
		{
			ExecRowMark *erm = (ExecRowMark *) lfirst(l);
			/* Do not propagate non-row-locking row marks. */
			if (erm->markType != ROW_MARK_REFERENCE &&
				erm->markType != ROW_MARK_COPY)
			{
				scandesc->ybscan->exec_params->rowmark = erm->markType;
				scandesc->ybscan->exec_params->pg_wait_policy = erm->waitPolicy;
				YBSetRowLockPolicy(
					&scandesc->ybscan->exec_params->docdb_wait_policy,
					erm->waitPolicy);
			}
			break;
		}
	}

	/*
	 * Since the scandesc is destroyed upon node rescan, the statement is
	 * executed if and only if a new scandesc is created. In other words,
	 * YBCPgExecSelect can be unconditionally executed in the "if" block above
	 * and ybScan->is_exec_done can be ignored.
	 * However, it is kinda convenient to safely assign ybScan here and use to
	 * execute and fetch the statement, so we make use of the flag.
	 */
	ybScan = scandesc->ybscan;
	if (!ybScan->is_exec_done)
	{
		HandleYBStatus(YBCPgExecSelect(ybScan->handle, ybScan->exec_params));
		ybScan->is_exec_done = true;
	}

	/* capture all fetch allocations in the short-lived context */
	oldcontext = MemoryContextSwitchTo(econtext->ecxt_per_tuple_memory);
	slot = ybFetchNext(ybScan->handle,
					   slot,
					   RelationGetRelid(node->ss.ss_currentRelation));
	MemoryContextSwitchTo(oldcontext);

	return slot;
}

/*
 * YbSeqRecheck -- access method routine to recheck a tuple in EvalPlanQual
 */
static bool
YbSeqRecheck(YbSeqScanState *node, TupleTableSlot *slot)
{
	/*
	 * Note that unlike IndexScan, SeqScan never use keys in heap_beginscan
	 * (and this is very bad) - so, here we do not check are keys ok or not.
	 */
	return true;
}

/* ----------------------------------------------------------------
 *		ExecYbSeqScan(node)
 *
 *		Scans the relation sequentially and returns the next qualifying
 *		tuple.
 *		We call the ExecScan() routine and pass it the appropriate
 *		access method functions.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
ExecYbSeqScan(PlanState *pstate)
{
	YbSeqScanState *node = castNode(YbSeqScanState, pstate);

	return ExecScan(&node->ss,
					(ExecScanAccessMtd) YbSeqNext,
					(ExecScanRecheckMtd) YbSeqRecheck);
}


/* ----------------------------------------------------------------
 *		ExecInitYbSeqScan
 * ----------------------------------------------------------------
 */
YbSeqScanState *
ExecInitYbSeqScan(YbSeqScan *node, EState *estate, int eflags)
{
	YbSeqScanState *scanstate;

	/*
	 * Once upon a time it was possible to have an outerPlan of a SeqScan, but
	 * not any more.
	 */
	Assert(outerPlan(node) == NULL);
	Assert(innerPlan(node) == NULL);

	/*
	 * create state structure
	 */
	scanstate = makeNode(YbSeqScanState);
	scanstate->ss.ps.plan = (Plan *) node;
	scanstate->ss.ps.state = estate;
	scanstate->ss.ps.ExecProcNode = ExecYbSeqScan;

	/*
	 * Miscellaneous initialization
	 *
	 * create expression context for node
	 */
	ExecAssignExprContext(estate, &scanstate->ss.ps);

	/*
	 * Initialize scan relation.
	 *
	 * Get the relation object id from the relid'th entry in the range table,
	 * open that relation and acquire appropriate lock on it.
	 */
	scanstate->ss.ss_currentRelation =
		ExecOpenScanRelation(estate,
							 node->scan.scanrelid,
							 eflags);

	/* and create slot with the appropriate rowtype */
	ExecInitScanTupleSlot(estate, &scanstate->ss,
						  RelationGetDescr(scanstate->ss.ss_currentRelation));

	/*
	 * Initialize result type and projection.
	 */
	ExecInitResultTypeTL(&scanstate->ss.ps);
	ExecAssignScanProjectionInfo(&scanstate->ss);

	/*
	 * initialize child expressions
	 */
	scanstate->ss.ps.qual =
		ExecInitQual(node->scan.plan.qual, (PlanState *) scanstate);

	return scanstate;
}

/* ----------------------------------------------------------------
 *		ExecEndYbSeqScan
 *
 *		frees any storage allocated through C routines.
 * ----------------------------------------------------------------
 */
void
ExecEndYbSeqScan(YbSeqScanState *node)
{
	Relation	relation;
	HeapScanDesc scanDesc;

	/*
	 * get information from node
	 */
	relation = node->ss.ss_currentRelation;
	scanDesc = node->ss.ss_currentScanDesc;

	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&node->ss.ps);

	/*
	 * clean out the tuple table
	 */
	if (node->ss.ps.ps_ResultTupleSlot)
		ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);
	ExecClearTuple(node->ss.ss_ScanTupleSlot);

	/*
	 * close heap scan
	 */
	if (scanDesc != NULL)
		ybc_heap_endscan(scanDesc);

	/*
	 * close the heap relation.
	 */
	ExecCloseScanRelation(relation);
}

/* ----------------------------------------------------------------
 *						Join Support
 * ----------------------------------------------------------------
 */

/* ----------------------------------------------------------------
 *		ExecReScanYbSeqScan
 *
 *		Rescans the relation.
 * ----------------------------------------------------------------
 */
void
ExecReScanYbSeqScan(YbSeqScanState *node)
{
	HeapScanDesc scanDesc;

	/*
	 * End previous YB scan to reinit it upon the next fetch.
	 */
	scanDesc = node->ss.ss_currentScanDesc;
	if (scanDesc != NULL)
	{
		ybc_heap_endscan(scanDesc);
		node->ss.ss_currentScanDesc = NULL;
	}

	ExecScanReScan((ScanState *) node);
}
