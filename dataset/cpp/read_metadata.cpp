#include "read_metadata.h"
#include "read_context.h"
#include "plain_reader/plain_read_data.h"
#include <ydb/core/tx/columnshard/hooks/abstract/abstract.h>
#include <ydb/core/tx/columnshard/columnshard__index_scan.h>
#include <ydb/core/tx/columnshard/columnshard__stats_scan.h>
#include <util/string/join.h>

namespace NKikimr::NOlap {

TDataStorageAccessor::TDataStorageAccessor(const std::unique_ptr<NOlap::TInsertTable>& insertTable,
                                const std::unique_ptr<NOlap::IColumnEngine>& index)
    : InsertTable(insertTable)
    , Index(index)
{}

std::shared_ptr<NOlap::TSelectInfo> TDataStorageAccessor::Select(const NOlap::TReadDescription& readDescription, const THashSet<ui32>& /*columnIds*/) const {
    if (readDescription.ReadNothing) {
        return std::make_shared<NOlap::TSelectInfo>();
    }
    return Index->Select(readDescription.PathId,
                            readDescription.GetSnapshot(),
                            readDescription.PKRangesFilter);
}

std::vector<NOlap::TCommittedBlob> TDataStorageAccessor::GetCommitedBlobs(const NOlap::TReadDescription& readDescription, const std::shared_ptr<arrow::Schema>& pkSchema) const {
    return std::move(InsertTable->Read(readDescription.PathId, readDescription.GetSnapshot(), pkSchema));
}

std::unique_ptr<NColumnShard::TScanIteratorBase> TReadMetadata::StartScan(const std::shared_ptr<NOlap::TReadContext>& readContext) const {
    return std::make_unique<NColumnShard::TColumnShardScanIterator>(readContext, this->shared_from_this());
}

bool TReadMetadata::Init(const TReadDescription& readDescription, const TDataStorageAccessor& dataAccessor, std::string& error) {
    auto& indexInfo = ResultIndexSchema->GetIndexInfo();

    std::vector<ui32> resultColumnsIds;
    if (readDescription.ColumnIds.size()) {
        resultColumnsIds = readDescription.ColumnIds;
    } else if (readDescription.ColumnNames.size()) {
        resultColumnsIds = indexInfo.GetColumnIds(readDescription.ColumnNames);
    } else {
        error = "Empty column list requested";
        return false;
    }
    ResultColumnsIds.swap(resultColumnsIds);

    if (!GetResultSchema()) {
        error = "Could not get ResultSchema.";
        return false;
    }

    SetPKRangesFilter(readDescription.PKRangesFilter);

    /// @note We could have column name changes between schema versions:
    /// Add '1:foo', Drop '1:foo', Add '2:foo'. Drop should hide '1:foo' from reads.
    /// It's expected that we have only one version on 'foo' in blob and could split them by schema {planStep:txId}.
    /// So '1:foo' would be omitted in blob records for the column in new snapshots. And '2:foo' - in old ones.
    /// It's not possible for blobs with several columns. There should be a special logic for them.
    {
        Y_ABORT_UNLESS(!ResultColumnsIds.empty(), "Empty column list");
        THashSet<TString> requiredColumns = indexInfo.GetRequiredColumns();

        // Snapshot columns
        requiredColumns.insert(NOlap::TIndexInfo::SPEC_COL_PLAN_STEP);
        requiredColumns.insert(NOlap::TIndexInfo::SPEC_COL_TX_ID);

        for (auto&& i : readDescription.PKRangesFilter.GetColumnNames()) {
            requiredColumns.emplace(i);
        }

        for (auto& col : ResultColumnsIds) {
            requiredColumns.erase(indexInfo.GetColumnName(col));
        }

        std::vector<ui32> auxiliaryColumns;
        auxiliaryColumns.reserve(requiredColumns.size());
        for (auto& reqCol : requiredColumns) {
            auxiliaryColumns.push_back(indexInfo.GetColumnId(reqCol));
        }
        AllColumns.reserve(AllColumns.size() + ResultColumnsIds.size() + auxiliaryColumns.size());
        AllColumns.insert(AllColumns.end(), ResultColumnsIds.begin(), ResultColumnsIds.end());
        AllColumns.insert(AllColumns.end(), auxiliaryColumns.begin(), auxiliaryColumns.end());
    }

    CommittedBlobs = dataAccessor.GetCommitedBlobs(readDescription, ResultIndexSchema->GetIndexInfo().GetReplaceKey());

    THashSet<ui32> columnIds;
    for (auto& columnId : AllColumns) {
        columnIds.insert(columnId);
    }

    for (auto& [id, name] : GetProgram().GetSourceColumns()) {
        columnIds.insert(id);
    }

    SelectInfo = dataAccessor.Select(readDescription, columnIds);
    return true;
}

std::set<ui32> TReadMetadata::GetEarlyFilterColumnIds() const {
    auto& indexInfo = ResultIndexSchema->GetIndexInfo();
    std::set<ui32> result = GetPKRangesFilter().GetColumnIds(indexInfo);
    if (LessPredicate) {
        for (auto&& i : LessPredicate->ColumnNames()) {
            result.emplace(indexInfo.GetColumnId(i));
            AFL_DEBUG(NKikimrServices::TX_COLUMNSHARD_SCAN)("early_filter_column", i);
        }
    }
    if (GreaterPredicate) {
        for (auto&& i : GreaterPredicate->ColumnNames()) {
            result.emplace(indexInfo.GetColumnId(i));
            AFL_DEBUG(NKikimrServices::TX_COLUMNSHARD_SCAN)("early_filter_column", i);
        }
    }
    for (auto&& i : GetProgram().GetEarlyFilterColumns()) {
        auto id = indexInfo.GetColumnIdOptional(i);
        if (id) {
            result.emplace(*id);
            AFL_DEBUG(NKikimrServices::TX_COLUMNSHARD_SCAN)("early_filter_column", i);
        }
    }
    if (Snapshot.GetPlanStep()) {
        auto snapSchema = TIndexInfo::ArrowSchemaSnapshot();
        for (auto&& i : snapSchema->fields()) {
            result.emplace(indexInfo.GetColumnId(i->name()));
            AFL_DEBUG(NKikimrServices::TX_COLUMNSHARD_SCAN)("early_filter_column", i->name());
        }
    }
    return result;
}

std::set<ui32> TReadMetadata::GetPKColumnIds() const {
    std::set<ui32> result;
    auto& indexInfo = ResultIndexSchema->GetIndexInfo();
    for (auto&& i : indexInfo.GetPrimaryKey()) {
        Y_ABORT_UNLESS(result.emplace(indexInfo.GetColumnId(i.first)).second);
    }
    return result;
}

std::vector<std::pair<TString, NScheme::TTypeInfo>> TReadStatsMetadata::GetResultYqlSchema() const {
    return NOlap::GetColumns(NColumnShard::PrimaryIndexStatsSchema, ResultColumnIds);
}

std::vector<std::pair<TString, NScheme::TTypeInfo>> TReadStatsMetadata::GetKeyYqlSchema() const {
    return NOlap::GetColumns(NColumnShard::PrimaryIndexStatsSchema, NColumnShard::PrimaryIndexStatsSchema.KeyColumns);
}

std::unique_ptr<NColumnShard::TScanIteratorBase> TReadStatsMetadata::StartScan(const std::shared_ptr<NOlap::TReadContext>& /*readContext*/) const {
    return std::make_unique<NColumnShard::TStatsIterator>(this->shared_from_this());
}

void TReadStats::PrintToLog() {
    AFL_DEBUG(NKikimrServices::TX_COLUMNSHARD_SCAN)
        ("event", "statistic")
        ("begin", BeginTimestamp)
        ("selected", SelectedIndex)
        ("index_granules", IndexGranules)
        ("index_portions", IndexPortions)
        ("index_batches", IndexBatches)
        ("committed_batches", CommittedBatches)
        ("schema_columns", SchemaColumns)
        ("filter_columns", FilterColumns)
        ("additional_columns", AdditionalColumns)
        ("portions_bytes", PortionsBytes)
        ("data_filter_bytes", DataFilterBytes)
        ("data_additional_bytes", DataAdditionalBytes)
        ("delta_bytes", PortionsBytes - DataFilterBytes - DataAdditionalBytes)
        ("selected_rows", SelectedRows)
        ;
}

std::shared_ptr<NKikimr::NOlap::IDataReader> TReadMetadata::BuildReader(const std::shared_ptr<NOlap::TReadContext>& context) const {
    return std::make_shared<NPlainReader::TPlainReadData>(context);
//    auto result = std::make_shared<TIndexedReadData>(self, context);
//    result->InitRead();
//    return result;
}

NIndexedReader::TSortableBatchPosition TReadMetadata::BuildSortedPosition(const NArrow::TReplaceKey& key) const {
    return NIndexedReader::TSortableBatchPosition(key.ToBatch(GetReplaceKey()), 0,
        GetReplaceKey()->field_names(), {}, IsDescSorted());
}

}
