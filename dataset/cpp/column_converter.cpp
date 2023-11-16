#include "column_converter.h"

#include "boolean_column_converter.h"
#include "floating_point_column_converter.h"
#include "integer_column_converter.h"
#include "null_column_converter.h"
#include "string_column_converter.h"

#include <yt/yt/client/table_client/row_base.h>
#include <yt/yt/client/table_client/schema.h>
#include <yt/yt/client/table_client/unversioned_row.h>

namespace NYT::NColumnConverters {

using namespace NTableClient;

////////////////////////////////////////////////////////////////////////////////

IColumnConverterPtr CreateColumnConvert(
    const NTableClient::TColumnSchema& columnSchema,
    int columnIndex)
{
    switch (columnSchema.GetWireType()) {
        case EValueType::Int64:
            return CreateInt64ColumnConverter(columnIndex, columnSchema);

        case EValueType::Uint64:
            return CreateUint64ColumnConverter(columnIndex, columnSchema);

        case EValueType::Double:
            switch (columnSchema.CastToV1Type()) {
                case NTableClient::ESimpleLogicalValueType::Float:
                    return CreateFloatingPoint32ColumnConverter(columnIndex, columnSchema);
                default:
                    return CreateFloatingPoint64ColumnConverter(columnIndex, columnSchema);
            }

        case EValueType::String:
            return CreateStringConverter(columnIndex, columnSchema);

        case EValueType::Boolean:
            return CreateBooleanColumnConverter(columnIndex, columnSchema);

        case EValueType::Any:
            return CreateAnyConverter(columnIndex, columnSchema);

        case EValueType::Composite:
            return CreateCompositeConverter(columnIndex, columnSchema);

        case EValueType::Null:
            return CreateNullConverter(columnIndex);

        case EValueType::Min:
        case EValueType::TheBottom:
        case EValueType::Max:
            break;
    }
    ThrowUnexpectedValueType(columnSchema.GetWireType());
}

////////////////////////////////////////////////////////////////////////////////


TConvertedColumnRange ConvertRowsToColumns(
    TRange<TUnversionedRow> rows,
    const std::vector<TColumnSchema>& columnSchema)
{
    TConvertedColumnRange convertedColumnsRange;
    std::vector<TUnversionedRowValues> rowsValues;
    rowsValues.reserve(rows.size());

    for (auto row : rows) {
        TUnversionedRowValues rowValues(columnSchema.size(), nullptr);
        for (const auto* item = row.Begin(); item != row.End(); ++item) {
            rowValues[item->Id] = item;
        }
        rowsValues.push_back(std::move(rowValues));
    }

    for (int columnId = 0; columnId < std::ssize(columnSchema); columnId++) {
        auto converter = CreateColumnConvert(columnSchema[columnId], columnId);
        auto columns = converter->Convert(rowsValues);
        convertedColumnsRange.push_back(columns);
    }
    return convertedColumnsRange;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NColumnConverters
