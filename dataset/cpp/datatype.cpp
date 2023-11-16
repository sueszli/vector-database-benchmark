// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "datatype.h"
#include "numericdatatype.h"
#include "primitivedatatype.h"
#include "documenttype.h"
#include "weightedsetdatatype.h"
#include <vespa/document/fieldvalue/fieldvalues.h>
#include <vespa/document/base/exceptions.h>
#include <vespa/vespalib/text/lowercase.h>

namespace document {

namespace {
NumericDataType BYTE_OBJ(DataType::T_BYTE);
NumericDataType SHORT_OBJ(DataType::T_SHORT);
NumericDataType INT_OBJ(DataType::T_INT);
NumericDataType LONG_OBJ(DataType::T_LONG);
NumericDataType FLOAT_OBJ(DataType::T_FLOAT);
NumericDataType DOUBLE_OBJ(DataType::T_DOUBLE);
NumericDataType BOOL_OBJ(DataType::T_BOOL);
PrimitiveDataType STRING_OBJ(DataType::T_STRING);
PrimitiveDataType RAW_OBJ(DataType::T_RAW);
DocumentType DOCUMENT_OBJ("document");
WeightedSetDataType TAG_OBJ(*DataType::STRING, true, true);
PrimitiveDataType URI_OBJ(DataType::T_URI);
PrimitiveDataType PREDICATE_OBJ(DataType::T_PREDICATE);
PrimitiveDataType TENSOR_OBJ(DataType::T_TENSOR);

}  // namespace

const DataType *const DataType::BYTE(&BYTE_OBJ);
const DataType *const DataType::SHORT(&SHORT_OBJ);
const DataType *const DataType::INT(&INT_OBJ);
const DataType *const DataType::LONG(&LONG_OBJ);
const DataType *const DataType::FLOAT(&FLOAT_OBJ);
const DataType *const DataType::DOUBLE(&DOUBLE_OBJ);
const DataType *const DataType::BOOL(&BOOL_OBJ);
const DataType *const DataType::STRING(&STRING_OBJ);
const DataType *const DataType::RAW(&RAW_OBJ);
const DocumentType *const DataType::DOCUMENT(&DOCUMENT_OBJ);
const DataType *const DataType::TAG(&TAG_OBJ);
const DataType *const DataType::URI(&URI_OBJ);
const DataType *const DataType::PREDICATE(&PREDICATE_OBJ);
const DataType *const DataType::TENSOR(&TENSOR_OBJ);


namespace {

class DataType2FieldValueId
{
public:
    DataType2FieldValueId();
    FieldValue::Type getFieldValueId(unsigned int id) const {
        return id < sizeof(_type2FieldValueId)/sizeof(_type2FieldValueId[0])
               ? _type2FieldValueId[id]
               : FieldValue::Type::NONE;
    }
private:
    FieldValue::Type _type2FieldValueId[DataType::MAX];
};

DataType2FieldValueId::DataType2FieldValueId()
{
    for (size_t i(0); i < sizeof(_type2FieldValueId)/sizeof(_type2FieldValueId[0]); i++) {
        _type2FieldValueId[i] = FieldValue::Type::NONE;
    }
    _type2FieldValueId[DataType::T_BYTE]  = FieldValue::Type::BYTE;
    _type2FieldValueId[DataType::T_SHORT] = FieldValue::Type::SHORT;
    _type2FieldValueId[DataType::T_INT] = FieldValue::Type::INT;
    _type2FieldValueId[DataType::T_LONG] = FieldValue::Type::LONG;
    _type2FieldValueId[DataType::T_FLOAT] = FieldValue::Type::FLOAT;
    _type2FieldValueId[DataType::T_DOUBLE] = FieldValue::Type::DOUBLE;
    _type2FieldValueId[DataType::T_BOOL] = FieldValue::Type::BOOL;
    _type2FieldValueId[DataType::T_STRING] = FieldValue::Type::STRING;
    _type2FieldValueId[DataType::T_RAW] = FieldValue::Type::RAW;
    _type2FieldValueId[DataType::T_URI] = FieldValue::Type::STRING;
    _type2FieldValueId[DataType::T_PREDICATE] = FieldValue::Type::PREDICATE;
    _type2FieldValueId[DataType::T_TENSOR] = FieldValue::Type::TENSOR;
}

DataType2FieldValueId _G_type2FieldValueId;

}

bool DataType::isValueType(const FieldValue & fv) const
{
    if ((_dataTypeId >= 0) && _dataTypeId < MAX) {
        if (fv.isA(_G_type2FieldValueId.getFieldValueId(_dataTypeId))) {
            return true;
        }
    }
    return _dataTypeId == fv.getDataType()->getId();
}

std::vector<const DataType *>
DataType::getDefaultDataTypes()
{
    std::vector<const DataType *> types;
    types.push_back(BYTE);
    types.push_back(SHORT);
    types.push_back(INT);
    types.push_back(LONG);
    types.push_back(FLOAT);
    types.push_back(DOUBLE);
    types.push_back(BOOL);
    types.push_back(STRING);
    types.push_back(RAW);
    types.push_back(DOCUMENT);
    types.push_back(TAG);
    types.push_back(URI);
    types.push_back(PREDICATE);
    return types;
}

namespace {
// This should be equal to java implementation if name only has 7-bit
// ASCII characters. Probably screwed up otherwise, but generated ids
// should only be used in testing anyways. In production this will be
// set from the document manager config.
uint32_t
crappyJavaStringHash(vespalib::stringref value) {
    uint32_t h = 0;
    for (uint32_t i = 0; i < value.size(); ++i) {
        h = 31 * h + value[i];
    }
    return h;
}

int32_t createId(vespalib::stringref name)
{
    if (name == "Tag") {
        return DataType::T_TAG;
    }
    return crappyJavaStringHash(vespalib::LowerCase::convert(name));
}

} // anon namespace

DataType::DataType(vespalib::stringref name, int dataTypeId) noexcept
    : _dataTypeId(dataTypeId),
      _name(name)
{
}

DataType::DataType(vespalib::stringref name) noexcept
    : DataType(name, createId(name))
{
}

DataType::~DataType() = default;

void
DataType::buildFieldPath(FieldPath & path, vespalib::stringref remainFieldName) const
{
    if ( !remainFieldName.empty() ) {
        path.reserve(4);  // Optimize for short paths
        onBuildFieldPath(path, remainFieldName);
    }
}

const Field&
DataType::getField(int fieldId) const
{
    throw FieldNotFoundException(fieldId, 7, VESPA_STRLOC);
}

} // document
