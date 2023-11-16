// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "document_field_extractor.h"
#include <vespa/document/datatype/arraydatatype.h>
#include <vespa/document/fieldvalue/arrayfieldvalue.h>
#include <vespa/document/fieldvalue/boolfieldvalue.h>
#include <vespa/document/fieldvalue/bytefieldvalue.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/fieldvalue/doublefieldvalue.h>
#include <vespa/document/fieldvalue/floatfieldvalue.h>
#include <vespa/document/fieldvalue/intfieldvalue.h>
#include <vespa/document/fieldvalue/longfieldvalue.h>
#include <vespa/document/fieldvalue/shortfieldvalue.h>
#include <vespa/document/fieldvalue/stringfieldvalue.h>
#include <vespa/document/fieldvalue/mapfieldvalue.h>
#include <vespa/searchcommon/common/undefinedvalues.h>
#include <vespa/vespalib/stllike/hash_map.hpp>
#include <vespa/vespalib/util/exceptions.h>
#include <cassert>

using document::FieldValue;
using document::BoolFieldValue;
using document::ByteFieldValue;
using document::ShortFieldValue;
using document::IntFieldValue;
using document::LongFieldValue;
using document::FloatFieldValue;
using document::DoubleFieldValue;
using document::StringFieldValue;
using document::StructFieldValue;
using document::MapFieldValue;
using document::DataType;
using document::ArrayDataType;
using document::ArrayFieldValue;
using document::Document;
using document::FieldPath;
using document::FieldPathEntry;
using document::FieldValueVisitor;
using vespalib::IllegalStateException;
using vespalib::make_string;
using search::attribute::getUndefined;

namespace proton {

namespace {

class SetUndefinedValueVisitor : public FieldValueVisitor
{
    void visit(document::AnnotationReferenceFieldValue &) override { }
    void visit(ArrayFieldValue &) override { }
    void visit(BoolFieldValue &) override { }
    void visit(ByteFieldValue &value) override { value = getUndefined<int8_t>(); }
    void visit(Document &) override { }
    void visit(DoubleFieldValue &value) override { value = getUndefined<double>(); }
    void visit(FloatFieldValue &value) override { value = getUndefined<float>(); }
    void visit(IntFieldValue &value) override { value = getUndefined<int32_t>(); }
    void visit(LongFieldValue &value) override { value = getUndefined<int64_t>(); }
    void visit(MapFieldValue &) override { }
    void visit(document::PredicateFieldValue &) override { }
    void visit(document::RawFieldValue &) override { }
    void visit(ShortFieldValue &value) override { value = getUndefined<int16_t>(); }
    void visit(StringFieldValue &) override { }
    void visit(StructFieldValue &) override { }
    void visit(document::WeightedSetFieldValue &) override { }
    void visit(document::TensorFieldValue &) override { }
    void visit(document::ReferenceFieldValue &) override { }
};

SetUndefinedValueVisitor setUndefinedValueVisitor;

const ArrayDataType arrayTypeByte(*DataType::BYTE);
const ArrayDataType arrayTypeBool(*DataType::BOOL);
const ArrayDataType arrayTypeShort(*DataType::SHORT);
const ArrayDataType arrayTypeInt(*DataType::INT);
const ArrayDataType arrayTypeLong(*DataType::LONG);
const ArrayDataType arrayTypeFloat(*DataType::FLOAT);
const ArrayDataType arrayTypeDouble(*DataType::DOUBLE);
const ArrayDataType arrayTypeString(*DataType::STRING);

const DataType *
getArrayType(const DataType &fieldType)
{
    switch (fieldType.getId()) {
    case DataType::Type::T_BYTE:
        return &arrayTypeByte;
    case DataType::Type::T_BOOL:
        return &arrayTypeByte;
    case DataType::Type::T_SHORT:
        return &arrayTypeShort;
    case DataType::Type::T_INT:
        return &arrayTypeInt;
    case DataType::Type::T_LONG:
        return &arrayTypeLong;
    case DataType::Type::T_FLOAT:
        return &arrayTypeFloat;
    case DataType::Type::T_DOUBLE:
        return &arrayTypeDouble;
    case DataType::Type::T_STRING:
        return &arrayTypeString;
    default:
        return nullptr;
    }
}

std::unique_ptr<ArrayFieldValue>
makeArray(const FieldPathEntry &fieldPathEntry, size_t size)
{
    const auto arrayType = getArrayType(fieldPathEntry.getDataType());
    auto array = std::make_unique<ArrayFieldValue>(*arrayType);
    array->resize(size);
    return array;
}

}

DocumentFieldExtractor::DocumentFieldExtractor(const Document &doc)
    : _doc(doc),
      _cachedFieldValues()
{
}

DocumentFieldExtractor::~DocumentFieldExtractor() = default;

bool
DocumentFieldExtractor::isSupported(const FieldPath &fieldPath)
{
    if (!fieldPath.empty() &&
        fieldPath[0].getType() != FieldPathEntry::Type::STRUCT_FIELD) {
        return false;
    }
    if (fieldPath.size() == 2) {
        if (fieldPath[1].getType() != FieldPathEntry::Type::STRUCT_FIELD &&
            fieldPath[1].getType() != FieldPathEntry::Type::MAP_ALL_KEYS &&
            fieldPath[1].getType() != FieldPathEntry::Type::MAP_ALL_VALUES) {
            return false;
        }
    } else if (fieldPath.size() == 3) {
        if (fieldPath[1].getType() != FieldPathEntry::Type::MAP_ALL_VALUES ||
            fieldPath[2].getType() != FieldPathEntry::Type::STRUCT_FIELD) {
            return false;
        }
    } else if (fieldPath.size() > 3) {
        return false;
    }
    return true;
}

const FieldValue *
DocumentFieldExtractor::getCachedFieldValue(const FieldPathEntry &fieldPathEntry)
{
    auto itr = _cachedFieldValues.find(fieldPathEntry.getName());
    if (itr != _cachedFieldValues.end()) {
        return itr->second.get();
    } else {
        auto insres = _cachedFieldValues.insert(std::make_pair(fieldPathEntry.getName(), _doc.getValue(fieldPathEntry.getFieldRef())));
        assert(insres.second);
        return insres.first->second.get();
    }
}

std::unique_ptr<FieldValue>
DocumentFieldExtractor::getSimpleFieldValue(const FieldPath &fieldPath)
{
    return _doc.getNestedFieldValue(fieldPath.getFullRange());
}

namespace {

template <typename ExtractorFunc>
std::unique_ptr<FieldValue>
extractFieldFromMap(const FieldValue *outerFieldValue, const FieldPathEntry &innerEntry, ExtractorFunc &&extractor)
{
    if (outerFieldValue && outerFieldValue->isA(FieldValue::Type::MAP)) {
        const auto outerMap = static_cast<const MapFieldValue *>(outerFieldValue);
        auto array = makeArray(innerEntry, outerMap->size());
        uint32_t arrayIndex = 0;
        for (const auto &mapElem : *outerMap) {
            (*array)[arrayIndex++].assign(*extractor(mapElem));
        }
        return array;
    }
    return std::unique_ptr<FieldValue>();
}

template <typename CollectionFieldValueT, typename ExtractorFunc>
std::unique_ptr<FieldValue>
extractFieldFromStructCollection(const FieldValue *outerFieldValue, FieldValue::Type expectedType, const FieldPathEntry &innerEntry, ExtractorFunc &&extractor)
{
    if (outerFieldValue && outerFieldValue->isA(expectedType)) {
        const auto *outerCollection = static_cast<const CollectionFieldValueT *>(outerFieldValue);
        auto array = makeArray(innerEntry, outerCollection->size());
        uint32_t arrayIndex = 0;
        for (const auto &outerElem : *outerCollection) {
            auto &arrayElem = (*array)[arrayIndex++];
            const auto &structElem = static_cast<const StructFieldValue &>(*extractor(&outerElem));
            if (!structElem.getValue(innerEntry.getFieldRef(), arrayElem)) {
                arrayElem.accept(setUndefinedValueVisitor);
            }
        }
        return array;
    }
    return std::unique_ptr<FieldValue>();
}

}

std::unique_ptr<FieldValue>
DocumentFieldExtractor::extractFieldFromStructArray(const FieldPath &fieldPath)
{
    return extractFieldFromStructCollection<ArrayFieldValue>(getCachedFieldValue(fieldPath[0]), FieldValue::Type::ARRAY, fieldPath[1],
                                                             [](const auto *elem){ return elem; });
}

std::unique_ptr<FieldValue>
DocumentFieldExtractor::extractKeyFieldFromMap(const FieldPath &fieldPath)
{
    return extractFieldFromMap(getCachedFieldValue(fieldPath[0]), fieldPath[1],
                               [](const auto &elem){ return elem.first; });
}

std::unique_ptr<document::FieldValue>
DocumentFieldExtractor::extractValueFieldFromPrimitiveMap(const FieldPath &fieldPath)
{
    return extractFieldFromMap(getCachedFieldValue(fieldPath[0]), fieldPath[1],
                               [](const auto &elem){ return elem.second; });
}

std::unique_ptr<document::FieldValue>
DocumentFieldExtractor::extractValueFieldFromStructMap(const FieldPath &fieldPath)
{
    return extractFieldFromStructCollection<MapFieldValue>(getCachedFieldValue(fieldPath[0]), FieldValue::Type::MAP, fieldPath[2],
                                                           [](const auto *elem){ return elem->second; });
}

std::unique_ptr<FieldValue>
DocumentFieldExtractor::getFieldValue(const FieldPath &fieldPath)
{
    if (fieldPath.size() == 1) {
        return getSimpleFieldValue(fieldPath);
    } else if (fieldPath.size() == 2) {
        auto lastElemType = fieldPath[1].getType();
        if (lastElemType == FieldPathEntry::Type::STRUCT_FIELD) {
            return extractFieldFromStructArray(fieldPath);
        } else if (lastElemType == FieldPathEntry::Type::MAP_ALL_KEYS) {
            return extractKeyFieldFromMap(fieldPath);
        } else {
            return extractValueFieldFromPrimitiveMap(fieldPath);
        }
    } else if (fieldPath.size() == 3) {
        return extractValueFieldFromStructMap(fieldPath);
    }
    return std::unique_ptr<FieldValue>();
}

}
