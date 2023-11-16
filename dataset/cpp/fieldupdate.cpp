// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "fieldupdate.h"
#include <vespa/document/base/exceptions.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/datatype/documenttype.h>
#include <vespa/vespalib/objects/nbostream.h>
#include <ostream>

namespace document {

using vespalib::nbostream;

FieldUpdate::FieldUpdate(const Field& field)
    : _field(field),
      _updates()
{
}

namespace {

int readInt(nbostream & stream) {
    int tmp;
    stream >> tmp;
    return tmp;
}

}

FieldUpdate::FieldUpdate(const DocumentTypeRepo& repo, const DataType & type, nbostream & stream)
    : _field(type.getField(readInt(stream))),
      _updates()
{
    int numUpdates = readInt(stream);
    _updates.reserve(numUpdates);
    const DataType& dataType = _field.getDataType();
    for(int i(0); i < numUpdates; i++) {
        _updates.emplace_back(ValueUpdate::createInstance(repo, dataType, stream));
    }
}

FieldUpdate::~FieldUpdate() = default;

bool
FieldUpdate::operator==(const FieldUpdate& other) const
{
    if (_field != other._field) return false;
    if (_updates.size() != other._updates.size()) return false;
    for (uint32_t i=0, n=_updates.size(); i<n; ++i) {
        if (*_updates[i] != *other._updates[i]) return false;
    }
    return true;
}


FieldUpdate&
FieldUpdate::addUpdate(std::unique_ptr<ValueUpdate> update) & {
    update->checkCompatibility(_field); // May throw exception.
    _updates.push_back(std::move(update));
    return *this;
}

FieldUpdate&&
FieldUpdate::addUpdate(std::unique_ptr<ValueUpdate> update) && {
    addUpdate(std::move(update));
    return std::move(*this);
}

void
FieldUpdate::printXml(XmlOutputStream& xos) const
{
    for(const auto & update : _updates) {
        update->printXml(xos);
    }
}

// Apply this field update to the given document.
void
FieldUpdate::applyTo(Document& doc) const
{
    const DataType& datatype = _field.getDataType();
    FieldValue::UP value = doc.getValue(_field);

    for (const auto & update : _updates) {
        if ( ! value) {
            // Avoid passing a null pointer to a value update.
            value = datatype.createFieldValue();
        }
        if (!update->applyTo(*value)) {
            value.reset();
        }
    }

    if (value) {
        doc.setFieldValue(_field, std::move(value));
    } else {
        doc.remove(_field);
    }
}

// Print this field update as a human readable string.
void
FieldUpdate::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "FieldUpdate(" << _field.toString(verbose);
    for(const auto & update : _updates) {
        out << "\n" << indent << "  ";
        update->print(out, verbose, indent + "  ");
    }
    if (_updates.size() > 0) {
        out << "\n" << indent;
    }
    out << ")";
}

// Deserialize this field update from the given buffer.
void
FieldUpdate::deserialize(const DocumentTypeRepo& repo, const DocumentType& docType, nbostream& stream)
{
    int fieldId = readInt(stream);
    _field = docType.getField(fieldId);
    const DataType& dataType = _field.getDataType();

    int numUpdates = readInt(stream);
    _updates.clear();
    _updates.resize(numUpdates);
    for(int i = 0; i < numUpdates; i++) {
        _updates[i].reset(ValueUpdate::createInstance(repo, dataType, stream).release());
    }
}

}  // namespace document
