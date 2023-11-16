// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "removefieldpathupdate.h"
#include <vespa/document/fieldvalue/iteratorhandler.h>
#include <ostream>

namespace document {

using namespace fieldvalue;

RemoveFieldPathUpdate::RemoveFieldPathUpdate() noexcept
    : FieldPathUpdate(Remove)
{
}

RemoveFieldPathUpdate::~RemoveFieldPathUpdate() = default;

RemoveFieldPathUpdate::RemoveFieldPathUpdate(stringref fieldPath, stringref whereClause)
    : FieldPathUpdate(Remove, fieldPath, whereClause)
{
}

void
RemoveFieldPathUpdate::print(std::ostream& out, bool verbose, const std::string& indent) const
{
    out << "RemoveFieldPathUpdate(\n";
    FieldPathUpdate::print(out, verbose, indent + "  ");
    out << "\n" << indent << ")";
}

void
RemoveFieldPathUpdate::deserialize(const DocumentTypeRepo& repo, const DataType& type, nbostream & stream)
{
    FieldPathUpdate::deserialize(repo, type, stream);
}

namespace {

class RemoveIteratorHandler : public IteratorHandler {
public:
    RemoveIteratorHandler() = default;

    ModificationStatus doModify(FieldValue &) override {
        return ModificationStatus::REMOVED;
    }
};

}

std::unique_ptr<IteratorHandler>
RemoveFieldPathUpdate::getIteratorHandler(Document&, const DocumentTypeRepo &) const {
    return std::make_unique<RemoveIteratorHandler>();
}

} // ns document
