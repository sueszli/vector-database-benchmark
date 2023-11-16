// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "idocumentstore.h"
#include <vespa/document/fieldvalue/document.h>

namespace search {

void IDocumentStore::visit(const LidVector & lids, const document::DocumentTypeRepo &repo, IDocumentVisitor & visitor) const {
    for (uint32_t lid : lids) {
        visitor.visit(lid, read(lid, repo));
    }
}

} // namespace search
