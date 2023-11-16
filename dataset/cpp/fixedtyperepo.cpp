// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "fixedtyperepo.h"
#include <cassert>

namespace document {

FixedTypeRepo::FixedTypeRepo(const DocumentTypeRepo &repo, const vespalib::string &type) noexcept
    : _repo(&repo), _doc_type(repo.getDocumentType(type))
{
    assert(_doc_type);
}

}  // namespace document
