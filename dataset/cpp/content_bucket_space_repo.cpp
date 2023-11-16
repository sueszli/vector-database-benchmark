// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "content_bucket_space_repo.h"
#include <vespa/document/bucket/fixed_bucket_spaces.h>

using document::BucketSpace;

namespace storage {

ContentBucketSpaceRepo::ContentBucketSpaceRepo(const ContentBucketDbOptions& db_opts)
    : _map()
{
    _map.emplace(document::FixedBucketSpaces::default_space(),
                 std::make_unique<ContentBucketSpace>(document::FixedBucketSpaces::default_space(), db_opts));
    _map.emplace(document::FixedBucketSpaces::global_space(),
                 std::make_unique<ContentBucketSpace>(document::FixedBucketSpaces::global_space(), db_opts));
}

ContentBucketSpace &
ContentBucketSpaceRepo::get(BucketSpace bucketSpace) const
{
    auto itr = _map.find(bucketSpace);
    assert(itr != _map.end());
    return *itr->second;
}

ContentBucketSpaceRepo::BucketSpaces
ContentBucketSpaceRepo::getBucketSpaces() const
{
    BucketSpaces result;
    for (const auto &elem : _map) {
        result.push_back(elem.first);
    }
    return result;
}

size_t
ContentBucketSpaceRepo::getBucketMemoryUsage() const
{
    size_t result = 0;
    for (const auto &elem : _map) {
        result += elem.second->bucketDatabase().getMemoryUsage();
    }
    return result;
}

}
