// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "resource_usage.h"
#include <iostream>

namespace storage::spi {

ResourceUsage::ResourceUsage(const ResourceUsage &rhs) = default;

ResourceUsage::ResourceUsage(ResourceUsage &&rhs) = default;

ResourceUsage::~ResourceUsage() = default;

ResourceUsage&
ResourceUsage::operator=(const ResourceUsage &rhs) = default;

ResourceUsage&
ResourceUsage::operator=(ResourceUsage &&rhs) = default;

std::ostream& operator<<(std::ostream& out, const ResourceUsage& resource_usage)
{
    out << "{disk_usage=" << resource_usage.get_disk_usage() <<
        ", memory_usage=" << resource_usage.get_memory_usage() <<
        ", attribute_address_space_usage=" << resource_usage.get_attribute_address_space_usage() << "}";
    return out;
}

}
