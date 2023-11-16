/*
 * Copyright (C) 2015-present ScyllaDB
 *
 * Modified by ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#pragma once

#include "cql3/statements/property_definitions.hh"
#include "data_dictionary/storage_options.hh"

#include <seastar/core/shared_ptr.hh>
#include <seastar/core/sstring.hh>
#include <optional>

namespace data_dictionary {
class keyspace_metadata;
}

namespace locator {
    class token_metadata;
    class shared_token_metadata;
    struct snitch_ptr;
    class abstract_replication_strategy;
} // namespace locator

namespace cql3 {

namespace statements {

class ks_prop_defs : public property_definitions {
public:
    static constexpr auto KW_DURABLE_WRITES = "durable_writes";
    static constexpr auto KW_REPLICATION = "replication";
    static constexpr auto KW_STORAGE = "storage";

    static constexpr auto REPLICATION_STRATEGY_CLASS_KEY = "class";
    static constexpr auto REPLICATION_FACTOR_KEY = "replication_factor";
private:
    std::optional<sstring> _strategy_class;
public:
    void validate();
    std::map<sstring, sstring> get_replication_options() const;
    std::optional<sstring> get_replication_strategy_class() const;
    data_dictionary::storage_options get_storage_options() const;
    lw_shared_ptr<data_dictionary::keyspace_metadata> as_ks_metadata(sstring ks_name, const locator::token_metadata&);
    lw_shared_ptr<data_dictionary::keyspace_metadata> as_ks_metadata_update(lw_shared_ptr<data_dictionary::keyspace_metadata> old, const locator::token_metadata&);

#if 0
    public KSMetaData asKSMetadataUpdate(KSMetaData old) throws RequestValidationException
    {
        String sClass = strategyClass;
        Map<String, String> sOptions = getReplicationOptions();
        if (sClass == null)
        {
            sClass = old.strategyClass.getName();
            sOptions = old.strategyOptions;
        }
        return KSMetaData.newKeyspace(old.name, sClass, sOptions, getBoolean(KW_DURABLE_WRITES, old.durableWrites));
    }
#endif
};

}

}
