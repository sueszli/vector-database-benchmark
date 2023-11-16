/*
 * Copyright 2016-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <optional>

#include "commitlog_types.hh"
#include "mutation/frozen_mutation.hh"
#include "schema/schema_fwd.hh"

class commitlog_entry {
    std::optional<column_mapping> _mapping;
    frozen_mutation _mutation;
public:
    commitlog_entry(std::optional<column_mapping> mapping, frozen_mutation&& mutation)
        : _mapping(std::move(mapping)), _mutation(std::move(mutation)) { }
    const std::optional<column_mapping>& mapping() const { return _mapping; }
    const frozen_mutation& mutation() const & { return _mutation; }
    frozen_mutation&& mutation() && { return std::move(_mutation); }
};

class commitlog_entry_writer {
public:
    using force_sync = db::commitlog_force_sync;
private:
    schema_ptr _schema;
    const frozen_mutation& _mutation;
    bool _with_schema = true;
    size_t _size = std::numeric_limits<size_t>::max();
    force_sync _sync;
private:
    template<typename Output>
    void serialize(Output&) const;
    void compute_size();
public:
    commitlog_entry_writer(schema_ptr s, const frozen_mutation& fm, force_sync sync)
        : _schema(std::move(s)), _mutation(fm), _sync(sync)
    {}

    void set_with_schema(bool value) {
        _with_schema = value;
        compute_size();
    }
    bool with_schema() const {
        return _with_schema;
    }
    schema_ptr schema() const {
        return _schema;
    }

    size_t size() const {
        assert(_size != std::numeric_limits<size_t>::max());
        return _size;
    }

    size_t mutation_size() const {
        return _mutation.representation().size();
    }
    force_sync sync() const {
        return _sync;
    }
    void write(typename seastar::memory_output_stream<std::vector<temporary_buffer<char>>::iterator>& out) const;
};

class commitlog_entry_reader {
    commitlog_entry _ce;
public:
    commitlog_entry_reader(const fragmented_temporary_buffer& buffer);

    const std::optional<column_mapping>& get_column_mapping() const { return _ce.mapping(); }
    const frozen_mutation& mutation() const & { return _ce.mutation(); }
    frozen_mutation&& mutation() && { return std::move(_ce).mutation(); }
};
