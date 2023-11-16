/*
 * Copyright (C) 2018-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

#include <vector>
#include <seastar/core/shared_ptr.hh>

#include "schema/schema_fwd.hh"
#include "sstables/types.hh"
#include "db/marshal/type_parser.hh"

namespace sstables {

using column_values_fixed_lengths = std::vector<std::optional<uint32_t>>;

/*
 * A helper method to get fixed lengths of clustering key values
 */
inline column_values_fixed_lengths get_clustering_values_fixed_lengths(const serialization_header& header) {
    column_values_fixed_lengths lengths;
    lengths.reserve(header.clustering_key_types_names.elements.size());
    for (auto&& t : header.clustering_key_types_names.elements) {
        auto type = db::marshal::type_parser::parse(to_sstring_view(t.value));
        lengths.push_back(type->value_length_if_fixed());
    }

    return lengths;
}

/*
 * This class caches a mapping from columns present in sstable to their column_id.
 * This way we don't need to looku them up by column name every time.
 */
class column_translation {
public:
    struct column_info {
        const bytes* name = nullptr;
        data_type type;
        // Disengaged 'id' means the column is missing from the current schema
        std::optional<column_id> id;
        std::optional<uint32_t> value_length;
        bool is_collection;
        bool is_counter;
        bool schema_mismatch;
    };

private:

    struct state {

        static std::vector<column_info> build(
                const schema& s,
                const utils::chunked_vector<serialization_header::column_desc>& src,
                const sstable_enabled_features& features,
                bool is_static);

        table_schema_version schema_uuid;
        std::vector<column_info> regular_schema_columns_from_sstable;
        std::vector<column_info> static_schema_columns_from_sstable;
        column_values_fixed_lengths clustering_column_value_fix_lengths;

        state() = default;
        state(const state&) = delete;
        state& operator=(const state&) = delete;
        state(state&&) = default;
        state& operator=(state&&) = default;

        state(const schema& s, const serialization_header& header, const sstable_enabled_features& features)
            : schema_uuid(s.version())
            , regular_schema_columns_from_sstable(build(s, header.regular_columns.elements, features, false))
            , static_schema_columns_from_sstable(build(s, header.static_columns.elements, features, true))
            , clustering_column_value_fix_lengths (get_clustering_values_fixed_lengths(header))
        {}
    };

    lw_shared_ptr<const state> _state = make_lw_shared<const state>();

public:
    column_translation get_for_schema(
            const schema& s, const serialization_header& header, const sstable_enabled_features& features) {
        if (s.version() != _state->schema_uuid) {
            _state = make_lw_shared<const state>(s, header, features);
        }
        return *this;
    }

    const std::vector<column_info>& regular_columns() const {
        return _state->regular_schema_columns_from_sstable;
    }
    const std::vector<column_info>& static_columns() const {
        return _state->static_schema_columns_from_sstable;
    }
    const std::vector<std::optional<uint32_t>>& clustering_column_value_fix_legths() const {
        return _state->clustering_column_value_fix_lengths;
    }
};

};   // namespace sstables
