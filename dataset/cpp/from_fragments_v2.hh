/*
 * Copyright (C) 2022-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once
#include "schema/schema_fwd.hh"
#include <deque>
#include "dht/i_partitioner_fwd.hh"

class flat_mutation_reader_v2;
class reader_permit;
class mutation_fragment_v2;
class ring_position;

namespace query {
    class partition_slice;
}

flat_mutation_reader_v2
make_flat_mutation_reader_from_fragments(schema_ptr, reader_permit, std::deque<mutation_fragment_v2>);

flat_mutation_reader_v2
make_flat_mutation_reader_from_fragments(schema_ptr, reader_permit, std::deque<mutation_fragment_v2>, const dht::partition_range& pr);

flat_mutation_reader_v2
make_flat_mutation_reader_from_fragments(schema_ptr, reader_permit, std::deque<mutation_fragment_v2>, const dht::partition_range& pr, const query::partition_slice& slice);

