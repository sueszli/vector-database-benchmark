/*
 * Copyright (C) 2022-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once

class flat_mutation_reader_v2;

// Create an adaptor which provides a next_partition() implementation for
// readers which don't have one.
// `next_partition()` is implemented by discarding fragments until the next one
// is a partition start one.
// The returned reader doesn't support any form of fast-forwarding.
flat_mutation_reader_v2 make_next_partition_adaptor(flat_mutation_reader_v2&& rd);
