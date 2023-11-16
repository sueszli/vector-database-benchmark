/*
 *
 * Modified by ScyllaDB
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: (AGPL-3.0-or-later and Apache-2.0)
 */

#pragma once

#include "schema/schema_fwd.hh"
#include <ostream>

namespace streaming {

/**
 * Summary of streaming.
 */
class stream_summary {
public:
    table_id cf_id;

    /**
     * Number of files to transfer. Can be 0 if nothing to transfer for some streaming request.
     */
    int files;
    long total_size;

    stream_summary() = default;
    stream_summary(table_id _cf_id, int _files, long _total_size)
        : cf_id (_cf_id)
        , files(_files)
        , total_size(_total_size) {
    }
    friend std::ostream& operator<<(std::ostream& os, const stream_summary& r);
};

} // namespace streaming
