/*
 * Copyright (C) 2022-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#pragma once
#include "schema/schema_fwd.hh"

class flat_mutation_reader_v2;
class reader_permit;

flat_mutation_reader_v2 make_empty_flat_reader_v2(schema_ptr s, reader_permit permit);

