/*
 * Copyright (C) 2015-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "canonical_mutation.hh"
#include "mutation.hh"
#include "mutation_partition_view.hh"
#include "mutation_partition_visitor.hh"
#include "mutation_partition_serializer.hh"
#include "counters.hh"
#include "converting_mutation_partition_applier.hh"
#include "hashing_partition_visitor.hh"
#include "idl/mutation.dist.hh"
#include "idl/mutation.dist.impl.hh"
#include <iostream>

canonical_mutation::canonical_mutation(bytes_ostream data)
        : _data(std::move(data))
{ }

canonical_mutation::canonical_mutation(const mutation& m)
{
    mutation_partition_serializer part_ser(*m.schema(), m.partition());

    ser::writer_of_canonical_mutation<bytes_ostream> wr(_data);
    std::move(wr).write_table_id(m.schema()->id())
                 .write_schema_version(m.schema()->version())
                 .write_key(m.key())
                 .write_mapping(m.schema()->get_column_mapping())
                 .partition([&] (auto wr) {
                     part_ser.write(std::move(wr));
                 }).end_canonical_mutation();
}

table_id canonical_mutation::column_family_id() const {
    auto in = ser::as_input_stream(_data);
    auto mv = ser::deserialize(in, boost::type<ser::canonical_mutation_view>());
    return mv.table_id();
}

mutation canonical_mutation::to_mutation(schema_ptr s) const {
    auto in = ser::as_input_stream(_data);
    auto mv = ser::deserialize(in, boost::type<ser::canonical_mutation_view>());

    auto cf_id = mv.table_id();
    if (s->id() != cf_id) {
        throw std::runtime_error(format("Attempted to deserialize canonical_mutation of table {} with schema of table {} ({}.{})",
                                        cf_id, s->id(), s->ks_name(), s->cf_name()));
    }

    auto version = mv.schema_version();
    auto pk = mv.key();

    mutation m(std::move(s), std::move(pk));

    if (version == m.schema()->version()) {
        auto partition_view = mutation_partition_view::from_view(mv.partition());
        mutation_application_stats app_stats;
        m.partition().apply(*m.schema(), partition_view, *m.schema(), app_stats);
    } else {
        column_mapping cm = mv.mapping();
        converting_mutation_partition_applier v(cm, *m.schema(), m.partition());
        auto partition_view = mutation_partition_view::from_view(mv.partition());
        partition_view.accept(cm, v);
    }
    return m;
}

static sstring bytes_to_text(bytes_view bv) {
    sstring ret = uninitialized_string(bv.size());
    std::copy_n(reinterpret_cast<const char*>(bv.data()), bv.size(), ret.data());
    return ret;
}

std::ostream& operator<<(std::ostream& os, const canonical_mutation& cm) {
    auto in = ser::as_input_stream(cm._data);
    auto mv = ser::deserialize(in, boost::type<ser::canonical_mutation_view>());
    column_mapping mapping = mv.mapping();
    auto partition_view = mutation_partition_view::from_view(mv.partition());
    fmt::print(os, "{{canonical_mutation: ");
    fmt::print(os, "table_id {} schema_version {} ", mv.table_id(), mv.schema_version());
    fmt::print(os, "partition_key {} ", mv.key());
    class printing_visitor : public mutation_partition_view_virtual_visitor {
        std::ostream& _os;
        const column_mapping& _cm;
        bool _first = true;
        bool _in_row = false;
    private:
        void print_separator() {
            if (!_first) {
                fmt::print(_os, ", ");
            }
            _first = false;
        }
    public:
        printing_visitor(std::ostream& os, const column_mapping& cm) : _os(os), _cm(cm) {}
        virtual void accept_partition_tombstone(tombstone t) override {
            print_separator();
            fmt::print(_os, "partition_tombstone {}", t);
        }
        virtual void accept_static_cell(column_id id, atomic_cell ac) override {
            print_separator();
            auto&& entry = _cm.static_column_at(id);
            fmt::print(_os, "static column {} {}", bytes_to_text(entry.name()), atomic_cell::printer(*entry.type(), ac));
        }
        virtual void accept_static_cell(column_id id, collection_mutation_view cmv) override {
            print_separator();
            auto&& entry = _cm.static_column_at(id);
            fmt::print(_os, "static column {} {}", bytes_to_text(entry.name()), collection_mutation_view::printer(*entry.type(), cmv));
        }
        virtual stop_iteration accept_row_tombstone(range_tombstone rt) override {
            print_separator();
            fmt::print(_os, "row tombstone {}", rt);
            return stop_iteration::no;
        }
        virtual stop_iteration accept_row(position_in_partition_view pipv, row_tombstone rt, row_marker rm, is_dummy, is_continuous) override {
            if (_in_row) {
                fmt::print(_os, "}}, ");
            }
            fmt::print(_os, "{{row {} tombstone {} marker {}", pipv, rt, rm);
            _in_row = true;
            _first = false;
            return stop_iteration::no;
        }
        virtual void accept_row_cell(column_id id, atomic_cell ac) override {
            print_separator();
            auto&& entry = _cm.regular_column_at(id);
            fmt::print(_os, "column {} {}", bytes_to_text(entry.name()), atomic_cell::printer(*entry.type(), ac));
        }
        virtual void accept_row_cell(column_id id, collection_mutation_view cmv) override {
            print_separator();
            auto&& entry = _cm.regular_column_at(id);
            fmt::print(_os, "column {} {}", bytes_to_text(entry.name()), collection_mutation_view::printer(*entry.type(), cmv));
        }
        void finalize() {
            if (_in_row) {
                fmt::print(_os, "}}");
            }
        }
    };
    printing_visitor pv(os, mapping);
    partition_view.accept(mapping, pv);
    pv.finalize();
    fmt::print(os, "}}");
    return os;
}

