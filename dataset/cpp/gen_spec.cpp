// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "gen_spec.h"
#include <vespa/eval/eval/string_stuff.h>
#include <vespa/vespalib/util/require.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <ostream>

using vespalib::make_string_short::fmt;

namespace vespalib::eval::test {

//-----------------------------------------------------------------------------

Sequence N(double bias) {
    return [bias](size_t i) noexcept { return (i + bias); };
}

Sequence AX_B(double a, double b) {
    return [a,b](size_t i) noexcept { return (a * i) + b; };
}

Sequence Div16(const Sequence &seq) {
    return [seq](size_t i) { return (seq(i) / 16.0); };
}

Sequence Div17(const Sequence &seq) {
    return [seq](size_t i) { return (seq(i) / 17.0); };
}

Sequence Sub2(const Sequence &seq) {
    return [seq](size_t i) { return (seq(i) - 2.0); };
}

Sequence OpSeq(const Sequence &seq, map_fun_t op) {
    return [seq,op](size_t i) { return op(seq(i)); };
}

Sequence SigmoidF(const Sequence &seq) {
    return [seq](size_t i) { return (float)operation::Sigmoid::f(seq(i)); };
}

Sequence Seq(const std::vector<double> &seq) {
    REQUIRE(!seq.empty());
    return [seq](size_t i) noexcept { return seq[i % seq.size()]; };
}

//-----------------------------------------------------------------------------

DimSpec::DimSpec(const vespalib::string &name, size_t size) noexcept
    : _name(name), _size(size), _dict()
{
    assert(_size);
}
DimSpec::DimSpec(const vespalib::string &name, std::vector<vespalib::string> dict) noexcept
    : _name(name), _size(), _dict(std::move(dict))
{
    assert(!_size);
}

DimSpec::DimSpec(DimSpec &&) noexcept = default;
DimSpec::DimSpec(const DimSpec &) = default;
DimSpec & DimSpec::operator=(DimSpec &&) noexcept = default;
DimSpec & DimSpec::operator=(const DimSpec &) = default;

DimSpec::~DimSpec() = default;

std::vector<vespalib::string>
DimSpec::make_dict(size_t size, size_t stride, const vespalib::string &prefix)
{
    std::vector<vespalib::string> dict;
    for (size_t i = 0; i < size; ++i) {
        dict.push_back(fmt("%s%zu", prefix.c_str(), (i + 1) * stride));
    }
    return dict;
}

namespace {
auto is_dim_name = [](char c) { 
    return ((c >= 'a') && (c <= 'z'))
        || ((c >= 'A') && (c <= 'Z')); };
}

// 'a2' -> DimSpec("a", 2);
// 'b2_3' -> DimSpec("b", make_dict(2, 3, ""));
DimSpec
DimSpec::from_desc(const vespalib::string &desc)
{
    size_t idx = 0;
    vespalib::string name;
    auto is_num = [](char c) { return ((c >= '0') && (c <= '9')); };
    auto as_num = [](char c) { return size_t(c - '0'); };
    auto is_map_tag = [](char c) { return (c == '_'); };
    auto extract_number = [&]() {
        REQUIRE(idx < desc.size());
        REQUIRE(is_num(desc[idx]));
        size_t num = as_num(desc[idx++]);
        while ((idx < desc.size()) && is_num(desc[idx])) {
            num = (num * 10) + as_num(desc[idx++]);
        }
        return num;
    };
    REQUIRE(!desc.empty());
    REQUIRE(is_dim_name(desc[idx]));
    name.push_back(desc[idx++]);
    size_t size = extract_number();
    if (idx < desc.size()) {
        // mapped
        REQUIRE(is_map_tag(desc[idx++]));
        size_t stride = extract_number();
        REQUIRE(idx == desc.size());
        return {name, make_dict(size, stride, "")};
    } else {
        // indexed
        return {name, size};
    }
}

// 'a2b12c5' -> GenSpec().idx("a", 2).idx("b", 12).idx("c", 5);
// 'a2_1b3_2c5_1' -> GenSpec().map("a", 2).map("b", 3, 2).map("c", 5);
GenSpec
GenSpec::from_desc(const vespalib::string &desc)
{
    size_t idx = 0;
    vespalib::string dim_desc;
    std::vector<DimSpec> dim_list;
    while (idx < desc.size()) {
        dim_desc.clear();
        REQUIRE(is_dim_name(desc[idx]));
        dim_desc.push_back(desc[idx++]);
        while ((idx < desc.size()) && !is_dim_name(desc[idx])) {
            dim_desc.push_back(desc[idx++]);
        }
        dim_list.push_back(DimSpec::from_desc(dim_desc));
    }
    return {std::move(dim_list)};
}

GenSpec::GenSpec(GenSpec &&other) noexcept = default;
GenSpec::GenSpec(const GenSpec &other) = default;
GenSpec &GenSpec::operator=(GenSpec &&other) noexcept = default;
GenSpec &GenSpec::operator=(const GenSpec &other) = default;

GenSpec::~GenSpec() = default;

bool
GenSpec::bad_scalar() const
{
    return (_dims.empty() && (_cells != CellType::DOUBLE));
}

ValueType
GenSpec::type() const
{
    std::vector<ValueType::Dimension> dim_types;
    for (const auto &dim: _dims) {
        dim_types.push_back(dim.type());
    }
    auto type = ValueType::make_type(_cells, dim_types);
    REQUIRE(!type.is_error());
    return type;
}

TensorSpec
GenSpec::gen() const
{
    size_t idx = 0;
    TensorSpec::Address addr;   
    REQUIRE(!bad_scalar());
    TensorSpec result(type().to_spec());
    std::function<void(size_t)> add_cells = [&](size_t dim_idx) {
        if (dim_idx == _dims.size()) {
            result.add(addr, _seq(idx++));
        } else {
            const auto &dim = _dims[dim_idx];
            for (size_t i = 0; i < dim.size(); ++i) {
                addr.insert_or_assign(dim.name(), dim.label(i));
                add_cells(dim_idx + 1);
            }
        }
    };
    add_cells(0);
    return result.normalize();
}

std::ostream &operator<<(std::ostream &out, const GenSpec &spec)
{
    out << spec.gen();
    return out;
}

} // namespace
