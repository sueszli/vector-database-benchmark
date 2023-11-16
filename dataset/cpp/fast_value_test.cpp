// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/fast_value.hpp>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/eval/eval/test/gen_spec.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/gtest/gtest.h>

using namespace vespalib;
using namespace vespalib::make_string_short;
using namespace vespalib::eval;
using namespace vespalib::eval::test;

using Handle = SharedStringRepo::Handle;

TEST(FastCellsTest, push_back_fast_works) {
    FastCells<float> cells(3);
    EXPECT_EQ(cells.capacity, 4);
    EXPECT_EQ(cells.size, 0);
    cells.push_back_fast(1.0);
    cells.push_back_fast(2.0);
    cells.push_back_fast(3.0);
    EXPECT_EQ(cells.capacity, 4);
    EXPECT_EQ(cells.size, 3);
    cells.ensure_free(3);
    EXPECT_EQ(cells.capacity, 8);
    EXPECT_EQ(cells.size, 3);
    cells.push_back_fast(4.0);
    cells.push_back_fast(5.0);
    cells.push_back_fast(6.0);
    EXPECT_EQ(cells.capacity, 8);
    EXPECT_EQ(cells.size, 6);
    auto usage = cells.estimate_extra_memory_usage();
    EXPECT_EQ(usage.allocatedBytes(), sizeof(float) * 8);
    EXPECT_EQ(usage.usedBytes(), sizeof(float) * 6);
    EXPECT_EQ(*cells.get(0), 1.0);
    EXPECT_EQ(*cells.get(1), 2.0);
    EXPECT_EQ(*cells.get(2), 3.0);
    EXPECT_EQ(*cells.get(3), 4.0);
    EXPECT_EQ(*cells.get(4), 5.0);
    EXPECT_EQ(*cells.get(5), 6.0);
}

TEST(FastCellsTest, add_cells_works) {
    FastCells<float> cells(3);
    auto arr1 = cells.add_cells(3);
    EXPECT_EQ(cells.capacity, 4);
    EXPECT_EQ(cells.size, 3);
    arr1[0] = 1.0;
    arr1[1] = 2.0;
    arr1[2] = 3.0;
    auto arr2 = cells.add_cells(3);
    EXPECT_EQ(cells.capacity, 8);
    EXPECT_EQ(cells.size, 6);
    arr2[0] = 4.0;
    arr2[1] = 5.0;
    arr2[2] = 6.0;
    EXPECT_EQ(*cells.get(0), 1.0);
    EXPECT_EQ(*cells.get(1), 2.0);
    EXPECT_EQ(*cells.get(2), 3.0);
    EXPECT_EQ(*cells.get(3), 4.0);
    EXPECT_EQ(*cells.get(4), 5.0);
    EXPECT_EQ(*cells.get(5), 6.0);
}

TEST(FastValueTest, insert_subspace) {
    Handle foo("foo");
    Handle bar("bar");
    string_id foo_id = foo.id();
    string_id bar_id = bar.id();
    auto addr = [](string_id &ref){ return ConstArrayRef<string_id>(&ref, 1); };
    auto type = ValueType::from_spec("tensor<float>(x{},y[2])");
    auto value = std::make_unique<FastValue<float,true>>(type, 1, 2, 5);
    EXPECT_EQ(value->index().size(), 0);
    {
        auto [cells, added] = value->insert_subspace(addr(foo_id));
        EXPECT_TRUE(added);
        EXPECT_EQ(value->index().size(), 1);
        ASSERT_EQ(cells.size(), 2);
        cells[0] = 10.0;
        cells[1] = 20.0;
    }{
        auto [cells, added] = value->insert_subspace(addr(bar_id));
        EXPECT_TRUE(added);
        EXPECT_EQ(value->index().size(), 2);
        ASSERT_EQ(cells.size(), 2);
        cells[0] = 30.0;
        cells[1] = 40.0;
    }{
        auto [cells, added] = value->insert_subspace(addr(foo_id));
        EXPECT_FALSE(added);
        EXPECT_EQ(value->index().size(), 2);
        ASSERT_EQ(cells.size(), 2);
        EXPECT_EQ(cells[0], 10.0);
        EXPECT_EQ(cells[1], 20.0);
        cells[0] = 11.0;
        cells[1] = 22.0;
    }{
        auto [cells, added] = value->insert_subspace(addr(bar_id));
        EXPECT_FALSE(added);
        EXPECT_EQ(value->index().size(), 2);
        ASSERT_EQ(cells.size(), 2);
        EXPECT_EQ(cells[0], 30.0);
        EXPECT_EQ(cells[1], 40.0);
        cells[0] = 33.0;
        cells[1] = 44.0;
    }
    auto actual = spec_from_value(*value);
    auto expected = TensorSpec("tensor<float>(x{},y[2])")
        .add({{"x", "foo"}, {"y", 0}}, 11.0)
        .add({{"x", "foo"}, {"y", 1}}, 22.0)
        .add({{"x", "bar"}, {"y", 0}}, 33.0)
        .add({{"x", "bar"}, {"y", 1}}, 44.0);
    EXPECT_EQ(actual, expected);
}

TEST(FastValueTest, insert_empty_subspace) {
    auto addr = []() { return ConstArrayRef<string_id>(); };
    auto type = ValueType::from_spec("double");
    auto value = std::make_unique<FastValue<double, true>>(type, 0, 1, 1);
    EXPECT_EQ(value->index().size(), 0);
    {
        auto [cells, added] = value->insert_subspace(addr());
        EXPECT_TRUE(added);
        EXPECT_EQ(value->index().size(), 1);
        ASSERT_EQ(cells.size(), 1);
        cells[0] = 10.0;
    }
    {
        auto [cells, added] = value->insert_subspace(addr());
        EXPECT_FALSE(added);
        EXPECT_EQ(value->index().size(), 1);
        ASSERT_EQ(cells.size(), 1);
        EXPECT_EQ(cells[0], 10.0);
        cells[0] = 11.0;
    }
    auto actual = spec_from_value(*value);
    auto expected = TensorSpec("double").add({}, 11.0);
    EXPECT_EQ(actual, expected);
}

void
verifyFastValueSize(TensorSpec spec, uint32_t elems, size_t expected) {
    for (uint32_t i=0; i < elems; i++) {
        spec.add({{"country", fmt("no%d", i)}}, 17.0);
    }
    auto value = value_from_spec(spec, FastValueBuilderFactory::get());
    EXPECT_EQ(expected, value->get_memory_usage().allocatedBytes());
}

TEST(FastValueTest, document_fast_value_memory_usage) {
    EXPECT_EQ(232, sizeof(FastValue<float,true>));
    FastValue<float,true> test(ValueType::from_spec("tensor<float>(country{})"), 1, 1, 1);
    EXPECT_EQ(412, test.get_memory_usage().allocatedBytes());

    verifyFastValueSize(TensorSpec("tensor<float>(country{})"), 1, 412);
    verifyFastValueSize(TensorSpec("tensor<float>(country{})"), 10, 792);
    verifyFastValueSize(TensorSpec("tensor<float>(country{})"), 20, 1280);
    verifyFastValueSize(TensorSpec("tensor<float>(country{})"), 50, 2296);
    verifyFastValueSize(TensorSpec("tensor<float>(country{})"), 100, 4288);
}

using SA = std::vector<vespalib::stringref>;

TEST(FastValueBuilderTest, scalar_add_subspace_robustness) {
    auto factory = FastValueBuilderFactory::get();
    ValueType type = ValueType::from_spec("double");
    auto builder = factory.create_value_builder<double>(type);
    auto subspace = builder->add_subspace();
    subspace[0] = 17.0;
    auto other = builder->add_subspace();
    other[0] = 42.0;
    auto value = builder->build(std::move(builder));
    EXPECT_EQ(value->index().size(), 1);
    auto actual = spec_from_value(*value);
    auto expected = TensorSpec("double").
                    add({}, 42.0);
    EXPECT_EQ(actual, expected);
}

TEST(FastValueBuilderTest, dense_add_subspace_robustness) {
    auto factory = FastValueBuilderFactory::get();
    ValueType type = ValueType::from_spec("tensor(x[2])");
    auto builder = factory.create_value_builder<double>(type);
    auto subspace = builder->add_subspace();
    subspace[0] = 17.0;
    subspace[1] = 666;
    auto other = builder->add_subspace();
    other[1] = 42.0;
    auto value = builder->build(std::move(builder));
    EXPECT_EQ(value->index().size(), 1);
    auto actual = spec_from_value(*value);
    auto expected = TensorSpec("tensor(x[2])").
                    add({{"x", 0}}, 17.0).
                    add({{"x", 1}}, 42.0);
    EXPECT_EQ(actual, expected);    
}

TEST(FastValueBuilderTest, mixed_add_subspace_robustness) {
    auto factory = FastValueBuilderFactory::get();
    ValueType type = ValueType::from_spec("tensor(x{},y[2])");
    auto builder = factory.create_value_builder<double>(type);
    auto subspace = builder->add_subspace(SA{"foo"});
    subspace[0] = 1.0;
    subspace[1] = 5.0;
    subspace = builder->add_subspace(SA{"bar"});
    subspace[0] = 2.0;
    subspace[1] = 10.0;
    auto other = builder->add_subspace(SA{"foo"});
    other[0] = 3.0;
    other[1] = 15.0;
    auto value = builder->build(std::move(builder));
    EXPECT_EQ(value->index().size(), 3);
    Handle foo("foo");
    Handle bar("bar");
    string_id label;
    string_id *label_ptr = &label;
    size_t subspace_idx;
    auto get_subspace = [&]() {
        auto cells = value->cells().typify<double>();
        return ConstArrayRef<double>(cells.begin() + subspace_idx * 2, 2);
    };
    auto view = value->index().create_view({});
    view->lookup({});
    while (view->next_result({&label_ptr, 1}, subspace_idx)) {
        if (label == bar.id()) {
            auto values = get_subspace();
            EXPECT_EQ(values[0], 2.0);
            EXPECT_EQ(values[1], 10.0);
        } else {
            EXPECT_EQ(label, foo.id());
            auto values = get_subspace();
            if (values[0] == 1) {
                EXPECT_EQ(values[1], 5.0);
            } else {
                EXPECT_EQ(values[0], 3.0);
                EXPECT_EQ(values[1], 15.0);
            }
        }
    }
}

GenSpec G() { return GenSpec(); }

const std::vector<GenSpec> layouts = {
    G(),
    G().idx("x", 3),
    G().idx("x", 3).idx("y", 5),
    G().idx("x", 3).idx("y", 5).idx("z", 7),
    G().map("x", {"a","b","c"}),
    G().map("x", {"a","b","c"}).map("y", {"foo","bar"}),
    G().map("x", {"a","b","c"}).map("y", {"foo","bar"}).map("z", {"i","j","k","l"}),
    G().idx("x", 3).map("y", {"foo", "bar"}).idx("z", 7),
    G().map("x", {"a","b","c"}).idx("y", 5).map("z", {"i","j","k","l"})
};

TEST(FastValueBuilderFactoryTest, fast_values_can_be_copied) {
    auto factory = FastValueBuilderFactory::get();
    for (const auto &layout: layouts) {
        for (CellType ct : CellTypeUtils::list_types()) {
            auto expect = layout.cpy().cells(ct);
            if (expect.bad_scalar()) continue;
            std::unique_ptr<Value> value = value_from_spec(expect, factory);
            std::unique_ptr<Value> copy = factory.copy(*value);
            TensorSpec actual = spec_from_value(*copy);
            EXPECT_EQ(actual, expect);
        }
    }
}

GTEST_MAIN_RUN_ALL_TESTS()
