// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/searchlib/tensor/dense_tensor_store.h>
#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/value_type.h>
#include <vespa/eval/eval/test/value_compare.h>
#include <vespa/vespalib/util/size_literals.h>

#include <vespa/log/log.h>
LOG_SETUP("dense_tensor_store_test");

using search::tensor::DenseTensorStore;
using vespalib::eval::SimpleValue;
using vespalib::eval::TensorSpec;
using vespalib::eval::Value;
using vespalib::eval::ValueType;

using EntryRef = DenseTensorStore::EntryRef;

Value::UP
makeTensor(const TensorSpec &spec)
{
    return SimpleValue::from_spec(spec);
}

struct Fixture
{
    DenseTensorStore store;
    explicit Fixture(const vespalib::string &tensorType)
        : store(ValueType::from_spec(tensorType), {})
    {}
    void assertSetAndGetTensor(const TensorSpec &tensorSpec) {
        Value::UP expTensor = makeTensor(tensorSpec);
        EntryRef ref = store.store_tensor(*expTensor);
        Value::UP actTensor = store.get_tensor(ref);
        EXPECT_EQUAL(*expTensor, *actTensor);
        assertTensorView(ref, *expTensor);
    }
    void assertEmptyTensor(const TensorSpec &tensorSpec) const {
        Value::UP expTensor = makeTensor(tensorSpec);
        EntryRef ref;
        Value::UP actTensor = store.get_tensor(ref);
        EXPECT_TRUE(actTensor.get() == nullptr);
        assertTensorView(ref, *expTensor);
    }
    void assertTensorView(EntryRef ref, const Value &expTensor) const {
        auto cells = store.get_typed_cells(ref);
        vespalib::eval::DenseValueView actTensor(store.type(), cells);
        EXPECT_EQUAL(expTensor, actTensor);
    }
};

TEST_F("require that we can store 1d bound tensor", Fixture("tensor(x[3])"))
{
    f.assertSetAndGetTensor(TensorSpec("tensor(x[3])").
                                       add({{"x", 0}}, 2).
                                       add({{"x", 1}}, 3).
                                       add({{"x", 2}}, 5));
}

TEST_F("require that correct empty tensor is returned for 1d bound tensor", Fixture("tensor(x[3])"))
{
    f.assertEmptyTensor(TensorSpec("tensor(x[3])").
                                   add({{"x", 0}}, 0).
                                   add({{"x", 1}}, 0).
                                   add({{"x", 2}}, 0));
}

void
assertArraySize(const vespalib::string &tensorType, uint32_t expArraySize) {
    Fixture f(tensorType);
    EXPECT_EQUAL(expArraySize, f.store.getArraySize());
}

TEST("require that array size is calculated correctly")
{
    TEST_DO(assertArraySize("tensor(x[1])", 8));
    TEST_DO(assertArraySize("tensor(x[10])", 96));
    TEST_DO(assertArraySize("tensor(x[3])", 32));
    TEST_DO(assertArraySize("tensor(x[10],y[10])", 800));
    TEST_DO(assertArraySize("tensor<int8>(x[1])", 8));
    TEST_DO(assertArraySize("tensor<int8>(x[8])", 8));
    TEST_DO(assertArraySize("tensor<int8>(x[9])", 16));
    TEST_DO(assertArraySize("tensor<int8>(x[16])", 16));
    TEST_DO(assertArraySize("tensor<int8>(x[17])", 32));
    TEST_DO(assertArraySize("tensor<int8>(x[32])", 32));
    TEST_DO(assertArraySize("tensor<int8>(x[33])", 64));
    TEST_DO(assertArraySize("tensor<int8>(x[64])", 64));
    TEST_DO(assertArraySize("tensor<int8>(x[65])", 96));
}

void
assert_max_buffer_entries(const vespalib::string& tensor_type, uint32_t exp_entries)
{
    Fixture f(tensor_type);
    EXPECT_EQUAL(exp_entries, f.store.get_max_buffer_entries());
}

TEST("require that max entries is calculated correctly")
{
    TEST_DO(assert_max_buffer_entries("tensor(x[1])", 1_Mi));

    TEST_DO(assert_max_buffer_entries("tensor(x[32])", 1_Mi));
    TEST_DO(assert_max_buffer_entries("tensor(x[64])", 512_Ki));
    TEST_DO(assert_max_buffer_entries("tensor(x[1024])", 32_Ki));
    TEST_DO(assert_max_buffer_entries("tensor(x[1024])", 32_Ki));
    TEST_DO(assert_max_buffer_entries("tensor(x[16777216])", 2));
    TEST_DO(assert_max_buffer_entries("tensor(x[33554428])", 2));
    TEST_DO(assert_max_buffer_entries("tensor(x[33554429])", 1));
    TEST_DO(assert_max_buffer_entries("tensor(x[33554432])", 1));
}

TEST_MAIN() { TEST_RUN_ALL(); }

