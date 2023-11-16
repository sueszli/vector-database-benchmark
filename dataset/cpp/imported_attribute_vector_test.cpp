// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/test/value_compare.h>
#include <vespa/searchcommon/attribute/search_context_params.h>
#include <vespa/searchlib/fef/termfieldmatchdata.h>
#include <vespa/searchlib/tensor/i_tensor_attribute.h>
#include <vespa/searchlib/tensor/tensor_attribute.h>
#include <vespa/searchlib/test/imported_attribute_fixture.h>

using search::attribute::IAttributeVector;
using search::tensor::ITensorAttribute;
using search::tensor::TensorAttribute;
using vespalib::eval::Value;
using vespalib::eval::ValueType;
using vespalib::eval::TensorSpec;
using vespalib::eval::SimpleValue;

Value::UP createTensor(const TensorSpec &spec) {
    return SimpleValue::from_spec(spec);
}

namespace search::attribute {

using Fixture = ImportedAttributeFixture;

TEST_F("Accessors return expected attributes", Fixture) {
    EXPECT_EQUAL(f.imported_attr->getReferenceAttribute().get(),
                 f.reference_attr.get());
    EXPECT_EQUAL(f.imported_attr->getTargetAttribute().get(),
                 f.target_attr.get());
}

TEST_F("getName() is equal to name given during construction", Fixture) {
    auto attr = f.create_attribute_vector_from_members("coolvector");
    EXPECT_EQUAL("coolvector", attr->getName());
    EXPECT_EQUAL("coolvector", attr->makeReadGuard(false)->attribute()->getName());
}

TEST_F("getNumDocs() returns number of documents in reference attribute vector", Fixture) {
    add_n_docs_with_undefined_values(*f.reference_attr, 42);
    EXPECT_EQUAL(42u, f.get_imported_attr()->getNumDocs());
}

TEST_F("hasEnum() is false for non-enum target attribute vector", Fixture) {
    EXPECT_FALSE(f.get_imported_attr()->hasEnum());
}

TEST_F("Collection type is inherited from target attribute", Fixture) {
    EXPECT_EQUAL(CollectionType::SINGLE, f.get_imported_attr()->getCollectionType());
    f.reset_with_new_target_attr(create_array_attribute<IntegerAttribute>(BasicType::INT32));
    EXPECT_EQUAL(CollectionType::ARRAY, f.get_imported_attr()->getCollectionType());
}

TEST_F("getBasicType() returns target vector basic type", Fixture) {
    f.reset_with_new_target_attr(create_single_attribute<IntegerAttribute>(BasicType::INT64));
    EXPECT_EQUAL(BasicType::INT64, f.get_imported_attr()->getBasicType());
    f.reset_with_new_target_attr(create_single_attribute<FloatingPointAttribute>(BasicType::DOUBLE));
    EXPECT_EQUAL(BasicType::DOUBLE, f.get_imported_attr()->getBasicType());
}

TEST_F("makeReadGuard(false) acquires guards on both target and reference attributes", Fixture) {
    add_n_docs_with_undefined_values(*f.reference_attr, 2);
    add_n_docs_with_undefined_values(*f.target_attr, 2);
    // Now at generation 1 in both attributes.
    {
        auto guard = f.imported_attr->makeReadGuard(false);
        add_n_docs_with_undefined_values(*f.reference_attr, 1);
        add_n_docs_with_undefined_values(*f.target_attr, 1);
        
        EXPECT_EQUAL(2u, f.target_attr->getCurrentGeneration());
        EXPECT_EQUAL(2u, f.reference_attr->getCurrentGeneration());
        // Should still be holding guard for first generation of writes for both attributes
        EXPECT_EQUAL(1u, f.target_attr->get_oldest_used_generation());
        EXPECT_EQUAL(1u, f.reference_attr->get_oldest_used_generation());
    }
    // Force a generation handler update
    add_n_docs_with_undefined_values(*f.reference_attr, 1);
    add_n_docs_with_undefined_values(*f.target_attr, 1);
    EXPECT_EQUAL(3u, f.target_attr->get_oldest_used_generation());
    EXPECT_EQUAL(3u, f.reference_attr->get_oldest_used_generation());
}

TEST_F("makeReadGuard(true) acquires enum guard on target and regular guard on reference attribute", Fixture) {
    f.reset_with_new_target_attr(create_single_attribute<StringAttribute>(BasicType::STRING));
    add_n_docs_with_undefined_values(*f.reference_attr, 2);
    add_n_docs_with_undefined_values(*f.target_attr, 2);
    {
        auto guard = f.imported_attr->makeReadGuard(true);
        add_n_docs_with_undefined_values(*f.target_attr, 1);
        add_n_docs_with_undefined_values(*f.reference_attr, 1);

        EXPECT_EQUAL(5u, f.target_attr->getCurrentGeneration());
        EXPECT_EQUAL(2u, f.reference_attr->getCurrentGeneration());

        EXPECT_EQUAL(3u, f.target_attr->get_oldest_used_generation());
        EXPECT_EQUAL(1u, f.reference_attr->get_oldest_used_generation());
        EXPECT_TRUE(has_active_enum_guards(*f.target_attr));
    }
    // Force a generation handler update
    add_n_docs_with_undefined_values(*f.reference_attr, 1);
    add_n_docs_with_undefined_values(*f.target_attr, 1);
    EXPECT_EQUAL(7u, f.target_attr->get_oldest_used_generation());
    EXPECT_EQUAL(3u, f.reference_attr->get_oldest_used_generation());
    EXPECT_FALSE(has_active_enum_guards(*f.target_attr));
}

TEST_F("Single-valued integer attribute values can be retrieved via reference", Fixture)
{
    reset_with_single_value_reference_mappings<IntegerAttribute, int32_t>(
            f, BasicType::INT32,
            {{DocId(1), dummy_gid(3), DocId(3), 1234},
             {DocId(3), dummy_gid(7), DocId(7), 5678}});

    EXPECT_EQUAL(1234, f.get_imported_attr()->getInt(DocId(1)));
    EXPECT_EQUAL(5678, f.get_imported_attr()->getInt(DocId(3)));
}

TEST_F("getValueCount() is 1 for mapped single value attribute", Fixture)
{
    reset_with_single_value_reference_mappings<IntegerAttribute, int32_t>(
            f, BasicType::INT32, {{DocId(1), dummy_gid(3), DocId(3), 1234}});
    EXPECT_EQUAL(1u, f.get_imported_attr()->getValueCount(DocId(1)));
}

TEST_F("getValueCount() is 0 for non-mapped single value attribute", Fixture)
{
    add_n_docs_with_undefined_values(*f.reference_attr, 3);
    EXPECT_EQUAL(0u, f.get_imported_attr()->getValueCount(DocId(2)));
}

TEST_F("getMaxValueCount() is 1 for single value attribute vectors", Fixture) {
    EXPECT_EQUAL(1u, f.get_imported_attr()->getMaxValueCount());
}

TEST_F("getFixedWidth() is inherited from target attribute vector", Fixture) {
    EXPECT_EQUAL(f.target_attr->getFixedWidth(),
                 f.get_imported_attr()->getFixedWidth());
}

TEST_F("asDocumentWeightAttribute() returns nullptr", Fixture) {
    EXPECT_TRUE(f.get_imported_attr()->asDocumentWeightAttribute() == nullptr);
}

TEST_F("asTensorAttribute() returns nullptr", Fixture) {
    EXPECT_TRUE(f.get_imported_attr()->asTensorAttribute() == nullptr);
}

TEST_F("isImported() returns true", Fixture) {
    EXPECT_TRUE(f.get_imported_attr()->isImported());
}

TEST_F("Multi-valued integer attribute values can be retrieved via reference", Fixture) {
    const std::vector<int64_t> doc3_values({1234});
    const std::vector<int64_t> doc7_values({5678, 9876, 555, 777});
    const std::vector<int64_t> doc8_values({});
    reset_with_array_value_reference_mappings<IntegerAttribute, int64_t>(
            f, BasicType::INT64,
            {{DocId(1), dummy_gid(3), DocId(3), doc3_values},
             {DocId(3), dummy_gid(7), DocId(7), doc7_values},
             {DocId(5), dummy_gid(8), DocId(8), doc8_values}});
    assert_multi_value_matches<IAttributeVector::largeint_t>(f, DocId(1), doc3_values);
    assert_multi_value_matches<IAttributeVector::largeint_t>(f, DocId(3), doc7_values);
    assert_multi_value_matches<IAttributeVector::largeint_t>(f, DocId(5), doc8_values);
}

TEST_F("Weighted integer attribute values can be retrieved via reference", Fixture) {
    const std::vector<WeightedInt> doc3_values({WeightedInt(1234, 5)});
    const std::vector<WeightedInt> doc7_values({WeightedInt(5678, 10), WeightedInt(9876, 20)});
    reset_with_wset_value_reference_mappings<IntegerAttribute, WeightedInt>(
            f, BasicType::INT32,
            {{DocId(1), dummy_gid(3), DocId(3), doc3_values},
             {DocId(3), dummy_gid(7), DocId(7), doc7_values}});
    assert_multi_value_matches<WeightedInt>(f, DocId(1), doc3_values);
    assert_multi_value_matches<WeightedInt>(f, DocId(3), doc7_values);
}

TEST_F("LID with not present GID reference mapping returns default value", Fixture)
{
    f.target_attr->addReservedDoc();
    add_n_docs_with_undefined_values(*f.reference_attr, 2);
    EXPECT_EQUAL(f.target_attr->getInt(DocId(0)), // Implicit default undefined value
                 f.get_imported_attr()->getInt(DocId(1)));
}

TEST_F("Singled-valued floating point attribute values can be retrieved via reference", Fixture) {
    reset_with_single_value_reference_mappings<FloatingPointAttribute, float>(
            f, BasicType::FLOAT,
            {{DocId(2), dummy_gid(3), DocId(3), 10.5f},
             {DocId(4), dummy_gid(8), DocId(8), 3.14f}});

    EXPECT_EQUAL(10.5f, f.get_imported_attr()->getFloat(DocId(2)));
    EXPECT_EQUAL(3.14f, f.get_imported_attr()->getFloat(DocId(4)));
}

TEST_F("Multi-valued floating point attribute values can be retrieved via reference", Fixture) {
    const std::vector<double> doc3_values({3.14, 133.7});
    const std::vector<double> doc7_values({5.5,  6.5, 10.5});
    reset_with_array_value_reference_mappings<FloatingPointAttribute, double>(
            f, BasicType::DOUBLE,
            {{DocId(2), dummy_gid(3), DocId(3), doc3_values},
             {DocId(4), dummy_gid(7), DocId(7), doc7_values}});
    assert_multi_value_matches<double>(f, DocId(2), doc3_values);
    assert_multi_value_matches<double>(f, DocId(4), doc7_values);
}

TEST_F("Weighted floating point attribute values can be retrieved via reference", Fixture) {
    const std::vector<WeightedFloat> doc3_values({WeightedFloat(3.14, 5)});
    const std::vector<WeightedFloat> doc7_values({WeightedFloat(5.5, 7), WeightedFloat(10.25, 42)});
    reset_with_wset_value_reference_mappings<FloatingPointAttribute, WeightedFloat>(
            f, BasicType::DOUBLE,
            {{DocId(1), dummy_gid(3), DocId(3), doc3_values},
             {DocId(3), dummy_gid(7), DocId(7), doc7_values}});
    assert_multi_value_matches<WeightedFloat>(f, DocId(1), doc3_values);
    assert_multi_value_matches<WeightedFloat>(f, DocId(3), doc7_values);
}

TEST_F("isUndefined() works for primitive attribute type", Fixture) {
    reset_with_single_value_reference_mappings<IntegerAttribute, int32_t>(
            f, BasicType::INT32,
            {{DocId(3), dummy_gid(7), DocId(7), 5678}});

    EXPECT_FALSE(f.get_imported_attr()->isUndefined(DocId(3))); // Mapped
    EXPECT_TRUE(f.get_imported_attr()->isUndefined(DocId(2))); // Not mapped
}

TEST_F("original lid range is used by read guard", Fixture)
{
    reset_with_single_value_reference_mappings<IntegerAttribute, int32_t>(
            f, BasicType::INT32,
            {{DocId(1), dummy_gid(3), DocId(3), 1234}});
    auto first_guard = f.get_imported_attr();
    add_n_docs_with_undefined_values(*f.reference_attr, 1);
    f.map_reference(DocId(10), dummy_gid(3), DocId(3));
    auto second_guard = f.get_imported_attr();
    EXPECT_EQUAL(1234, second_guard->getInt(DocId(10)));
    EXPECT_NOT_EQUAL(1234, first_guard->getInt(DocId(10)));
    EXPECT_EQUAL(getUndefined<int>(), first_guard->getInt(DocId(10)));
}

TEST_F("Original target lid range is used by read guard", Fixture)
{
    reset_with_single_value_reference_mappings<IntegerAttribute, int32_t>(
            f, BasicType::INT32,
            {});
    EXPECT_EQUAL(11u, f.target_attr->getNumDocs());
    auto first_guard = f.get_imported_attr();
    add_n_docs_with_undefined_values(*f.target_attr, 1);
    EXPECT_EQUAL(12u, f.target_attr->getNumDocs());
    auto typed_target_attr = f.template target_attr_as<IntegerAttribute>();
    ASSERT_TRUE(typed_target_attr->update(11, 2345));
    f.target_attr->commit();
    f.map_reference(DocId(8), dummy_gid(11), DocId(11));
    auto second_guard = f.get_imported_attr();
    EXPECT_EQUAL(2345, second_guard->getInt(DocId(8)));
    EXPECT_NOT_EQUAL(2345, first_guard->getInt(DocId(8)));
}

struct SingleStringAttrFixture : Fixture {
    SingleStringAttrFixture() : Fixture() {
        setup();
    }
    ~SingleStringAttrFixture() override;

    void setup() {
        this->template reset_with_single_value_reference_mappings<StringAttribute, const char*>(
                BasicType::STRING,
                {{DocId(2), dummy_gid(3), DocId(3), "foo"},
                 {DocId(4), dummy_gid(7), DocId(7), "bar"}});
    }
};

SingleStringAttrFixture::~SingleStringAttrFixture() = default;

TEST_F("Single-valued string attribute values can be retrieved via reference", SingleStringAttrFixture)
{
    auto buf = f.get_imported_attr()->get_raw(DocId(2));
    EXPECT_EQUAL(vespalib::stringref("foo"), vespalib::stringref(buf.data(), buf.size()));
    buf = f.get_imported_attr()->get_raw(DocId(4));
    EXPECT_EQUAL(vespalib::stringref("bar"), vespalib::stringref(buf.data(), buf.size()));
}

TEST_F("getEnum() returns target vector enum via reference", SingleStringAttrFixture)
{
    EXPECT_EQUAL(f.target_attr->getEnum(DocId(3)),
                 f.get_imported_attr()->getEnum(DocId(2)));
    EXPECT_EQUAL(f.target_attr->getEnum(DocId(7)),
                 f.get_imported_attr()->getEnum(DocId(4)));
}

TEST_F("findEnum() returns target vector enum via reference", SingleStringAttrFixture) {
    EnumHandle expected_handle{};
    ASSERT_TRUE(f.target_attr->findEnum("foo", expected_handle));
    EnumHandle actual_handle{};
    ASSERT_TRUE(f.get_imported_attr()->findEnum("foo", actual_handle));
    EXPECT_EQUAL(expected_handle, actual_handle);
}

TEST_F("isUndefined() works for enumerated attribute type", SingleStringAttrFixture) {
    EXPECT_FALSE(f.get_imported_attr()->isUndefined(DocId(2))); // Mapped
    EXPECT_TRUE(f.get_imported_attr()->isUndefined(DocId(3))); // Not mapped
}

// Note: assumes that fixture has set up a string enum of value "foo" in target attribute
template <typename FixtureType>
void verify_get_string_from_enum_is_mapped(FixtureType& f) {
    EnumHandle handle{};
    ASSERT_TRUE(f.target_attr->findEnum("foo", handle));
    const char* from_enum = f.get_imported_attr()->getStringFromEnum(handle);
    ASSERT_TRUE(from_enum != nullptr);
    EXPECT_EQUAL(vespalib::string("foo"), vespalib::string(from_enum));
}

TEST_F("Single-value getStringFromEnum() returns string enum is mapped to", SingleStringAttrFixture) {
    verify_get_string_from_enum_is_mapped(f);
}

TEST_F("hasEnum() is true for enum target attribute vector", SingleStringAttrFixture) {
    EXPECT_TRUE(f.get_imported_attr()->hasEnum());
}

TEST_F("createSearchContext() returns an imported search context", SingleStringAttrFixture) {
    auto ctx = f.get_imported_attr()->createSearchContext(word_term("bar"), SearchContextParams());
    ASSERT_TRUE(ctx.get() != nullptr);
    fef::TermFieldMatchData match;
    // Iterator specifics are tested in imported_search_context_test, so just make sure
    // we get the expected iterator functionality. In this case, a non-strict iterator.
    auto iter = ctx->createIterator(&match, false);
    iter->initRange(1, f.get_imported_attr()->getNumDocs());
    EXPECT_FALSE(iter->seek(DocId(1)));
    EXPECT_FALSE(iter->seek(DocId(2)));
    EXPECT_FALSE(iter->seek(DocId(3)));
    EXPECT_TRUE(iter->seek(DocId(4)));
}

bool string_eq(const char* lhs, const char* rhs) noexcept {
    return strcmp(lhs, rhs) == 0;
};

template <typename T>
std::vector<T> as_vector(const AttributeContent<T>& content) {
    return {content.begin(), content.end()};
}

struct MultiStringAttrFixture : Fixture {
    std::vector<const char*> doc3_values{{"foo", "bar"}};
    std::vector<const char*> doc7_values{{"baz", "bjarne", "betjent"}};

    MultiStringAttrFixture() : Fixture() {
        setup();
    }
    ~MultiStringAttrFixture() override;

    void setup() {
        reset_with_array_value_reference_mappings<StringAttribute, const char *>(
                BasicType::STRING,
                {{DocId(2), dummy_gid(3), DocId(3), doc3_values},
                 {DocId(4), dummy_gid(7), DocId(7), doc7_values}});
    }
};

MultiStringAttrFixture::~MultiStringAttrFixture() = default;

TEST_F("Multi-valued string attribute values can be retrieved via reference", MultiStringAttrFixture) {
    assert_multi_value_matches<const char*>(f, DocId(2), f.doc3_values, string_eq);
    assert_multi_value_matches<const char*>(f, DocId(4), f.doc7_values, string_eq);
}

TEST_F("Multi-valued enum attribute values can be retrieved via reference", MultiStringAttrFixture) {
    AttributeContent<EnumHandle> expected;
    expected.fill(*f.target_attr, DocId(3));
    assert_multi_value_matches<EnumHandle>(f, DocId(2), as_vector(expected));
}

TEST_F("Multi-value getStringFromEnum() returns string enum is mapped to", MultiStringAttrFixture) {
    verify_get_string_from_enum_is_mapped(f);
}

TEST_F("getValueCount() is equal to stored values for mapped multi value attribute", MultiStringAttrFixture) {
    EXPECT_EQUAL(f.doc7_values.size(), f.get_imported_attr()->getValueCount(DocId(4)));
}

TEST_F("getMaxValueCount() is greater than 1 for multi value attribute vectors", MultiStringAttrFixture) {
    EXPECT_GREATER(f.get_imported_attr()->getMaxValueCount(), 1u);
}

struct WeightedMultiStringAttrFixture : Fixture {
    std::vector<WeightedString> doc3_values{{WeightedString("foo", 5)}};
    std::vector<WeightedString> doc7_values{{WeightedString("bar", 7), WeightedString("baz", 42)}};

    WeightedMultiStringAttrFixture() : Fixture() {
        setup();
    }
    ~WeightedMultiStringAttrFixture() override;

    void setup() {
        reset_with_wset_value_reference_mappings<StringAttribute, WeightedString>(
                BasicType::STRING,
                {{DocId(1), dummy_gid(3), DocId(3), doc3_values},
                 {DocId(3), dummy_gid(7), DocId(7), doc7_values}});
    }
};

WeightedMultiStringAttrFixture::~WeightedMultiStringAttrFixture() = default;

TEST_F("Weighted string attribute values can be retrieved via reference", WeightedMultiStringAttrFixture) {
    assert_multi_value_matches<WeightedString>(f, DocId(1), f.doc3_values);
    assert_multi_value_matches<WeightedString>(f, DocId(3), f.doc7_values);
}

TEST_F("Weighted enum attribute values can be retrieved via reference", WeightedMultiStringAttrFixture) {
    AttributeContent<WeightedEnum> expected;
    expected.fill(*f.target_attr, DocId(7));
    assert_multi_value_matches<WeightedEnum>(f, DocId(3), as_vector(expected));
}

bool weighted_string_eq(const WeightedConstChar& lhs, const WeightedConstChar& rhs) noexcept {
    if (lhs.weight() != rhs.weight()) {
        return false;
    }
    return (strcmp(lhs.value(), rhs.value()) == 0);
};

TEST_F("Weighted const char attribute values can be retrieved via reference", WeightedMultiStringAttrFixture) {
    AttributeContent<WeightedConstChar> expected;
    expected.fill(*f.target_attr, DocId(7));

    assert_multi_value_matches<WeightedConstChar>(f, DocId(3), as_vector(expected), weighted_string_eq);
}

TEST_F("Weighted set getStringFromEnum() returns string enum is mapped to", WeightedMultiStringAttrFixture) {
    verify_get_string_from_enum_is_mapped(f);
}

// Poor man's function call mock matching
struct MockAttributeVector : NotImplementedAttribute {
    // Mutable is dirty, but funcs are called in a const context and we know
    // there won't be multiple threads touching anything.
    mutable DocId _doc_id{0};
    mutable void* _ser_to{nullptr};
    mutable long  _available{0};
    mutable const common::BlobConverter* _bc{nullptr};
    mutable bool _ascending_called{false};
    mutable bool _descending_called{false};

    long _return_value{1234};

    MockAttributeVector()
            : NotImplementedAttribute("mock") {
    }

    void set_received_args(DocId doc_id, void* ser_to,
                           long available, const common::BlobConverter* bc) const {
        _doc_id = doc_id;
        _ser_to = ser_to;
        _available = available;
        _bc = bc;
    }

    long onSerializeForAscendingSort(
            DocId doc_id, void* ser_to,
            long available, const common::BlobConverter* bc) const override {
        set_received_args(doc_id, ser_to, available, bc);
        _ascending_called = true;
        return _return_value;
    }
    long onSerializeForDescendingSort(
            DocId doc_id, void* ser_to,
            long available, const common::BlobConverter* bc) const override {
        set_received_args(doc_id, ser_to, available, bc);
        _descending_called = true;
        return _return_value;
    }

    // Not covered by NotImplementedAttribute
    void onCommit() override {}
    void onUpdateStat() override {}
};

struct MockBlobConverter : common::BlobConverter {
    ConstBufferRef onConvert(const ConstBufferRef&) const override {
        return ConstBufferRef();
    }
};

template <typename BaseFixture>
struct SerializeFixture : BaseFixture {
    std::shared_ptr<MockAttributeVector> mock_target;
    MockBlobConverter mock_converter;

    SerializeFixture() : mock_target(std::make_shared<MockAttributeVector>()) {
        this->reset_with_new_target_attr(mock_target);
        mock_target->setCommittedDocIdLimit(8); // Target LID of 7 is highest used by ref attribute. Limit is +1.
    }
    ~SerializeFixture() override;
};

template <typename BaseFixture>
SerializeFixture<BaseFixture>::~SerializeFixture() {}

template <typename FixtureT>
void check_onSerializeForAscendingSort_is_forwarded_with_remapped_lid() {
    FixtureT f;
    int dummy_tag;
    void* ser_to = &dummy_tag;
    EXPECT_EQUAL(f.mock_target->_return_value,
                 f.get_imported_attr()->serializeForAscendingSort(
                         DocId(4), ser_to, 777, &f.mock_converter)); // child lid 4 -> parent lid 7
    EXPECT_TRUE(f.mock_target->_ascending_called);
    EXPECT_EQUAL(DocId(7), f.mock_target->_doc_id);
    EXPECT_EQUAL(ser_to, f.mock_target->_ser_to);
    EXPECT_EQUAL(777, f.mock_target->_available);
    EXPECT_EQUAL(&f.mock_converter, f.mock_target->_bc);
}

TEST("onSerializeForAscendingSort() is forwarded with remapped LID to target vector") {
    TEST_DO(check_onSerializeForAscendingSort_is_forwarded_with_remapped_lid<
                    SerializeFixture<SingleStringAttrFixture>>());
}

template <typename FixtureT>
void check_onSerializeForDescendingSort_is_forwarded_with_remapped_lid() {
    FixtureT f;
    int dummy_tag;
    void* ser_to = &dummy_tag;
    EXPECT_EQUAL(f.mock_target->_return_value,
                 f.get_imported_attr()->serializeForDescendingSort(
                         DocId(2), ser_to, 555, &f.mock_converter)); // child lid 2 -> parent lid 3
    EXPECT_TRUE(f.mock_target->_descending_called);
    EXPECT_EQUAL(DocId(3), f.mock_target->_doc_id);
    EXPECT_EQUAL(ser_to, f.mock_target->_ser_to);
    EXPECT_EQUAL(555, f.mock_target->_available);
    EXPECT_EQUAL(&f.mock_converter, f.mock_target->_bc);
}

TEST("onSerializeForDescendingSort() is forwarded with remapped LID to target vector") {
    TEST_DO(check_onSerializeForDescendingSort_is_forwarded_with_remapped_lid<
                    SerializeFixture<SingleStringAttrFixture>>());
}

struct TensorAttrFixture : Fixture {
    std::shared_ptr<Value> tensor1;
    std::shared_ptr<Value> tensor2;

    TensorAttrFixture(bool dense)
        : Fixture(),
          tensor1(),
          tensor2()
    {
        setup(dense);
    }
    ~TensorAttrFixture() override;
    void setup(bool dense) {
        if (dense) {
            tensor1 = createTensor(TensorSpec("tensor(x[2])").add({{"x", 1}}, 11));
            tensor2 = createTensor(TensorSpec("tensor(x[2])").add({{"x", 0}}, 12).add({{"x", 1}}, 0));
        } else {
            tensor1 = createTensor(TensorSpec("tensor(x{})").add({{"x", "1"}}, 11));
            tensor2 = createTensor(TensorSpec("tensor(x{})").add({{"x", "0"}}, 12));
        }
        const std::vector<ImportedAttributeFixture::LidToLidMapping<std::shared_ptr<Value>>> mappings =
            {   {DocId(2), dummy_gid(3), DocId(3), tensor1 },
                {DocId(4), dummy_gid(7), DocId(7), tensor2 } };
        this->template reset_with_tensor_reference_mappings<TensorAttribute, std::shared_ptr<Value>>(
                ValueType::from_spec(dense ? "tensor(x[2])" : "tensor(x{})"),
                mappings);
    }
    Value::UP getTensor(DocId docId) {
        auto imp_attr = this->get_imported_attr();
        const ITensorAttribute *tensorAttr = imp_attr->asTensorAttribute();
        ASSERT_TRUE(tensorAttr != nullptr);
        return tensorAttr->getTensor(docId);
    }
    void assertNoTensor(DocId docId) {
        auto tensor = getTensor(docId);
        EXPECT_TRUE(!tensor);
    }
    void assertTensor(DocId docId, const Value &expTensor) {
        auto tensor = getTensor(docId);
        ASSERT_TRUE(!!tensor);
        EXPECT_EQUAL(expTensor, *tensor);
    }
    void assertTensors() {
        assertNoTensor(0);
        assertNoTensor(1);
        assertTensor(2, *tensor1);
        assertNoTensor(3);
        assertTensor(4, *tensor2);
    }
};

TensorAttrFixture::~TensorAttrFixture() = default;

TEST_F("Imported sparse tensor", TensorAttrFixture(false))
{
    f.assertTensors();
}

TEST_F("Imported dense tensor", TensorAttrFixture(true))
{
    f.assertTensors();
}

}

TEST_MAIN() { TEST_RUN_ALL(); }
