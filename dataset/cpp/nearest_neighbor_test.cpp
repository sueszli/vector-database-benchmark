// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/searchcommon/attribute/config.h>
#include <vespa/searchlib/common/bitvector.h>
#include <vespa/searchlib/common/feature.h>
#include <vespa/searchlib/fef/matchdata.h>
#include <vespa/searchlib/queryeval/global_filter.h>
#include <vespa/searchlib/queryeval/nearest_neighbor_iterator.h>
#include <vespa/searchlib/queryeval/nns_index_iterator.h>
#include <vespa/searchlib/queryeval/simpleresult.h>
#include <vespa/searchlib/tensor/dense_tensor_attribute.h>
#include <vespa/searchlib/tensor/distance_calculator.h>
#include <vespa/searchlib/tensor/distance_function_factory.h>
#include <vespa/searchlib/tensor/serialized_fast_value_attribute.h>
#include <vespa/vespalib/gtest/gtest.h>
#include <vespa/vespalib/test/insertion_operators.h>
#include <vespa/vespalib/util/stringfmt.h>

#include <vespa/log/log.h>
LOG_SETUP("nearest_neighbor_test");

#define EPS 1.0e-6

using search::AttributeVector;
using search::BitVector;
using search::attribute::DistanceMetric;
using search::feature_t;
using search::tensor::DenseTensorAttribute;
using search::tensor::DistanceCalculator;
using search::tensor::SerializedFastValueAttribute;
using search::tensor::TensorAttribute;
using vespalib::eval::CellType;
using vespalib::eval::SimpleValue;
using vespalib::eval::TensorSpec;
using vespalib::eval::Value;
using vespalib::eval::ValueType;

using namespace search::fef;
using namespace search::queryeval;

using BasicType = search::attribute::BasicType;
using CollectionType = search::attribute::CollectionType;
using Config = search::attribute::Config;

vespalib::string denseSpecDouble("tensor(x[2])");
vespalib::string denseSpecFloat("tensor<float>(x[2])");
vespalib::string mixed_spec("tensor(m{},x[2])");

std::unique_ptr<Value> createTensor(const TensorSpec &spec) {
    return SimpleValue::from_spec(spec);
}

std::unique_ptr<Value> createTensor(const vespalib::string& type_spec, double v1, double v2) {
    auto type = vespalib::eval::ValueType::from_spec(type_spec);
    if (type.is_dense()) {
        return createTensor(TensorSpec(type_spec).add({{"x", 0}}, v1)
                                                 .add({{"x", 1}}, v2));
    } else {
        return createTensor(TensorSpec(type_spec).add({{"m", "a"},{"x", 0}}, v1)
                                                 .add({{"m", "a"},{"x", 1}}, v2));
    }
}

std::shared_ptr<TensorAttribute> make_attr(const vespalib::string& name, const Config& cfg) {
    if (cfg.tensorType().is_dense()) {
        return std::make_shared<DenseTensorAttribute>(name, cfg);
    } else {
        return std::make_shared<SerializedFastValueAttribute>(name, cfg);
    }
}

struct Fixture {
    Config _cfg;
    vespalib::string _name;
    vespalib::string _typeSpec;
    std::shared_ptr<TensorAttribute> _attr;
    std::shared_ptr<GlobalFilter> _global_filter;

    Fixture(const vespalib::string &typeSpec)
        : _cfg(BasicType::TENSOR, CollectionType::SINGLE),
          _name("test"),
          _typeSpec(typeSpec),
          _attr(),
          _global_filter(GlobalFilter::create())
    {
        _cfg.setTensorType(ValueType::from_spec(typeSpec));
        _attr = make_attr(_name, _cfg);
        _attr->addReservedDoc();
    }

    ~Fixture() {}

    void ensureSpace(uint32_t docId) {
        while (_attr->getNumDocs() <= docId) {
            uint32_t newDocId = 0u;
            _attr->addDoc(newDocId);
            _attr->commit();
        }
    }

    void setFilter(std::vector<uint32_t> docids) {
        uint32_t sz = _attr->getNumDocs();
        _global_filter = GlobalFilter::create(docids, sz);
    }

    void setTensor(uint32_t docId, const Value &tensor) {
        ensureSpace(docId);
        _attr->setTensor(docId, tensor);
        _attr->commit();
    }

    void setTensor(uint32_t docId, double v1, double v2) {
        auto t = createTensor(_typeSpec, v1, v2);
        setTensor(docId, *t);
    }
};

template <bool strict>
SimpleResult find_matches(Fixture &env, const Value &qtv, double threshold = std::numeric_limits<double>::max()) {
    auto md = MatchData::makeTestInstance(2, 2);
    auto &tfmd = *(md->resolveTermField(0));
    auto &attr = *(env._attr);

    auto dff = search::tensor::make_distance_function_factory(DistanceMetric::Euclidean, qtv.cells().type);
    auto df = dff->for_query_vector(qtv.cells());
    threshold = df->convert_threshold(threshold);
    NearestNeighborDistanceHeap dh(2);
    dh.set_distance_threshold(threshold);
    const GlobalFilter &filter = *env._global_filter;
    auto search = NearestNeighborIterator::create(strict, tfmd,
                                                  std::make_unique<DistanceCalculator>(attr, qtv),
                                                  dh, filter);
    if (strict) {
        return SimpleResult().searchStrict(*search, attr.getNumDocs());
    } else {
        return SimpleResult().search(*search, attr.getNumDocs());
    }
}

void
verify_iterator_returns_expected_results(const vespalib::string& attribute_tensor_type_spec,
                                         const vespalib::string& query_tensor_type_spec)
{
    Fixture fixture(attribute_tensor_type_spec);
    fixture.ensureSpace(6);
    fixture.setTensor(1, 3.0, 4.0);
    fixture.setTensor(2, 6.0, 8.0);
    fixture.setTensor(3, 5.0, 12.0);
    fixture.setTensor(4, 4.0, 3.0);
    fixture.setTensor(5, 8.0, 6.0);
    fixture.setTensor(6, 4.0, 3.0);
    auto nullTensor = createTensor(query_tensor_type_spec, 0.0, 0.0);
    SimpleResult result = find_matches<true>(fixture, *nullTensor);
    SimpleResult nullExpect({1,2,4,6});
    EXPECT_EQ(result, nullExpect);
    result = find_matches<false>(fixture, *nullTensor);
    EXPECT_EQ(result, nullExpect);
    auto farTensor = createTensor(query_tensor_type_spec, 9.0, 9.0);
    SimpleResult farExpect({1,2,3,5});
    result = find_matches<true>(fixture, *farTensor);
    EXPECT_EQ(result, farExpect);
    result = find_matches<false>(fixture, *farTensor);
    EXPECT_EQ(result, farExpect);

    SimpleResult null_thr5_exp({1,4,6});
    result = find_matches<true>(fixture, *nullTensor, 5.0);
    EXPECT_EQ(result, null_thr5_exp);
    result = find_matches<false>(fixture, *nullTensor, 5.0);
    EXPECT_EQ(result, null_thr5_exp);

    SimpleResult far_thr4_exp({2,5});
    result = find_matches<true>(fixture, *farTensor, 4.0);
    EXPECT_EQ(result, far_thr4_exp);
    result = find_matches<false>(fixture, *farTensor, 4.0);
    EXPECT_EQ(result, far_thr4_exp);
}

struct TestParam {
    vespalib::string attribute_tensor_type_spec;
    vespalib::string query_tensor_type_spec;
    TestParam(const vespalib::string& attribute_tensor_type_spec_in,
              const vespalib::string& query_tensor_type_spec_in) noexcept
        : attribute_tensor_type_spec(attribute_tensor_type_spec_in),
          query_tensor_type_spec(query_tensor_type_spec_in)
    {}
    TestParam(const TestParam &) noexcept;
    TestParam & operator=(TestParam &) noexcept = delete;
    TestParam(TestParam &&) noexcept = default;
    TestParam & operator=(TestParam &&) noexcept = default;
    ~TestParam();
};

TestParam::TestParam(const TestParam &) noexcept = default;
TestParam::~TestParam() = default;

std::ostream& operator<<(std::ostream& os, const TestParam& param)
{
    os << "{" << param.attribute_tensor_type_spec << ", " << param.query_tensor_type_spec << "}";
    return os;
}

struct NnsIndexIteratorParameterizedTest : public ::testing::TestWithParam<TestParam> {};

INSTANTIATE_TEST_SUITE_P(NnsTestSuite,
                         NnsIndexIteratorParameterizedTest,
                         ::testing::Values(
                                 TestParam(denseSpecDouble, denseSpecDouble),
                                 TestParam(denseSpecFloat, denseSpecFloat),
                                 TestParam(mixed_spec, denseSpecDouble)
                         ));

TEST_P(NnsIndexIteratorParameterizedTest, require_that_iterator_returns_expected_results) {
    auto param = GetParam();
    verify_iterator_returns_expected_results(param.attribute_tensor_type_spec, param.query_tensor_type_spec);
}

void
verify_iterator_returns_filtered_results(const vespalib::string& attribute_tensor_type_spec,
                                         const vespalib::string& query_tensor_type_spec)
{
    Fixture fixture(attribute_tensor_type_spec);
    fixture.ensureSpace(6);
    fixture.setFilter({1,3,4});
    fixture.setTensor(1, 3.0, 4.0);
    fixture.setTensor(2, 6.0, 8.0);
    fixture.setTensor(3, 5.0, 12.0);
    fixture.setTensor(4, 4.0, 3.0);
    fixture.setTensor(5, 8.0, 6.0);
    fixture.setTensor(6, 4.0, 3.0);
    auto nullTensor = createTensor(query_tensor_type_spec, 0.0, 0.0);
    SimpleResult result = find_matches<true>(fixture, *nullTensor);
    SimpleResult nullExpect({1,3,4});
    EXPECT_EQ(result, nullExpect);
    result = find_matches<false>(fixture, *nullTensor);
    EXPECT_EQ(result, nullExpect);
    auto farTensor = createTensor(query_tensor_type_spec, 9.0, 9.0);
    SimpleResult farExpect({1,3,4});
    result = find_matches<true>(fixture, *farTensor);
    EXPECT_EQ(result, farExpect);
    result = find_matches<false>(fixture, *farTensor);
    EXPECT_EQ(result, farExpect);
}

TEST_P(NnsIndexIteratorParameterizedTest, require_that_iterator_returns_filtered_results) {
    auto param = GetParam();
    verify_iterator_returns_filtered_results(param.attribute_tensor_type_spec, param.query_tensor_type_spec);
}

template <bool strict>
std::vector<feature_t> get_rawscores(Fixture &env, const Value &qtv) {
    auto md = MatchData::makeTestInstance(2, 2);
    auto &tfmd = *(md->resolveTermField(0));
    auto &attr = *(env._attr);
    auto dff = search::tensor::make_distance_function_factory(DistanceMetric::Euclidean, qtv.cells().type);
    NearestNeighborDistanceHeap dh(2);
    auto dummy_filter = GlobalFilter::create();
    auto search = NearestNeighborIterator::create(strict, tfmd,
                                                  std::make_unique<DistanceCalculator>(attr, qtv),
                                                  dh, *dummy_filter);
    uint32_t limit = attr.getNumDocs();
    uint32_t docid = 1;
    search->initRange(docid, limit);
    std::vector<feature_t> rv;
    while (docid < limit) {
        if (search->seek(docid)) {
            search->unpack(docid);
            rv.push_back(tfmd.getRawScore());
        }
        docid = std::max(search->getDocId(), docid + 1);
    }
    return rv;
}

void
verify_iterator_sets_expected_rawscore(const vespalib::string& attribute_tensor_type_spec,
                                       const vespalib::string& query_tensor_type_spec)
{
    Fixture fixture(attribute_tensor_type_spec);
    fixture.ensureSpace(6);
    fixture.setTensor(1, 3.0, 4.0);
    fixture.setTensor(2, 5.0, 12.0);
    fixture.setTensor(3, 6.0, 8.0);
    fixture.setTensor(4, 5.0, 12.0);
    fixture.setTensor(5, 8.0, 6.0);
    fixture.setTensor(6, 4.0, 3.0);
    auto nullTensor = createTensor(query_tensor_type_spec, 0.0, 0.0);
    std::vector<feature_t> got = get_rawscores<true>(fixture, *nullTensor);
    std::vector<feature_t> expected{5.0, 13.0, 10.0, 10.0, 5.0};
    EXPECT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(1.0/(1.0+expected[i]), got[i], EPS);
    }
    got = get_rawscores<false>(fixture, *nullTensor);
    EXPECT_EQ(got.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(1.0/(1.0+expected[i]), got[i], EPS);
    }
}

TEST_P(NnsIndexIteratorParameterizedTest, require_that_iterator_sets_expected_rawscore) {
    auto param = GetParam();
    verify_iterator_sets_expected_rawscore(param.attribute_tensor_type_spec, param.query_tensor_type_spec);
}

void expect_match(SearchIterator& itr, uint32_t docid) {
    bool match = itr.seek(docid);
    EXPECT_TRUE(match);
    EXPECT_FALSE(itr.isAtEnd());
    EXPECT_EQ(docid, itr.getDocId());
}

void expect_not_match(SearchIterator& itr, uint32_t curr_docid, uint32_t exp_next_docid) {
    bool match = itr.seek(curr_docid);
    EXPECT_FALSE(match);
    EXPECT_FALSE(itr.isAtEnd());
    EXPECT_EQ(exp_next_docid, itr.getDocId());
}

void expect_at_end(SearchIterator& itr, uint32_t docid) {
    bool match = itr.seek(docid);
    EXPECT_FALSE(match);
    EXPECT_TRUE(itr.isAtEnd());
}

TEST(NnsIndexIteratorTest, require_that_iterator_works_as_expected) {
    std::vector<NnsIndexIterator::Hit> hits{{2,4.0}, {3,9.0}, {5,1.0}, {8,16.0}, {9,36.0}};
    auto md = MatchData::makeTestInstance(2, 2);
    auto &tfmd = *(md->resolveTermField(0));
    auto dff = search::tensor::make_distance_function_factory(DistanceMetric::Euclidean, CellType::DOUBLE);
    vespalib::eval::TypedCells dummy;
    auto df = dff->for_query_vector(dummy);
    auto search = NnsIndexIterator::create(tfmd, hits, *df);
    search->initFullRange();
    expect_not_match(*search, 1, 2);
    expect_match(*search, 2);
    search->unpack(2);
    EXPECT_NEAR(1.0/(1.0+2.0), tfmd.getRawScore(), EPS);

    expect_match(*search, 3);
    search->unpack(3);
    EXPECT_NEAR(1.0/(1.0+3.0), tfmd.getRawScore(), EPS);

    expect_not_match(*search, 4, 5);
    expect_not_match(*search, 6, 8);
    search->unpack(8);
    EXPECT_NEAR(1.0/(1.0+4.0), tfmd.getRawScore(), EPS);

    expect_match(*search, 9);
    expect_at_end(*search, 10);

    search->initRange(4, 7);
    expect_not_match(*search, 4, 5);
    search->unpack(5);
    EXPECT_NEAR(1.0/(1.0+1.0), tfmd.getRawScore(), EPS);
    expect_at_end(*search, 6);
}

GTEST_MAIN_RUN_ALL_TESTS()
