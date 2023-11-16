// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "tensor_from_labels_feature.h"
#include "array_parser.hpp"
#include "constant_tensor_executor.h"
#include "tensor_from_attribute_executor.h"
#include <vespa/searchlib/fef/properties.h>
#include <vespa/searchlib/fef/feature_type.h>
#include <vespa/searchcommon/attribute/attributecontent.h>
#include <vespa/searchcommon/attribute/iattributevector.h>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/value_type.h>
#include <vespa/vespalib/util/issue.h>

#include <vespa/log/log.h>
LOG_SETUP(".features.tensor_from_labels_feature");

using namespace search::fef;
using search::attribute::IAttributeVector;
using search::attribute::WeightedConstCharContent;
using search::attribute::WeightedStringContent;
using vespalib::eval::FastValueBuilderFactory;
using vespalib::eval::ValueType;
using vespalib::eval::CellType;
using vespalib::Issue;
using search::fef::FeatureType;

namespace search {
namespace features {

TensorFromLabelsBlueprint::TensorFromLabelsBlueprint()
    : TensorFactoryBlueprint("tensorFromLabels")
{
}

bool
TensorFromLabelsBlueprint::setup(const search::fef::IIndexEnvironment &env,
                                 const search::fef::ParameterList &params)
{
    (void) env;
    // _params[0] = source ('attribute(name)' OR 'query(param)');
    // _params[1] = dimension (optional);
    bool validSource = extractSource(params[0].getValue());
    if (! validSource) {
        return fail("invalid source: '%s'", params[0].getValue().c_str());
    }
    if (params.size() == 2) {
        _dimension = params[1].getValue();
    } else {
        _dimension = _sourceParam;
    }
    auto vt = ValueType::make_type(CellType::DOUBLE, {{_dimension}});
    _valueType = ValueType::from_spec(vt.to_spec());
    if (_valueType.is_error()) {
        return fail("invalid dimension name: '%s'", _dimension.c_str());
    }
    describeOutput("tensor",
                   "The tensor created from the given source (attribute field or query parameter)",
                   FeatureType::object(_valueType));
    return true;
}

namespace {

FeatureExecutor &
createAttributeExecutor(const search::fef::IQueryEnvironment &env,
                        const vespalib::string &attrName,
                        const ValueType &valueType,
                        vespalib::Stash &stash)
{
    const IAttributeVector *attribute = env.getAttributeContext().getAttribute(attrName);
    if (attribute == NULL) {
        Issue::report("tensor_from_labels feature: The attribute vector '%s' was not found."
                      " Returning empty tensor.", attrName.c_str());
        return ConstantTensorExecutor::createEmpty(valueType, stash);
    }
    if (attribute->isFloatingPointType()) {
        Issue::report("tensor_from_labels feature: The attribute vector '%s' must have basic type string or integer."
                      " Returning empty tensor.", attrName.c_str());
        return ConstantTensorExecutor::createEmpty(valueType, stash);
    }
    if (attribute->getCollectionType() == search::attribute::CollectionType::WSET) {
        Issue::report("tensor_from_labels feature: The attribute vector '%s' is a weighted set - use tensorFromWeightedSet instead."
                      " Returning empty tensor.", attrName.c_str());
        return ConstantTensorExecutor::createEmpty(valueType, stash);
    }
    // Note that for array attribute vectors the default weight is 1.0 for all values.
    // This means we can get the attribute content as weighted content and build
    // the tensor the same way as with weighted set attributes in tensorFromWeightedSet.
    if (attribute->isIntegerType()) {
        // Using WeightedStringContent ensures that the integer values are converted
        // to strings while extracting them from the attribute.
        return stash.create<TensorFromAttributeExecutor<WeightedStringContent>>(attribute, valueType);
    }
    // When the underlying attribute is of type string we can reference these values
    // using WeightedConstCharContent.
    return stash.create<TensorFromAttributeExecutor<WeightedConstCharContent>>(attribute, valueType);
}

FeatureExecutor &
createQueryExecutor(const search::fef::IQueryEnvironment &env,
                    const vespalib::string &queryKey,
                    const ValueType &valueType,
                    vespalib::Stash &stash)
{
    search::fef::Property prop = env.getProperties().lookup(queryKey);
    if (prop.found() && !prop.get().empty()) {
        std::vector<vespalib::string> vector;
        ArrayParser::parse(prop.get(), vector);
        auto factory = FastValueBuilderFactory::get();
        auto builder = factory.create_value_builder<double>(valueType, 1, 1, vector.size());
        std::vector<vespalib::stringref> addr_ref;
        for (const auto &elem : vector) {
            addr_ref.clear();
            addr_ref.push_back(elem);
            auto cell_array = builder->add_subspace(addr_ref);
            cell_array[0] = 1.0;
        }
        return ConstantTensorExecutor::create(builder->build(std::move(builder)), stash);
    }
    return ConstantTensorExecutor::createEmpty(valueType, stash);
}

}

FeatureExecutor &
TensorFromLabelsBlueprint::createExecutor(const search::fef::IQueryEnvironment &env, vespalib::Stash &stash) const
{
    if (_sourceType == ATTRIBUTE_SOURCE) {
        return createAttributeExecutor(env, _sourceParam, _valueType, stash);
    } else if (_sourceType == QUERY_SOURCE) {
        return createQueryExecutor(env, _sourceParam, _valueType, stash);
    }
    return ConstantTensorExecutor::createEmpty(_valueType, stash);
}

} // namespace features
} // namespace search
