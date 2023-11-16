// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "onnx_feature.h"
#include <vespa/searchlib/fef/properties.h>
#include <vespa/searchlib/fef/onnx_model.h>
#include <vespa/searchlib/fef/featureexecutor.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/value_codec.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/util/stash.h>
#include <vespa/vespalib/util/issue.h>

#include <vespa/log/log.h>
LOG_SETUP(".features.onnx_feature");

using search::fef::Blueprint;
using search::fef::FeatureExecutor;
using search::fef::FeatureType;
using search::fef::IIndexEnvironment;
using search::fef::IQueryEnvironment;
using search::fef::ParameterList;
using vespalib::Stash;
using vespalib::eval::Value;
using vespalib::eval::ValueType;
using vespalib::eval::TensorSpec;
using vespalib::eval::FastValueBuilderFactory;
using vespalib::eval::value_from_spec;
using vespalib::make_string_short::fmt;
using vespalib::eval::Onnx;
using vespalib::Issue;

namespace search::features {

namespace {

vespalib::string normalize_name(const vespalib::string &name, const char *context) {
    vespalib::string result;
    for (char c: name) {
        if (isalnum(c)) {
            result.push_back(c);
        } else {
            result.push_back('_');
        }
    }
    if (result != name) {
        LOG(warning, "normalized %s name: '%s' -> '%s'", context, name.c_str(), result.c_str());
    }
    return result;
}

vespalib::string my_dry_run(const Onnx &model, const Onnx::WireInfo &wire) {
    vespalib::string error_msg;
    try {
        Onnx::EvalContext context(model, wire);
        std::vector<Value::UP> inputs;
        for (const auto &input_type: wire.vespa_inputs) {
            TensorSpec spec(input_type.to_spec());
            inputs.push_back(value_from_spec(spec, FastValueBuilderFactory::get()));
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            context.bind_param(i, *inputs[i]);
        }
        context.eval();
    } catch (const Ort::Exception &ex) {
        error_msg = ex.what();
    }
    return error_msg;
}

} // <unnamed>

/**
 * Feature executor that evaluates an onnx model
 */
class OnnxFeatureExecutor : public FeatureExecutor
{
private:
    Onnx::EvalContext _eval_context;
public:
    OnnxFeatureExecutor(const Onnx &model, const Onnx::WireInfo &wire_info)
        : _eval_context(model, wire_info) {}
    bool isPure() override { return true; }
    void handle_bind_outputs(vespalib::ArrayRef<fef::NumberOrObject>) override {
        for (size_t i = 0; i < _eval_context.num_results(); ++i) {
            outputs().set_object(i, _eval_context.get_result(i));
        }
    }
    void execute(uint32_t) override {
        for (size_t i = 0; i < _eval_context.num_params(); ++i) {
            _eval_context.bind_param(i, inputs().get_object(i).get());
        }
        try {
            _eval_context.eval();
        } catch (const Ort::Exception &ex) {
            Issue::report("onnx model evaluation failed: %s", ex.what());
            _eval_context.clear_results();
        }
    }
};

OnnxBlueprint::OnnxBlueprint(vespalib::stringref baseName)
    : Blueprint(baseName),
      _cache_token(),
      _debug_model(),
      _model(nullptr),
      _wire_info()
{
    assert((baseName == "onnx") || (baseName == "onnxModel"));
}

OnnxBlueprint::~OnnxBlueprint() = default;

bool
OnnxBlueprint::setup(const IIndexEnvironment &env,
                     const ParameterList &params)
{
    auto model_cfg = env.getOnnxModel(params[0].getValue());
    if (!model_cfg) {
        return fail("no model with name '%s' found", params[0].getValue().c_str());
    }
    try {
        if (env.getFeatureMotivation() == env.FeatureMotivation::VERIFY_SETUP) {
            _debug_model = std::make_unique<Onnx>(model_cfg->file_path(), Optimize::DISABLE);
            _model = _debug_model.get();
        } else {
            _cache_token = OnnxModelCache::load(model_cfg->file_path());
            _model = &(_cache_token->get());
        }
    } catch (const Ort::Exception &ex) {
        return fail("model setup failed: %s", ex.what());
    }
    Onnx::WirePlanner planner;
    for (const auto & model_input : _model->inputs()) {
        auto input_feature = model_cfg->input_feature(model_input.name);
        if (!input_feature.has_value()) {
            input_feature = fmt("rankingExpression(\"%s\")", normalize_name(model_input.name, "input").c_str());
        }
        if (auto maybe_input = defineInput(input_feature.value(), AcceptInput::OBJECT)) {
            const FeatureType &feature_input = maybe_input.value();
            assert(feature_input.is_object());
            if (!planner.bind_input_type(feature_input.type(), model_input)) {
                return fail("incompatible type for input (%s -> %s): %s -> %s",
                            input_feature.value().c_str(), model_input.name.c_str(),
                            feature_input.type().to_spec().c_str(), model_input.type_as_string().c_str());
            }
        } else {
            return fail("undefined input: %s (->%s)", input_feature.value().c_str(), model_input.name.c_str());
        }
    }
    planner.prepare_output_types(*_model);
    for (const auto & model_output : _model->outputs()) {
        auto output_name = model_cfg->output_name(model_output.name);
        if (!output_name.has_value()) {
            output_name = normalize_name(model_output.name, "output");
        }
        ValueType output_type = planner.make_output_type(model_output);
        if (output_type.is_error()) {
            return fail("unable to make compatible type for output (%s -> %s): %s -> error",
                        model_output.name.c_str(), output_name.value().c_str(),
                        model_output.type_as_string().c_str());
        }
        describeOutput(output_name.value(), "output from onnx model", FeatureType::object(output_type));
    }
    _wire_info = planner.get_wire_info(*_model);
    if (model_cfg->dry_run_on_setup()) {
        auto error_msg = my_dry_run(*_model, _wire_info);
        if (!error_msg.empty()) {
            return fail("onnx model dry-run failed: %s", error_msg.c_str());
        }
    } else {
        LOG(warning, "dry-run disabled for onnx model '%s'", model_cfg->name().c_str());
    }
    return true;
}

FeatureExecutor &
OnnxBlueprint::createExecutor(const IQueryEnvironment &, Stash &stash) const
{
    assert(_model != nullptr);
    return stash.create<OnnxFeatureExecutor>(*_model, _wire_info);
}

}
