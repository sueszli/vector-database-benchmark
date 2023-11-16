// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#include <vespa/vespalib/testkit/test_kit.h>
#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/function.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/tensor_function.h>
#include <vespa/eval/eval/operation.h>
#include <vespa/eval/eval/interpreted_function.h>
#include <vespa/eval/eval/compile_tensor_function.h>
#include <vespa/eval/eval/test/eval_spec.h>
#include <vespa/eval/eval/basic_nodes.h>
#include <vespa/eval/eval/simple_value.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/util/stash.h>
#include <vespa/vespalib/test/insertion_operators.h>
#include <iostream>

using namespace vespalib::eval;
using vespalib::Stash;

//-----------------------------------------------------------------------------

struct MyEvalTest : test::EvalSpec::EvalTest {
    size_t pass_cnt = 0;
    size_t fail_cnt = 0;
    bool print_pass = false;
    bool print_fail = false;

    ~MyEvalTest() override;

    virtual void next_expression(const std::vector<vespalib::string> &param_names,
                                 const vespalib::string &expression) override
    {
        auto function = Function::parse(param_names, expression);
        ASSERT_TRUE(!function->has_error());
        bool is_supported = true;
        bool has_issues = InterpretedFunction::detect_issues(*function);
        if (is_supported == has_issues) {
            const char *supported_str = is_supported ? "supported" : "not supported";
            const char *issues_str = has_issues ? "has issues" : "does not have issues";
            print_fail && fprintf(stderr, "expression %s is %s, but %s\n",
                                  expression.c_str(), supported_str, issues_str);
            ++fail_cnt;
        }
    }

    virtual void handle_case(const std::vector<vespalib::string> &param_names,
                             const std::vector<double> &param_values,
                             const vespalib::string &expression,
                             double expected_result) override
    {
        auto function = Function::parse(param_names, expression);
        ASSERT_TRUE(!function->has_error());
        bool is_supported = true;
        bool has_issues = InterpretedFunction::detect_issues(*function);
        if (is_supported && !has_issues) {
            vespalib::string desc = as_string(param_names, param_values, expression);
            SimpleParams params(param_values);
            verify_result(SimpleValueBuilderFactory::get(), *function, "[simple] "+desc, params, expected_result);
            verify_result(FastValueBuilderFactory::get(),   *function, "[prod]   "+desc, params, expected_result);
        }
    }

    void report_result(bool is_double, double result, double expect, const vespalib::string &desc)
    {
        if (is_double && is_same(expect, result)) {
            print_pass && fprintf(stderr, "verifying: %s -> %g ... PASS\n",
                                  desc.c_str(), expect);
            ++pass_cnt;
        } else {
            print_fail && fprintf(stderr, "verifying: %s -> %g ... FAIL: got %g\n",
                                  desc.c_str(), expect, result);
            ++fail_cnt;
        }
    }

    void verify_result(const ValueBuilderFactory &factory,
                       const Function &function,
                       const vespalib::string &description,
                       const SimpleParams &params,
                       double expected_result)
    {
        auto node_types = NodeTypes(function, std::vector<ValueType>(params.params.size(), ValueType::double_type()));
        InterpretedFunction ifun(factory, function, node_types);
        InterpretedFunction::Context ictx(ifun);
        const Value &result_value = ifun.eval(ictx, params);
        report_result(result_value.type().is_double(), result_value.as_double(), expected_result, description);
    }
};

MyEvalTest::~MyEvalTest() = default;

TEST_FF("require that interpreted evaluation passes all conformance tests", MyEvalTest(), test::EvalSpec()) {
    f1.print_fail = true;
    f2.add_all_cases();
    f2.each_case(f1);
    EXPECT_GREATER(f1.pass_cnt, 1000u);
    EXPECT_EQUAL(0u, f1.fail_cnt);
}

//-----------------------------------------------------------------------------

TEST("require that invalid function is tagged with error") {
    std::vector<vespalib::string> params({"x", "y", "z", "w"});
    auto function = Function::parse(params, "x & y");
    EXPECT_TRUE(function->has_error());
}

//-----------------------------------------------------------------------------

size_t count_ifs(const vespalib::string &expr, std::initializer_list<double> params_in) {
    auto fun = Function::parse(expr);
    auto node_types = NodeTypes(*fun, std::vector<ValueType>(params_in.size(), ValueType::double_type()));
    InterpretedFunction ifun(SimpleValueBuilderFactory::get(), *fun, node_types);
    InterpretedFunction::Context ctx(ifun);
    SimpleParams params(params_in);
    ifun.eval(ctx, params);
    return ctx.if_cnt();
}

TEST("require that if_cnt in eval context is updated correctly") {
    EXPECT_EQUAL(0u, count_ifs("1", {}));
    EXPECT_EQUAL(1u, count_ifs("if(a<10,if(a<9,if(a<8,if(a<7,5,4),3),2),1)", {10}));
    EXPECT_EQUAL(2u, count_ifs("if(a<10,if(a<9,if(a<8,if(a<7,5,4),3),2),1)", {9}));
    EXPECT_EQUAL(3u, count_ifs("if(a<10,if(a<9,if(a<8,if(a<7,5,4),3),2),1)", {8}));
    EXPECT_EQUAL(4u, count_ifs("if(a<10,if(a<9,if(a<8,if(a<7,5,4),3),2),1)", {7}));
    EXPECT_EQUAL(4u, count_ifs("if(a<10,if(a<9,if(a<8,if(a<7,5,4),3),2),1)", {6}));
}

//-----------------------------------------------------------------------------

TEST("require that interpreted function instructions have expected size") {
    EXPECT_EQUAL(sizeof(InterpretedFunction::Instruction), 16u);
}

TEST("require that function pointers can be passed as instruction parameters") {
    EXPECT_EQUAL(sizeof(&operation::Add::f), sizeof(uint64_t));
}

TEST("require that basic addition works") {
    auto function = Function::parse("a+10");
    auto node_types = NodeTypes(*function, {ValueType::double_type()});
    InterpretedFunction interpreted(SimpleValueBuilderFactory::get(), *function, node_types);
    InterpretedFunction::Context ctx(interpreted);
    SimpleParams params_20({20});
    SimpleParams params_40({40});
    EXPECT_EQUAL(interpreted.eval(ctx, params_20).as_double(), 30.0);
    EXPECT_EQUAL(interpreted.eval(ctx, params_40).as_double(), 50.0);
}

//-----------------------------------------------------------------------------

TEST("require that functions with non-compilable simple lambdas cannot be interpreted") {
    auto good_map = Function::parse("map(a,f(x)(x+1))");
    auto good_join = Function::parse("join(a,b,f(x,y)(x+y))");
    auto good_merge = Function::parse("merge(a,b,f(x,y)(x+y))");
    auto bad_map = Function::parse("map(a,f(x)(map(x,f(i)(i+1))))");
    auto bad_join = Function::parse("join(a,b,f(x,y)(join(x,y,f(i,j)(i+j))))");
    auto bad_merge = Function::parse("merge(a,b,f(x,y)(join(x,y,f(i,j)(i+j))))");
    for (const Function *good: {good_map.get(), good_join.get(), good_merge.get()}) {
        if (!EXPECT_TRUE(!good->has_error())) {
            fprintf(stderr, "parse error: %s\n", good->get_error().c_str());
        }
        EXPECT_TRUE(!InterpretedFunction::detect_issues(*good));
    }
    for (const Function *bad: {bad_map.get(), bad_join.get(), bad_merge.get()}) {
        if (!EXPECT_TRUE(!bad->has_error())) {
            fprintf(stderr, "parse error: %s\n", bad->get_error().c_str());
        }
        EXPECT_TRUE(InterpretedFunction::detect_issues(*bad));
    }
    std::cerr << "Example function issues:" << std::endl
              << InterpretedFunction::detect_issues(*bad_join).list
              << std::endl;
}

TEST("require that functions with non-interpretable complex lambdas cannot be interpreted") {
    auto good_tensor_lambda = Function::parse("tensor(x[5])(map(x,f(y)(y)))");
    auto good_map_subspaces = Function::parse("map_subspaces(a,f(x)(concat(x,x,y)))");
    auto bad_tensor_lambda = Function::parse("tensor(x[5])(map(x,f(y)(map(y,f(i)(i+1)))))");
    auto bad_map_subspaces = Function::parse("map_subspaces(a,f(x)(map(x,f(y)(map(y,f(i)(i+1))))))");
    for (const Function *good: {good_tensor_lambda.get(), good_map_subspaces.get()}) {
        if (!EXPECT_TRUE(!good->has_error())) {
            fprintf(stderr, "parse error: %s\n", good->get_error().c_str());
        }
        EXPECT_TRUE(!InterpretedFunction::detect_issues(*good));
    }
    for (const Function *bad: {bad_tensor_lambda.get(), bad_map_subspaces.get()}) {
        if (!EXPECT_TRUE(!bad->has_error())) {
            fprintf(stderr, "parse error: %s\n", bad->get_error().c_str());
        }
        EXPECT_TRUE(InterpretedFunction::detect_issues(*bad));
    }
    std::cerr << "Example function issues:" << std::endl
              << InterpretedFunction::detect_issues(*bad_map_subspaces).list
              << std::endl;
}

//-----------------------------------------------------------------------------

TEST("require that compilation meta-data can be collected") {
    Stash stash;
    const auto &x2 = tensor_function::inject(ValueType::from_spec("tensor(x[2])"), 0, stash);
    const auto &x3 = tensor_function::inject(ValueType::from_spec("tensor(x[3])"), 1, stash);
    const auto &concat_x5 = tensor_function::concat(x3, x2, "x", stash);
    const auto &x5 = tensor_function::inject(ValueType::from_spec("tensor(x[5])"), 2, stash);
    const auto &mapped_x5 = tensor_function::map(x5, operation::Relu::f, stash);
    const auto &flag = tensor_function::inject(ValueType::from_spec("double"), 0, stash);
    const auto &root = tensor_function::if_node(flag, concat_x5, mapped_x5, stash);
    CTFMetaData meta;
    InterpretedFunction ifun(FastValueBuilderFactory::get(), root, &meta);
    fprintf(stderr, "compilation meta-data:\n");
    for (const auto &step: meta.steps) {
        fprintf(stderr, "  %s -> %s\n", step.class_name.c_str(), step.symbol_name.c_str());        
    }
}

//-----------------------------------------------------------------------------

TEST_MAIN() { TEST_RUN_ALL(); }
