// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/eval/fast_value.h>
#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/instruction/sparse_112_dot_product.h>
#include <vespa/eval/eval/test/eval_fixture.h>
#include <vespa/eval/eval/test/gen_spec.h>
#include <vespa/vespalib/util/stringfmt.h>
#include <vespa/vespalib/gtest/gtest.h>

using namespace vespalib::eval;
using namespace vespalib::eval::test;

using vespalib::make_string_short::fmt;

//-----------------------------------------------------------------------------

struct FunInfo {
    using LookFor = Sparse112DotProduct;
    void verify(const LookFor &fun) const {
        EXPECT_TRUE(fun.result_is_mutable());
    }
};

void verify_optimized_cell_types(const vespalib::string &expr)
{
    CellTypeSpace types(CellTypeUtils::list_types(), 3);
    EvalFixture::verify<FunInfo>(expr, {FunInfo()}, CellTypeSpace(types).same());
    EvalFixture::verify<FunInfo>(expr, {}, CellTypeSpace(types).different());
}

void verify_optimized(const vespalib::string &expr, size_t num_params = 3)
{
    CellTypeSpace just_float({CellType::FLOAT}, num_params);
    EvalFixture::verify<FunInfo>(expr, {FunInfo()}, just_float);
}

void verify_not_optimized(const vespalib::string &expr) {
    CellTypeSpace just_double({CellType::DOUBLE}, 3);
    EvalFixture::verify<FunInfo>(expr, {}, just_double);
}

//-----------------------------------------------------------------------------

TEST(Sparse112DotProduct, expression_can_be_optimized)
{
    verify_optimized_cell_types("reduce(x5_2*y4_2*x5_1y4_1,sum)");
}

TEST(Sparse112DotProduct, different_input_placement_is_handled)
{
    std::array<vespalib::string,3> params = {"x3_1", "y3_1", "x3_1y3_1"};
    for (size_t p1 = 0; p1 < params.size(); ++p1) {
        for (size_t p2 = 0; p2 < params.size(); ++p2) {
            for (size_t p3 = 0; p3 < params.size(); ++p3) {
                if ((p1 != p2) && (p1 != p3) && (p2 != p3)) {
                    verify_optimized(fmt("reduce((%s*%s)*%s,sum)", params[p1].c_str(), params[p2].c_str(), params[p3].c_str()));
                    verify_optimized(fmt("reduce(%s*(%s*%s),sum)", params[p1].c_str(), params[p2].c_str(), params[p3].c_str()));
                }
            }
        }
    }
}

TEST(Sparse112DotProduct, expression_can_be_optimized_with_extra_tensors)
{
    verify_optimized("reduce((x5_2*y4_2)*(x5_1y4_1*x3_1),sum)", 4);
    verify_optimized("reduce((x5_2*x3_1)*(y4_2*x5_1y4_1),sum)", 4);
}

TEST(Sparse112DotProduct, similar_expressions_are_not_optimized)
{
    verify_not_optimized("reduce(x5_2*y4_2*x5_1y4_1,prod)");
    verify_not_optimized("reduce(x5_2+y4_2*x5_1y4_1,sum)");
    verify_not_optimized("reduce(x5_2*y4_2+x5_1y4_1,sum)");
    verify_not_optimized("reduce(x5_2*z4_2*x5_1y4_1,sum)");
    verify_not_optimized("reduce(x5_2*y4_2*x5_1z4_1,sum)");
    verify_not_optimized("reduce(x5_2*x1_1y4_2*x5_1y4_1,sum)");
    verify_not_optimized("reduce(x5_2*y4_2*x5_1,sum)");
    verify_not_optimized("reduce(x5*y4*x5y4,sum)");
    verify_not_optimized("reduce(x5*y4_1*x5y4_1,sum)");
}

//-----------------------------------------------------------------------------

GTEST_MAIN_RUN_ALL_TESTS()
