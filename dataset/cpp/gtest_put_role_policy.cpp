/*
   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
   SPDX-License-Identifier: Apache-2.0
*/
/*
 * Test types are indicated by the test label ending.
 *
 * _1_ Requires credentials, permissions, and AWS resources.
 * _2_ Requires credentials and permissions.
 * _3_ Does not require credentials.
 *
 */

#include <gtest/gtest.h>
#include "iam_samples.h"
#include "iam_gtests.h"

namespace AwsDocTest {
    // NOLINTNEXTLINE(readability-named-parameter)
    TEST_F(IAM_GTests, put_role_policy_2_) {
        auto roleName = getRole();
        ASSERT_FALSE(roleName.empty()) << preconditionError() << std::endl;

        auto policyName = uuidName("policy");

        auto result = AwsDoc::IAM::putRolePolicy(roleName, policyName,
                                                 getRolePolicyJSON(), *s_clientConfig);
        ASSERT_TRUE(result);

        deleteRolePolicy(roleName, policyName);
    }
} // namespace AwsDocTest
