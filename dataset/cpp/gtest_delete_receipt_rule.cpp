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
#include "ses_samples.h"
#include "ses_gtests.h"

namespace AwsDocTest {
    // NOLINTNEXTLINE(readability-named-parameter)
    TEST_F(SES_GTests, delete_receipt_rule_3_) {
        MockHTTP mockHttp;
        bool result = mockHttp.addResponseWithBody("mock_input/DeleteReceiptRule.xml");
        ASSERT_TRUE(result) << preconditionError() << std::endl;

        result = AwsDoc::SES::deleteReceiptRule("mock-name", "mock-rule-set", *s_clientConfig);
        ASSERT_TRUE(result);
    }
} // namespace AwsDocTest
