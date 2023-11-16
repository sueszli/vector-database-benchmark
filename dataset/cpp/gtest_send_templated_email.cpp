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
    TEST_F(SES_GTests, send_templated_email_3_) {
        MockHTTP mockHttp;
        bool result = mockHttp.addResponseWithBody("mock_input/SendTemplatedEmail.xml");
        ASSERT_TRUE(result) << preconditionError() << std::endl;
        Aws::Map<Aws::String, Aws::String> templateData;
        templateData["mock-key1"] = "mock-value1";
        templateData["mock-key2"] = "mock-value2";
        result = AwsDoc::SES::sendTemplatedEmail({"mock@email.com"}, "mock-name", templateData,
                                                 "sender@email.com", {"cc@email.com"}, "repy_to@email.com",
                                                 *s_clientConfig);
        ASSERT_TRUE(result);
    }
} // namespace AwsDocTest
/*
   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
   SPDX-License-Identifier: Apache-2.0
*/
