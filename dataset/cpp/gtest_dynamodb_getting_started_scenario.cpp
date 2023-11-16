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
#include <fstream>
#include "dynamodb_gtests.h"
#include "dynamodb_samples.h"

namespace AwsDocTest {
    // Designated _2R_ because it requires resources.
    // NOLINTNEXTLINE (readability-named-parameter)
    TEST_F(DynamoDB_GTests, getting_started_scenario_2R_) {
        AddCommandLineResponses({"Jaws",
                                 "1972",
                                 "8",
                                 "Sharks",
                                 "7",
                                 "Sharks and Dude",
                                 "y",
                                 "2011",
                                 "2011",
                                 "2018",
                                 "30",
                                 "y"});

        bool result = createTableForScenario();
        ASSERT_TRUE(result);

        result = AwsDoc::DynamoDB::dynamodbGettingStartedScenario(*s_clientConfig);
        ASSERT_TRUE(result);
    }
} // AwsDocTest