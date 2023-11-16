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
#include "awsdoc/s3/s3_examples.h"
#include "S3_GTests.h"

static const int BUCKETS_NEEDED = 1;

namespace AwsDocTest {
// NOLINTNEXTLINE(readability-named-parameter)
    TEST_F(S3_GTests, put_bucket_policy_2_) {
        std::vector<Aws::String> bucketNames = GetCachedS3Buckets(BUCKETS_NEEDED);
        ASSERT_GE(bucketNames.size(), BUCKETS_NEEDED)
                                    << "Unable to create bucket as precondition for test" << std::endl;

        Aws::String policyString = GetBucketPolicy(bucketNames[0]);
        ASSERT_TRUE(!policyString.empty()) << "Unable to add policy to bucket as precondition for test" << std::endl;

        bool result = AwsDoc::S3::PutBucketPolicy(bucketNames[0], policyString, *s_clientConfig);
        ASSERT_TRUE(result);
    }
} // namespace AwsDocTest
