/**
 * Copyright (c) 2014-present, The osquery authors
 *
 * This source code is licensed as defined by the LICENSE file found in the
 * root directory of this source tree.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR GPL-2.0-only)
 */

// Sanity check integration test for ec2_instance_metadata
// Spec file: specs/linux/ec2_instance_metadata.table

#include <osquery/tests/integration/tables/helper.h>

namespace osquery {

DECLARE_uint32(aws_imdsv2_request_interval);
DECLARE_uint32(aws_imdsv2_request_attempts);

namespace table_tests {

class ec2InstanceMetadata : public testing::Test {
 protected:
  void SetUp() override {
    setUpEnvironment();
  }
};

TEST_F(ec2InstanceMetadata, test_sanity) {
  // Speed up the querying in the case we're not on EC2
  FLAGS_aws_imdsv2_request_interval = 1;
  FLAGS_aws_imdsv2_request_attempts = 1;

  // 1. Query data
  auto const data = execute_query("select * from ec2_instance_metadata");
  // 2. Check size before validation
  // ASSERT_GE(data.size(), 0ul);
  // ASSERT_EQ(data.size(), 1ul);
  // ASSERT_EQ(data.size(), 0ul);
  // 3. Build validation map
  // See helper.h for available flags
  // Or use custom DataCheck object
  // ValidationMap row_map = {
  //      {"instance_id", NormalType}
  //      {"instance_type", NormalType}
  //      {"architecture", NormalType}
  //      {"region", NormalType}
  //      {"availability_zone", NormalType}
  //      {"local_hostname", NormalType}
  //      {"local_ipv4", NormalType}
  //      {"mac", NormalType}
  //      {"security_groups", NormalType}
  //      {"iam_arn", NormalType}
  //      {"ami_id", NormalType}
  //      {"reservation_id", NormalType}
  //      {"account_id", NormalType}
  //      {"ssh_public_key", NormalType}
  //}
  // 4. Perform validation
  // validate_rows(data, row_map);
}

} // namespace table_tests
} // namespace osquery
