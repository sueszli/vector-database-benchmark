/**
 * Copyright (c) 2014-present, The osquery authors
 *
 * This source code is licensed as defined by the LICENSE file found in the
 * root directory of this source tree.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR GPL-2.0-only)
 */

// Sanity check integration test for managed_policies
// Spec file: specs/darwin/managed_policies.table

#include <osquery/tests/integration/tables/helper.h>

namespace osquery {
namespace table_tests {

class managedPolicies : public testing::Test {
 protected:
  void SetUp() override {
    setUpEnvironment();
  }
};

TEST_F(managedPolicies, test_sanity) {
  // 1. Query data
  auto const data = execute_query("select * from managed_policies");
  // 2. Check size before validation
  ASSERT_GE(data.size(), 0);
  // 3. Build validation map
  // See helper.h for available flags
  // Or use custom DataCheck object
  ValidationMap row_map = {
      {"domain", NormalType},
      {"uuid", NormalType},
      {"name", NormalType},
      {"value", NormalType},
      {"username", NormalType},
      {"manual", IntType},
  };
  // 4. Perform validation
  validate_rows(data, row_map);
}

} // namespace table_tests
} // namespace osquery
