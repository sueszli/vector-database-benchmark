/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

int arith_divide_by_zero() {
  int x = 0;
  int y = 5;
  return y / x;
}
