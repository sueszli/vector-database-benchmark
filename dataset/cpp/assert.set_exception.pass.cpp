//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, no-threads
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCUDACXX_ENABLE_ASSERTIONS=1
#define _LIBCUDACXX_ENABLE_ASSERTIONS 1

// <future>

// class promise<R>

// void set_exception(exception_ptr p);
// Test that a null exception_ptr is diagnosed.

#include <future>
#include <exception>

#include "check_assertion.h"

int main(int, char**) {
    {
        typedef int T;
        std::promise<T> p;
        TEST_LIBCUDACXX_ASSERT_FAILURE(p.set_exception(std::exception_ptr()), "promise::set_exception: received nullptr");
    }

    {
        typedef int& T;
        std::promise<T> p;
        TEST_LIBCUDACXX_ASSERT_FAILURE(p.set_exception(std::exception_ptr()), "promise::set_exception: received nullptr");
    }

    return 0;
}
