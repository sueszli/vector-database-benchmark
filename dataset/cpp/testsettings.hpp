/*************************************************************************
 *
 * Copyright 2016 Realm Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************/

#ifndef TESTSETTINGS_H
#define TESTSETTINGS_H

#ifndef TEST_DURATION
#define TEST_DURATION 0 // Only brief unit tests. < 1 sec
//#  define TEST_DURATION 1  // All unit tests, plus monkey tests. ~1 minute
//#  define TEST_DURATION 2  // Same as 2, but longer monkey tests. 8 minutes
//#  define TEST_DURATION 3
#endif

// Some threading robustness tests are not enable by default, because
// they interfere badly with Valgrind.
#define TEST_THREAD_ROBUSTNESS 0

// Wrap pthread function calls with the pthread bug finding tool (program execution will be slower) by
// #including pthread_test.h. Works both in debug and release mode.
//#define REALM_PTHREADS_TEST

#define TEST_BASIC_UTILS
#define TEST_BPLUS_TREE
#define TEST_COLUMN_MIXED
#define TEST_ALLOC
#define TEST_ARRAY
#define TEST_ARRAY_BINARY
#define TEST_ARRAY_BLOB
#define TEST_ARRAY_FLOAT
#define TEST_ARRAY_MIXED
#define TEST_ARRAY_STRING
#define TEST_ARRAY_STRING_LONG
#define TEST_COLUMN
#define TEST_COLUMN_BASIC
#define TEST_COLUMN_BINARY
#define TEST_COLUMN_TIMESTAMP
#define TEST_COLUMN_FLOAT
#define TEST_COLUMN_MIXED
#define TEST_COLUMN_STRING
#define TEST_FILE
#define TEST_FILE_LOCKS
#define TEST_GEO
#define TEST_GROUP
#define TEST_UPGRADE
#define TEST_INDEX_STRING
#define TEST_LANG_BIND_HELPER
#define TEST_PARSER
#define TEST_QUERY
#define TEST_SHARED
#define TEST_STRING_DATA
#define TEST_BINARY_DATA
#define TEST_TABLE
#define TEST_TABLE_VIEW
#define TEST_LINK_VIEW
#define TEST_THREAD
#define TEST_TRANSACTIONS
#define TEST_TRANSACTIONS_LASSE
#define TEST_REPLICATION
#define TEST_UTF8
#define TEST_COLUMN_LARGE
#define TEST_JSON
#define TEST_LINKS
#define TEST_ENCRYPTED_FILE_MAPPING
#define TEST_DESTRUCTOR_THREAD_SAFETY

#define TEST_UTIL_ANY
#define TEST_UTIL_BASE64
#define TEST_UTIL_ERROR
#define TEST_UTIL_FLAT_MAP
#define TEST_UTIL_INSPECT
#define TEST_UTIL_FILE
#define TEST_UTIL_FUTURE
#define TEST_UTIL_URI
#define TEST_UTIL_TO_STRING
#define TEST_UTIL_TYPE_LIST
#define TEST_UTIL_FIXED_SIZE_BUFFER
#define TEST_UTIL_FUNCTIONAL
#define TEST_UTIL_FROM_CHARS

#ifndef _WIN32
#define TEST_UTIL_NETWORK
#endif

// Takes a long time. Also currently fails to reproduce the Java bug, but once it has been identified, this
// test could perhaps be modified to trigger it (unless it's a language binding problem).
//#define JAVA_MANY_COLUMNS_CRASH

// Temporarily disable async testing until use of sleep() in the async tests have
// been replaced with a better solution.
#define DISABLE_ASYNC

#endif
