// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/document/bucket/bucketselector.h>

#include <vespa/document/bucket/bucketid.h>
#include <vespa/document/select/parser.h>
#include <vespa/document/base/testdocrepo.h>
#include <vespa/vespalib/test/insertion_operators.h>
#include <algorithm>
#include <iostream>
#include <gtest/gtest.h>

using document::select::Node;
using document::select::Parser;

namespace document {

#define ASSERT_BUCKET_COUNT(expression, count) \
{ \
    TestDocRepo testRepo; \
    BucketIdFactory idfactory; \
    BucketSelector selector(idfactory); \
    Parser parser(testRepo.getTypeRepo(), idfactory); \
    std::unique_ptr<Node> node(parser.parse(expression)); \
    ASSERT_TRUE(node.get() != 0);                       \
    std::unique_ptr<BucketSelector::BucketVector> buckets( \
            selector.select(*node)); \
    size_t bcount(buckets.get() ? buckets->size() : 0); \
    std::ostringstream ost; \
    ost << "Expression " << expression << " did not contain " << count \
        << " buckets as expected"; \
    EXPECT_EQ((size_t) count, bcount) << ost.str(); \
}

#define ASSERT_BUCKET(expression, bucket) \
{ \
    TestDocRepo testRepo; \
    BucketIdFactory idfactory; \
    BucketSelector selector(idfactory); \
    Parser parser(testRepo.getTypeRepo(), idfactory); \
    std::unique_ptr<Node> node(parser.parse(expression)); \
    ASSERT_TRUE(node.get() != 0);           \
    std::unique_ptr<BucketSelector::BucketVector> buckets( \
            selector.select(*node)); \
    std::ostringstream ost; \
    ost << "Expression " << expression << " did not contain bucket " \
        << bucket.toString(); \
    if (buckets.get()) { \
        ost << ". Buckets: " << std::hex << *buckets; \
    } else { \
        ost << ". Matches all buckets"; \
    } \
    EXPECT_TRUE(buckets.get() &&     \
                std::find(buckets->begin(), buckets->end(),     \
                          bucket) != buckets->end()) << ost.str();      \
}

TEST(BucketSelectorTest, testSimple)
{
    ASSERT_BUCKET_COUNT("id = \"id:ns:testdoc:n=123:foobar\"", 1u);
    ASSERT_BUCKET_COUNT("id = \"id:ns:testdoc:n=123:foo*\"", 0u);
    ASSERT_BUCKET_COUNT("id == \"id:ns:testdoc:n=123:f?oo*\"", 1u);
    ASSERT_BUCKET_COUNT("id =~ \"id:ns:testdoc:n=123:foo*\"", 0u);
    ASSERT_BUCKET_COUNT("id =~ \"id:ns:testdoc:n=123:foo?\"", 0u);
    ASSERT_BUCKET_COUNT("id.user = 123", 1u);
    ASSERT_BUCKET_COUNT("id.user == 123", 1u);
    ASSERT_BUCKET_COUNT("id.group = \"yahoo.com\"", 1u);
    ASSERT_BUCKET_COUNT("id.group = \"yahoo.com\" or id.user=123", 2u);
    ASSERT_BUCKET_COUNT("id.group = \"yahoo.com\" and id.user=123", 0u);
    ASSERT_BUCKET_COUNT(
            "id.group = \"yahoo.com\" and testdoctype1.hstringval=\"Doe\"", 1u);
    ASSERT_BUCKET_COUNT("not id.group = \"yahoo.com\"", 0u);
    ASSERT_BUCKET_COUNT("id.group != \"yahoo.com\"", 0u);
    ASSERT_BUCKET_COUNT("id.group <= \"yahoo.com\"", 0u);

    ASSERT_BUCKET_COUNT("id.bucket = 0x4000000000003018", 1u); // Bucket 12312
    ASSERT_BUCKET_COUNT("id.bucket == 0x4000000000000258", 1u); // Bucket 600
    ASSERT_BUCKET_COUNT("(testdoctype1 and id.bucket=0)", 1u);

        // Check that the correct buckets is found
    ASSERT_BUCKET("id = \"id:ns:testdoc:n=123:foobar\"",
                  document::BucketId(58, 123));

    ASSERT_BUCKET("id.bucket == 0x4000000000000258", document::BucketId(16, 600));

    ASSERT_BUCKET("id.user = 123", document::BucketId(32, 123));
    ASSERT_BUCKET("id.group = \"yahoo.com\"",
                  document::BucketId(32, 0x9a1acd50));
}

} // document
