// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/vespalib/testkit/testapp.h>

#include <vespa/searchcore/proton/attribute/imported_attributes_context.h>
#include <vespa/searchcore/proton/attribute/imported_attributes_repo.h>
#include <vespa/searchcore/proton/documentmetastore/documentmetastorecontext.h>
#include <vespa/searchcore/proton/bucketdb/bucket_db_owner.h>
#include <vespa/searchlib/attribute/attribute.h>
#include <vespa/searchlib/attribute/attributefactory.h>
#include <vespa/searchlib/attribute/imported_attribute_vector.h>
#include <vespa/searchlib/attribute/imported_attribute_vector_factory.h>
#include <vespa/searchlib/attribute/reference_attribute.h>
#include <vespa/searchlib/test/mock_gid_to_lid_mapping.h>
#include <vespa/searchcommon/attribute/config.h>
#include <future>

#include <vespa/log/log.h>
LOG_SETUP("imported_attributes_context_test");
using namespace proton;
using search::AttributeVector;
using search::attribute::BasicType;
using search::attribute::Config;
using search::attribute::IAttributeVector;
using search::attribute::ImportedAttributeVector;
using search::attribute::ImportedAttributeVectorFactory;
using search::attribute::ReferenceAttribute;
using search::attribute::test::MockGidToLidMapperFactory;
using generation_t = AttributeVector::generation_t;

std::shared_ptr<ReferenceAttribute>
createReferenceAttribute(const vespalib::string &name)
{
    auto refAttr = std::make_shared<ReferenceAttribute>(name);
    refAttr->setGidToLidMapperFactory(std::make_shared<MockGidToLidMapperFactory>());
    return refAttr;
}

AttributeVector::SP
createTargetAttribute(const vespalib::string &name)
{
    return search::AttributeFactory::createAttribute(name, Config(BasicType::STRING));
}

void
addDoc(AttributeVector &attr)
{
    attr.addDocs(1);
    attr.commit();
}

bool
hasActiveEnumGuards(AttributeVector &attr)
{
    return std::async(std::launch::async, [&attr] { return attr.hasActiveEnumGuards(); }).get();
}

void
assertGuards(AttributeVector &attr, generation_t expCurrentGeneration, generation_t exp_oldest_used_generation,
             bool expHasActiveEnumGuards)
{
    EXPECT_EQUAL(expCurrentGeneration, attr.getCurrentGeneration());
    EXPECT_EQUAL(exp_oldest_used_generation, attr.get_oldest_used_generation());
    EXPECT_EQUAL(expHasActiveEnumGuards, hasActiveEnumGuards(attr));
}

void
addDocAndAssertGuards(AttributeVector &attr, generation_t expCurrentGeneration, generation_t expFirstUsedGeneration, bool expHasActiveEnumGuards)
{
    addDoc(attr);
    assertGuards(attr, expCurrentGeneration, expFirstUsedGeneration, expHasActiveEnumGuards);
}

struct Fixture {
    ImportedAttributesRepo repo;
    std::unique_ptr<ImportedAttributesContext> ctx;
    Fixture()
        : repo(),
          ctx(std::make_unique<ImportedAttributesContext>(repo))
    {
    }
    Fixture &addAttribute(const vespalib::string &name) {
        auto attr = ImportedAttributeVectorFactory::create(name,
                                                           createReferenceAttribute(name + "_ref"),
                                                           std::shared_ptr<search::IDocumentMetaStoreContext>(),
                                                           createTargetAttribute(name + "_target"),
                                                           std::make_shared<const DocumentMetaStoreContext>(std::make_shared<bucketdb::BucketDBOwner>()),
                                                           false);
        repo.add(name, attr);
        return *this;
    }
    AttributeVector::SP getTargetAttribute(const vespalib::string &importedName) const {
        auto readable_target_attr = repo.get(importedName)->getTargetAttribute();
        auto target_attr = std::dynamic_pointer_cast<AttributeVector>(readable_target_attr);
        ASSERT_TRUE(target_attr);
        return target_attr;
    }
    void clearContext() {
        ctx.reset();
    }
};

TEST_F("require that attributes can be retrieved", Fixture)
{
    f.addAttribute("foo").addAttribute("bar");
    EXPECT_EQUAL("foo", f.ctx->getAttribute("foo")->getName());
    EXPECT_EQUAL("bar", f.ctx->getAttribute("bar")->getName());
    EXPECT_EQUAL("bar", f.ctx->getAttribute("bar")->getName());
    EXPECT_TRUE(f.ctx->getAttribute("not_found") == nullptr);
}

TEST_F("require that stable enum attributes can be retrieved", Fixture)
{
    f.addAttribute("foo").addAttribute("bar");
    EXPECT_EQUAL("foo", f.ctx->getAttributeStableEnum("foo")->getName());
    EXPECT_EQUAL("bar", f.ctx->getAttributeStableEnum("bar")->getName());
    EXPECT_EQUAL("bar", f.ctx->getAttributeStableEnum("bar")->getName());
    EXPECT_TRUE(f.ctx->getAttributeStableEnum("not_found") == nullptr);
}

TEST_F("require that all attributes can be retrieved", Fixture)
{
    f.addAttribute("foo").addAttribute("bar");
    std::vector<const IAttributeVector *> list;
    f.ctx->getAttributeList(list);
    EXPECT_EQUAL(2u, list.size());
    // Don't depend on internal (unspecified) ordering
    std::sort(list.begin(), list.end(), [](auto* lhs, auto* rhs){
        return lhs->getName() < rhs->getName();
    });
    EXPECT_EQUAL("bar", list[0]->getName());
    EXPECT_EQUAL("foo", list[1]->getName());
}

TEST_F("require that guards are cached", Fixture)
{
    f.addAttribute("foo");
    auto targetAttr = f.getTargetAttribute("foo");
    TEST_DO(addDocAndAssertGuards(*targetAttr, 2, 2, false));

    f.ctx->getAttribute("foo"); // guard is taken and cached
    TEST_DO(addDocAndAssertGuards(*targetAttr, 4, 2, false));

    f.clearContext(); // guard is released
    TEST_DO(addDocAndAssertGuards(*targetAttr, 6, 6, false));
}

TEST_F("require that stable enum guards are cached", Fixture)
{
    f.addAttribute("foo");
    auto targetAttr = f.getTargetAttribute("foo");
    TEST_DO(addDocAndAssertGuards(*targetAttr, 2, 2, false));

    f.ctx->getAttributeStableEnum("foo"); // enum guard is taken and cached
    TEST_DO(addDocAndAssertGuards(*targetAttr, 4, 2, true));

    f.clearContext(); // guard is released
    TEST_DO(addDocAndAssertGuards(*targetAttr, 6, 6, false));
}

TEST_F("require that stable enum guards can be released", Fixture)
{
    f.addAttribute("foo");
    auto targetAttr = f.getTargetAttribute("foo");
    TEST_DO(addDocAndAssertGuards(*targetAttr, 2, 2, false));

    f.ctx->getAttributeStableEnum("foo"); // enum guard is taken and cached
    TEST_DO(addDocAndAssertGuards(*targetAttr, 4, 2, true));

    f.ctx->releaseEnumGuards();
    TEST_DO(addDocAndAssertGuards(*targetAttr, 6, 6, false));
}

TEST_MAIN()
{
    TEST_RUN_ALL();
}
