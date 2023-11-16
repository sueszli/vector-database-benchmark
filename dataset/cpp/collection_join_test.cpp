#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <collection_manager.h>
#include "collection.h"

class CollectionJoinTest : public ::testing::Test {
protected:
    Store *store;
    CollectionManager & collectionManager = CollectionManager::get_instance();
    std::atomic<bool> quit = false;

    std::vector<std::string> query_fields;
    std::vector<sort_by> sort_fields;

    void setupCollection() {
        std::string state_dir_path = "/tmp/typesense_test/collection_join";
        LOG(INFO) << "Truncating and creating: " << state_dir_path;
        system(("rm -rf "+state_dir_path+" && mkdir -p "+state_dir_path).c_str());

        store = new Store(state_dir_path);
        collectionManager.init(store, 1.0, "auth_key", quit);
        collectionManager.load(8, 1000);
    }

    virtual void SetUp() {
        setupCollection();
    }

    virtual void TearDown() {
        collectionManager.dispose();
        delete store;
    }
};

TEST_F(CollectionJoinTest, SchemaReferenceField) {
    nlohmann::json schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_.*", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;

    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Wildcard field cannot have a reference.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": ".*", "type": "auto", "reference": "Products.product_id"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Field `.*` cannot be a reference field.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": 123},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Reference should be a string.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": "foo"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;

    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_FALSE(collection_create_op.ok());
    ASSERT_EQ("Invalid reference `foo`.", collection_create_op.error());

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto collection = collection_create_op.get();
    auto schema = collection->get_schema();

    ASSERT_EQ(schema.count("customer_name"), 1);
    ASSERT_TRUE(schema.at("customer_name").reference.empty());
    ASSERT_EQ(schema.count("product_id"), 1);
    ASSERT_FALSE(schema.at("product_id").reference.empty());

    auto reference_fields = collection->get_reference_fields();
    ASSERT_EQ(reference_fields.count("product_id"), 1);
    ASSERT_EQ(reference_fields.at("product_id").collection, "Products");
    ASSERT_EQ(reference_fields.at("product_id").field, "product_id");

    // Add a `foo_sequence_id` field in the schema for `foo` reference field.
    ASSERT_EQ(schema.count("product_id_sequence_id"), 1);
    ASSERT_TRUE(schema.at("product_id_sequence_id").index);

    collectionManager.drop_collection("Customers");
}

TEST_F(CollectionJoinTest, IndexDocumentHavingReferenceField) {
    auto customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "products.product_id"}
                ]
            })"_json;
    auto collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    auto customer_collection = collection_create_op.get();

    nlohmann::json customer_json = R"({
                                        "customer_id": "customer_a",
                                        "customer_name": "Joe",
                                        "product_price": 143
                                    })"_json;
    auto add_doc_op = customer_collection->add(customer_json.dump());

    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Missing the required reference field `reference_id` in the document.", add_doc_op.error());

    customer_json = R"({
                        "customer_id": "customer_a",
                        "customer_name": "Joe",
                        "product_price": 143,
                        "reference_id": "a"
                    })"_json;
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced collection `products` not found.", add_doc_op.error());
    collectionManager.drop_collection("Customers");

    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.id"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    customer_collection = collection_create_op.get();

    auto products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string", "index": false, "optional": true},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced field `id` not found in the collection `Products`.", add_doc_op.error());
    collectionManager.drop_collection("Customers");

    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    customer_collection = collection_create_op.get();
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_FALSE(add_doc_op.ok());
    ASSERT_EQ("Referenced field `product_id` in the collection `Products` must be indexed.", add_doc_op.error());

    collectionManager.drop_collection("Products");
    products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_EQ("Referenced document having `product_id: a` not found in the collection `Products`.", add_doc_op.error());

    std::vector<nlohmann::json> products = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_a",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };
    for (auto const &json: products){
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    customer_json["reference_id"] = "product_a";
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_EQ("Multiple documents having `product_id: product_a` found in the collection `Products`.", add_doc_op.error());

    collectionManager.drop_collection("Products");
    products[1]["product_id"] = "product_b";
    products_schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(products_schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: products){
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    collectionManager.drop_collection("Customers");
    customers_schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "reference_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    collection_create_op = collectionManager.create_collection(customers_schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    customer_collection = collection_create_op.get();
    add_doc_op = customer_collection->add(customer_json.dump());
    ASSERT_TRUE(add_doc_op.ok());
    ASSERT_EQ(customer_collection->get("0").get().count("reference_id_sequence_id"), 1);

    nlohmann::json document;
    // Referenced document's sequence_id must be valid.
    auto get_op = collectionManager.get_collection("Products")->get_document_from_store(
            customer_collection->get("0").get()["reference_id_sequence_id"].get<uint32_t>(),
                    document);
    ASSERT_TRUE(get_op.ok());
    ASSERT_EQ(document.count("product_id"), 1);
    ASSERT_EQ(document["product_id"], "product_a");
    ASSERT_EQ(document["product_name"], "shampoo");

    collectionManager.drop_collection("Customers");
    collectionManager.drop_collection("Products");
}

TEST_F(CollectionJoinTest, FilterByReference_SingleMatch) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto coll = collectionManager.get_collection_unsafe("Products");
    auto search_op = coll->search("s", {"product_name"}, "$foo:=customer_a", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Could not parse the reference filter.");

    search_op = coll->search("s", {"product_name"}, "$foo(:=customer_a", {}, {}, {0},
                                  10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Could not parse the reference filter.");

    search_op = coll->search("s", {"product_name"}, "$foo(:=customer_a)", {}, {}, {0},
                                  10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Referenced collection `foo` not found.");

    search_op = coll->search("s", {"product_name"}, "$Customers(foo:=customer_a)", {}, {}, {0},
                             10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ(search_op.error(), "Failed to apply reference filter on `Customers` collection: Could not find a filter field named `foo` in the schema.");

    auto result = coll->search("s", {"product_name"}, "$Customers(customer_id:=customer_a && product_price:<100)", {}, {}, {0},
                             10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(1, result["found"].get<size_t>());
    ASSERT_EQ(1, result["hits"].size());
    ASSERT_EQ("soap", result["hits"][0]["document"]["product_name"].get<std::string>());

    collectionManager.drop_collection("Customers");
    collectionManager.drop_collection("Products");
}

TEST_F(CollectionJoinTest, FilterByReference_MultipleMatch) {
    auto schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto coll = collectionManager.get_collection_unsafe("Users");

    // Search for users linked to repo_b
    auto result = coll->search("R", {"user_name"}, "$Links(repo_id:=repo_b)", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(2, result["found"].get<size_t>());
    ASSERT_EQ(2, result["hits"].size());
    ASSERT_EQ("user_b", result["hits"][0]["document"]["user_id"].get<std::string>());
    ASSERT_EQ("user_a", result["hits"][1]["document"]["user_id"].get<std::string>());

    collectionManager.drop_collection("Users");
    collectionManager.drop_collection("Repos");
    collectionManager.drop_collection("Links");
}

TEST_F(CollectionJoinTest, AndFilterResults_NoReference) {
    filter_result_t a;
    a.count = 9;
    a.docs = new uint32_t[a.count];
    for (size_t i = 0; i < a.count; i++) {
        a.docs[i] = i;
    }

    filter_result_t b;
    b.count = 0;
    uint32_t limit = 10;
    b.docs = new uint32_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            b.docs[b.count++] = i;
        }
    }

    // a.docs: [0..8] , b.docs: [3, 6, 9]
    filter_result_t result;
    filter_result_t::and_filter_results(a, b, result);

    ASSERT_EQ(2, result.count);
    ASSERT_EQ(0, result.reference_filter_results.size());

    std::vector<uint32_t> docs = {3, 6};

    for(size_t i = 0; i < result.count; i++) {
        ASSERT_EQ(docs[i], result.docs[i]);
    }
}

TEST_F(CollectionJoinTest, AndFilterResults_WithReferences) {
    filter_result_t a;
    a.count = 9;
    a.docs = new uint32_t[a.count];
    a.reference_filter_results["foo"] = new reference_filter_result_t[a.count];
    for (size_t i = 0; i < a.count; i++) {
        a.docs[i] = i;

        // Having only one reference of each document for brevity.
        auto& reference = a.reference_filter_results["foo"][i];
        reference.count = 1;
        reference.docs = new uint32_t[1];
        reference.docs[0] = 10 - i;
    }

    filter_result_t b;
    b.count = 0;
    uint32_t limit = 10;
    b.docs = new uint32_t[limit];
    b.reference_filter_results["bar"] = new reference_filter_result_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            b.docs[b.count] = i;

            auto& reference = b.reference_filter_results["bar"][b.count++];
            reference.count = 1;
            reference.docs = new uint32_t[1];
            reference.docs[0] = 2 * i;
        }
    }

    // a.docs: [0..8] , b.docs: [3, 6, 9]
    filter_result_t result;
    filter_result_t::and_filter_results(a, b, result);

    ASSERT_EQ(2, result.count);
    ASSERT_EQ(2, result.reference_filter_results.size());
    ASSERT_EQ(1, result.reference_filter_results.count("foo"));
    ASSERT_EQ(1, result.reference_filter_results.count("bar"));

    std::vector<uint32_t> docs = {3, 6}, foo_reference = {7, 4}, bar_reference = {6, 12};

    for(size_t i = 0; i < result.count; i++) {
        ASSERT_EQ(docs[i], result.docs[i]);

        // result should contain correct references to the foo and bar collection.
        ASSERT_EQ(1, result.reference_filter_results["foo"][i].count);
        ASSERT_EQ(foo_reference[i], result.reference_filter_results["foo"][i].docs[0]);
        ASSERT_EQ(1, result.reference_filter_results["bar"][i].count);
        ASSERT_EQ(bar_reference[i], result.reference_filter_results["bar"][i].docs[0]);
    }
}

TEST_F(CollectionJoinTest, OrFilterResults_NoReference) {
    filter_result_t a, b;
    a.count = 0;
    uint32_t limit = 10;
    a.docs = new uint32_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            a.docs[a.count++] = i;
        }
    }

    // a.docs: [3, 6, 9], b.docs: []
    filter_result_t result1;
    filter_result_t::or_filter_results(a, b, result1);
    ASSERT_EQ(3, result1.count);
    ASSERT_EQ(0, result1.reference_filter_results.size());

    std::vector<uint32_t> expected = {3, 6, 9};
    for (size_t i = 0; i < result1.count; i++) {
        ASSERT_EQ(expected[i], result1.docs[i]);
    }

    b.count = 9;
    b.docs = new uint32_t[b.count];
    for (size_t i = 0; i < b.count; i++) {
        b.docs[i] = i;
    }

    // a.docs: [3, 6, 9], b.docs: [0..8]
    filter_result_t result2;
    filter_result_t::or_filter_results(a, b, result2);
    ASSERT_EQ(10, result2.count);
    ASSERT_EQ(0, result2.reference_filter_results.size());

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (size_t i = 0; i < result2.count; i++) {
        ASSERT_EQ(expected[i], result2.docs[i]);
    }


    filter_result_t c, result3;

    std::vector<uint32_t> vec = {0, 4, 5};
    c.docs = new uint32_t[vec.size()];
    auto j = 0;
    for(auto i: vec) {
        a.docs[j++] = i;
    }

    // b.docs: [0..8], c.docs: [0, 4, 5]
    filter_result_t::or_filter_results(b, c, result3);
    ASSERT_EQ(9, result3.count);
    ASSERT_EQ(0, result3.reference_filter_results.size());

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for(size_t i = 0; i < result3.count; i++) {
        ASSERT_EQ(expected[i], result3.docs[i]);
    }
}

TEST_F(CollectionJoinTest, OrFilterResults_WithReferences) {
    filter_result_t a, b;
    uint32_t limit = 10;

    a.count = 0;
    a.docs = new uint32_t[limit];
    a.reference_filter_results["foo"] = new reference_filter_result_t[limit];
    for (size_t i = 2; i < limit; i++) {
        if (i % 3 == 0) {
            a.docs[a.count] = i;

            auto& reference = a.reference_filter_results["foo"][a.count++];
            reference.count = 1;
            reference.docs = new uint32_t[1];
            reference.docs[0] = 2 * i;
        }
    }

    // a.docs: [3, 6, 9], b.docs: []
    filter_result_t result1;
    filter_result_t::or_filter_results(a, b, result1);

    ASSERT_EQ(3, result1.count);
    ASSERT_EQ(1, result1.reference_filter_results.size());
    ASSERT_EQ(1, result1.reference_filter_results.count("foo"));

    std::vector<uint32_t> expected = {3, 6, 9}, foo_reference = {6, 12, 18};
    for (size_t i = 0; i < result1.count; i++) {
        ASSERT_EQ(expected[i], result1.docs[i]);

        ASSERT_EQ(1, result1.reference_filter_results["foo"][i].count);
        ASSERT_EQ(foo_reference[i], result1.reference_filter_results["foo"][i].docs[0]);
    }

    b.count = 9;
    b.docs = new uint32_t[b.count];
    b.reference_filter_results["bar"] = new reference_filter_result_t[b.count];
    for (size_t i = 0; i < b.count; i++) {
        b.docs[i] = i;

        auto& reference = b.reference_filter_results["bar"][i];
        reference.count = 1;
        reference.docs = new uint32_t[1];
        reference.docs[0] = 10 - i;
    }

    // a.docs: [3, 6, 9], b.docs: [0..8]
    filter_result_t result2;
    filter_result_t::or_filter_results(a, b, result2);
    ASSERT_EQ(10, result2.count);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // doc_id -> reference_id
    std::map<uint32_t, uint32_t> foo_map = {{3, 6}, {6, 12}, {9, 18}}, bar_map = {{0, 10}, {1, 9}, {2, 8}, {3, 7},
                                                                                  {4, 6}, {5, 5}, {6, 4}, {7, 3}, {8, 2}};
    for (size_t i = 0; i < result2.count; i++) {
        ASSERT_EQ(expected[i], result2.docs[i]);

        if (foo_map.count(i) != 0) {
            ASSERT_EQ(1, result2.reference_filter_results["foo"][i].count);
            ASSERT_EQ(foo_map[i], result2.reference_filter_results["foo"][i].docs[0]);
        } else {
            // Reference count should be 0 for the docs that were not present in the a result.
            ASSERT_EQ(0, result2.reference_filter_results["foo"][i].count);
        }

        if (bar_map.count(i) != 0) {
            ASSERT_EQ(1, result2.reference_filter_results["bar"][i].count);
            ASSERT_EQ(bar_map[i], result2.reference_filter_results["bar"][i].docs[0]);
        } else {
            ASSERT_EQ(0, result2.reference_filter_results["bar"][i].count);
        }
    }

    filter_result_t c, result3;

    std::map<uint32_t, uint32_t> baz_map = {{0, 2}, {4, 0}, {5, 8}};
    c.count = baz_map.size();
    c.docs = new uint32_t[baz_map.size()];
    c.reference_filter_results["baz"] = new reference_filter_result_t[baz_map.size()];
    auto j = 0;
    for(auto i: baz_map) {
        c.docs[j] = i.first;

        auto& reference = c.reference_filter_results["baz"][j++];
        reference.count = 1;
        reference.docs = new uint32_t[1];
        reference.docs[0] = i.second;
    }

    // b.docs: [0..8], c.docs: [0, 4, 5]
    filter_result_t::or_filter_results(b, c, result3);
    ASSERT_EQ(9, result3.count);

    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    for (size_t i = 0; i < result3.count; i++) {
        ASSERT_EQ(expected[i], result3.docs[i]);

        if (bar_map.count(i) != 0) {
            ASSERT_EQ(1, result3.reference_filter_results["bar"][i].count);
            ASSERT_EQ(bar_map[i], result3.reference_filter_results["bar"][i].docs[0]);
        } else {
            ASSERT_EQ(0, result3.reference_filter_results["bar"][i].count);
        }

        if (baz_map.count(i) != 0) {
            ASSERT_EQ(1, result3.reference_filter_results["baz"][i].count);
            ASSERT_EQ(baz_map[i], result3.reference_filter_results["baz"][i].docs[0]);
        } else {
            ASSERT_EQ(0, result3.reference_filter_results["baz"][i].count);
        }
    }
}

TEST_F(CollectionJoinTest, FilterByNReferences) {
    auto schema_json =
            R"({
                "name": "Users",
                "fields": [
                    {"name": "user_id", "type": "string"},
                    {"name": "user_name", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "user_id": "user_a",
                "user_name": "Roshan"
            })"_json,
            R"({
                "user_id": "user_b",
                "user_name": "Ruby"
            })"_json,
            R"({
                "user_id": "user_c",
                "user_name": "Joe"
            })"_json,
            R"({
                "user_id": "user_d",
                "user_name": "Aby"
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Repos",
                "fields": [
                    {"name": "repo_id", "type": "string"},
                    {"name": "repo_content", "type": "string"},
                    {"name": "repo_stars", "type": "int32"},
                    {"name": "repo_is_private", "type": "bool"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "repo_content": "body1",
                "repo_stars": 431,
                "repo_is_private": true
            })"_json,
            R"({
                "repo_id": "repo_b",
                "repo_content": "body2",
                "repo_stars": 4562,
                "repo_is_private": false
            })"_json,
            R"({
                "repo_id": "repo_c",
                "repo_content": "body3",
                "repo_stars": 945,
                "repo_is_private": false
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Links",
                "fields": [
                    {"name": "repo_id", "type": "string", "reference": "Repos.repo_id"},
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"}
                ]
            })"_json;
    documents = {
            R"({
                "repo_id": "repo_a",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_a",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_b",
                "user_id": "user_d"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_a"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_b"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_c"
            })"_json,
            R"({
                "repo_id": "repo_c",
                "user_id": "user_d"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Organizations",
                "fields": [
                    {"name": "org_id", "type": "string"},
                    {"name": "org_name", "type": "string"}
                ]
            })"_json;
    documents = {
            R"({
                "org_id": "org_a",
                "org_name": "Typesense"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Participants",
                "fields": [
                    {"name": "user_id", "type": "string", "reference": "Users.user_id"},
                    {"name": "org_id", "type": "string", "reference": "Organizations.org_id"}
                ]
            })"_json;
    documents = {
            R"({
                "user_id": "user_a",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_b",
                "org_id": "org_a"
            })"_json,
            R"({
                "user_id": "user_d",
                "org_id": "org_a"
            })"_json,
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());

    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    auto coll = collectionManager.get_collection_unsafe("Users");

    // Search for users within an organization with access to a particular repo.
    auto result = coll->search("R", {"user_name"}, "$Participants(org_id:=org_a) && $Links(repo_id:=repo_b)", {}, {}, {0},
                               10, 1, FREQUENCY, {true}, Index::DROP_TOKENS_THRESHOLD).get();

    ASSERT_EQ(2, result["found"].get<size_t>());
    ASSERT_EQ(2, result["hits"].size());
    ASSERT_EQ("user_b", result["hits"][0]["document"]["user_id"].get<std::string>());
    ASSERT_EQ("user_a", result["hits"][1]["document"]["user_id"].get<std::string>());

    collectionManager.drop_collection("Users");
    collectionManager.drop_collection("Repos");
    collectionManager.drop_collection("Links");
}

TEST_F(CollectionJoinTest, IncludeFieldsByReference_SingleMatch) {
    auto schema_json =
            R"({
                "name": "Products",
                "fields": [
                    {"name": "product_id", "type": "string"},
                    {"name": "product_name", "type": "string"},
                    {"name": "product_description", "type": "string"}
                ]
            })"_json;
    std::vector<nlohmann::json> documents = {
            R"({
                "product_id": "product_a",
                "product_name": "shampoo",
                "product_description": "Our new moisturizing shampoo is perfect for those with dry or damaged hair."
            })"_json,
            R"({
                "product_id": "product_b",
                "product_name": "soap",
                "product_description": "Introducing our all-natural, organic soap bar made with essential oils and botanical ingredients."
            })"_json
    };
    auto collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    schema_json =
            R"({
                "name": "Customers",
                "fields": [
                    {"name": "customer_id", "type": "string"},
                    {"name": "customer_name", "type": "string"},
                    {"name": "product_price", "type": "float"},
                    {"name": "product_id", "type": "string", "reference": "Products.product_id"}
                ]
            })"_json;
    documents = {
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 143,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_a",
                "customer_name": "Joe",
                "product_price": 73.5,
                "product_id": "product_b"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 75,
                "product_id": "product_a"
            })"_json,
            R"({
                "customer_id": "customer_b",
                "customer_name": "Dan",
                "product_price": 140,
                "product_id": "product_b"
            })"_json
    };
    collection_create_op = collectionManager.create_collection(schema_json);
    ASSERT_TRUE(collection_create_op.ok());
    for (auto const &json: documents) {
        auto add_op = collection_create_op.get()->add(json.dump());
        if (!add_op.ok()) {
            LOG(INFO) << add_op.error();
        }
        ASSERT_TRUE(add_op.ok());
    }

    std::map<std::string, std::string> req_params = {
            {"collection", "Products"},
            {"q", "s"},
            {"query_by", "product_name"},
            {"filter_by", "$Customers(customer_id:=customer_a && product_price:<100)"},
    };

    nlohmann::json embedded_params;
    std::string json_res;
    auto now_ts = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    req_params["include_fields"] = "$foo.bar";
    auto search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Invalid reference in include_fields, expected `$CollectionName(fieldA, ...)`.", search_op.error());

    req_params["include_fields"] = "$foo(bar";
    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
    ASSERT_FALSE(search_op.ok());
    ASSERT_EQ("Invalid reference in include_fields, expected `$CollectionName(fieldA, ...)`.", search_op.error());

//    req_params["include_fields"] = "$foo(bar)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_FALSE(search_op.ok());
//    ASSERT_EQ("Referenced collection `foo` not found.", search_op.error());
//
//    req_params["include_fields"] = "$Customers(bar)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    nlohmann::json res_obj = nlohmann::json::parse(json_res);
//    ASSERT_EQ(1, res_obj["found"].get<size_t>());
//    ASSERT_EQ(1, res_obj["hits"].size());
//    ASSERT_EQ(0, res_obj["hits"][0]["document"].size());
//
//    req_params["include_fields"] = "$Customers(product_price)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    res_obj = nlohmann::json::parse(json_res);
//    ASSERT_EQ(1, res_obj["found"].get<size_t>());
//    ASSERT_EQ(1, res_obj["hits"].size());
//    ASSERT_EQ(1, res_obj["hits"][0]["document"].size());
//    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
//    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
//
//    req_params["include_fields"] = "$Customers(product_price, customer_id)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    res_obj = nlohmann::json::parse(json_res);
//    ASSERT_EQ(1, res_obj["found"].get<size_t>());
//    ASSERT_EQ(1, res_obj["hits"].size());
//    ASSERT_EQ(2, res_obj["hits"][0]["document"].size());
//    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_price"));
//    ASSERT_EQ(73.5, res_obj["hits"][0]["document"].at("product_price"));
//    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("customer_id"));
//    ASSERT_EQ("customer_a", res_obj["hits"][0]["document"].at("customer_id"));
//
//    req_params["include_fields"] = "*, $Customers(product_price, customer_id)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    res_obj = nlohmann::json::parse(json_res);
//    ASSERT_EQ(1, res_obj["found"].get<size_t>());
//    ASSERT_EQ(1, res_obj["hits"].size());
//    // 3 fields in Products document and 2 fields from Customers document
//    ASSERT_EQ(5, res_obj["hits"][0]["document"].size());
//
//    req_params["include_fields"] = "*, $Customers(product*)";
//    search_op = collectionManager.do_search(req_params, embedded_params, json_res, now_ts);
//    ASSERT_TRUE(search_op.ok());
//
//    res_obj = nlohmann::json::parse(json_res);
//    ASSERT_EQ(1, res_obj["found"].get<size_t>());
//    ASSERT_EQ(1, res_obj["hits"].size());
//    // 3 fields in Products document and 2 fields from Customers document
//    ASSERT_EQ(5, res_obj["hits"][0]["document"].size());
//    ASSERT_EQ(1, res_obj["hits"][0]["document"].count("product_id_sequence_id"));
}
