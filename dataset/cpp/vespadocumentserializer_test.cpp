// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
// Unit tests for vespadocumentserializer.

#include <vespa/document/test/fieldvalue_helpers.h>
#include <vespa/document/annotation/annotation.h>
#include <vespa/document/annotation/span.h>
#include <vespa/document/annotation/spantree.h>
#include <vespa/document/config/documenttypes_config_fwd.h>
#include <vespa/document/datatype/annotationreferencedatatype.h>
#include <vespa/document/datatype/arraydatatype.h>
#include <vespa/document/datatype/documenttype.h>
#include <vespa/document/datatype/weightedsetdatatype.h>
#include <vespa/document/datatype/mapdatatype.h>
#include <vespa/document/datatype/tensor_data_type.h>
#include <vespa/document/fieldvalue/annotationreferencefieldvalue.h>
#include <vespa/document/fieldvalue/arrayfieldvalue.h>
#include <vespa/document/fieldvalue/bytefieldvalue.h>
#include <vespa/document/fieldvalue/document.h>
#include <vespa/document/fieldvalue/doublefieldvalue.h>
#include <vespa/document/fieldvalue/floatfieldvalue.h>
#include <vespa/document/fieldvalue/intfieldvalue.h>
#include <vespa/document/fieldvalue/longfieldvalue.h>
#include <vespa/document/fieldvalue/mapfieldvalue.h>
#include <vespa/document/fieldvalue/predicatefieldvalue.h>
#include <vespa/document/fieldvalue/rawfieldvalue.h>
#include <vespa/document/fieldvalue/shortfieldvalue.h>
#include <vespa/document/fieldvalue/stringfieldvalue.h>
#include <vespa/document/fieldvalue/structfieldvalue.h>
#include <vespa/document/fieldvalue/weightedsetfieldvalue.h>
#include <vespa/document/fieldvalue/tensorfieldvalue.h>
#include <vespa/document/fieldvalue/referencefieldvalue.h>
#include <vespa/document/predicate/predicate.h>
#include <vespa/document/predicate/predicate_slime_builder.h>
#include <vespa/document/repo/configbuilder.h>
#include <vespa/document/repo/fixedtyperepo.h>
#include <vespa/document/repo/documenttyperepo.h>
#include <vespa/document/serialization/vespadocumentdeserializer.h>
#include <vespa/document/serialization/vespadocumentserializer.h>
#include <vespa/document/serialization/annotationserializer.h>
#include <vespa/eval/eval/value.h>
#include <vespa/eval/eval/simple_value.h>
#include <vespa/eval/eval/tensor_spec.h>
#include <vespa/eval/eval/test/value_compare.h>
#include <vespa/vespalib/io/fileutil.h>
#include <vespa/vespalib/objects/nbostream.h>
#include <vespa/vespalib/testkit/testapp.h>
#include <vespa/document/base/exceptions.h>
#include <vespa/vespalib/util/compressionconfig.h>
#include <filesystem>

using vespalib::File;
using vespalib::Slime;
using vespalib::nbostream;
using vespalib::nbostream_longlivedbuf;
using vespalib::slime::Cursor;
using vespalib::eval::TensorSpec;
using vespalib::eval::SimpleValue;
using vespalib::compression::CompressionConfig;
using namespace document;
using std::string;
using std::vector;
using namespace document::config_builder;

namespace {

const int doc_type_id = 1234;
const string doc_name = "my_doctype";
const int body_id = 94;
const int inner_type_id = 95;
const int outer_type_id = 96;
const string type_name = "outer_doc";
const string inner_name = "inner_doc";
const int a_id = 12345;
const string a_name = "annotation";
const int predicate_doc_type_id = 321;
const string predicate_field_name = "my_predicate";
const int doc_with_ref_type_id = 54321;
const string doc_with_ref_name = "doc_with_ref";
const string ref_field_name = "ref_field";
const int ref_type_id = 789;

constexpr uint16_t serialization_version = Document::getNewestSerializationVersion();

DocumenttypesConfig getDocTypesConfig() {
    DocumenttypesConfigBuilderHelper builder;
    builder.document(doc_type_id, doc_name,
                     Struct(doc_name + ".header")
                     .addField("header field", DataType::T_INT),
                     Struct(doc_name + ".body")
                     .addField("body field", DataType::T_STRING))
        .annotationType(42, "foo_type", DataType::T_INT);
    builder.document(inner_type_id, inner_name,
                     Struct(inner_name + ".header"),
                     Struct(inner_name + ".body")
                     .addField("str", DataType::T_STRING))
        .annotationType(a_id, a_name, DataType::T_STRING);
    builder.document(outer_type_id, type_name,
                     Struct(type_name + ".header"),
                     Struct(type_name + ".body")
                     .addField(inner_name, inner_type_id).setId(body_id));
    builder.document(predicate_doc_type_id, "my_type",
                     Struct("my_type.header"),
                     Struct("my_type.body")
                     .addField(predicate_field_name, DataType::T_PREDICATE));
    builder.document(doc_with_ref_type_id, doc_with_ref_name,
                     Struct(doc_with_ref_name + ".header")
                     .addField(ref_field_name, ref_type_id),
                     Struct(doc_with_ref_name + ".body"))
        .referenceType(ref_type_id, doc_type_id);
    return builder.config();
}

const DocumentTypeRepo doc_repo(getDocTypesConfig());
const FixedTypeRepo repo(doc_repo, *doc_repo.getDocumentType(doc_type_id));

template <typename T> T newFieldValue(const T&) {
    return T();
}
template <> ArrayFieldValue newFieldValue(const ArrayFieldValue &value) {
    return ArrayFieldValue(*value.getDataType());
}
template <> MapFieldValue newFieldValue(const MapFieldValue &value) {
    return MapFieldValue(*value.getDataType());
}
template <> WeightedSetFieldValue newFieldValue(
        const WeightedSetFieldValue &value) {
    return WeightedSetFieldValue(*value.getDataType());
}
template <> StructFieldValue newFieldValue(const StructFieldValue &value) {
    return StructFieldValue(*value.getDataType());
}
template <> AnnotationReferenceFieldValue newFieldValue(
        const AnnotationReferenceFieldValue &value) {
    return AnnotationReferenceFieldValue(*value.getDataType());
}
template <> ReferenceFieldValue newFieldValue(const ReferenceFieldValue& value) {
    return ReferenceFieldValue(dynamic_cast<const ReferenceDataType&>(
            *value.getDataType()));
}

template <> TensorFieldValue newFieldValue(const TensorFieldValue& value) {
    return TensorFieldValue(dynamic_cast<const TensorDataType&>(
            *value.getDataType()));
}

template<typename T>
void testDeserializeAndClone(const T& value, const nbostream &stream, bool checkEqual=true) {
    T read_value = newFieldValue(value);
    vespalib::MallocPtr buf(stream.size());
    memcpy(buf.str(), stream.peek(), stream.size());
    nbostream_longlivedbuf is(buf.c_str(), buf.size());
    VespaDocumentDeserializer deserializer(repo, is, serialization_version);
    deserializer.read(read_value);

    EXPECT_EQUAL(0u, is.size());
    if (checkEqual) {
        EXPECT_EQUAL(value, read_value);
    }
    T clone(read_value);
    buf.reset();
    if (checkEqual) {
        EXPECT_EQUAL(value, clone);
    }
}

// Leaves the stream's read position at the start of the serialized object.
template<typename T>
void serializeAndDeserialize(const T& value, nbostream &stream,
                             const FixedTypeRepo& fixed_repo, bool checkEqual = true) {
    size_t start_size = stream.size();
    VespaDocumentSerializer serializer(stream);
    serializer.write(value);
    size_t serialized_size = stream.size() - start_size;

    testDeserializeAndClone(value, stream, checkEqual);
    T read_value = newFieldValue(value);

    VespaDocumentDeserializer deserializer(fixed_repo, stream, serialization_version);
    deserializer.read(read_value);

    EXPECT_EQUAL(0u, stream.size());
    if (checkEqual) {
        EXPECT_EQUAL(value, read_value);
    }
    stream.adjustReadPos(-serialized_size);
    nbostream stream2;
    VespaDocumentSerializer serializer2(stream2);
    serializer2.write(read_value);
    EXPECT_EQUAL(serialized_size, stream2.size());
    EXPECT_EQUAL(0, memcmp(stream.peek() + start_size, stream2.peek(), serialized_size));
}

template<typename T>
void serializeAndDeserialize(const T& value, nbostream &stream, bool checkEqual=true) {
    serializeAndDeserialize(value, stream, repo, checkEqual);
}

template <typename T> struct ValueType { typedef typename T::Number Type; };
template <> struct ValueType<IntFieldValue> { typedef uint32_t Type; };
template <> struct ValueType<LongFieldValue> { typedef uint64_t Type; };

template<typename T>
void serializeAndDeserializeNumber(const T& value) {
    const typename ValueType<T>::Type val = value.getValue();
    nbostream stream;
    serializeAndDeserialize(value, stream);

    typename ValueType<T>::Type read_val;
    stream >> read_val;
    EXPECT_EQUAL(val, read_val);
}

TEST("requireThatPrimitiveTypeFieldValueCanBeSerialized") {
    TEST_DO(serializeAndDeserializeNumber(ByteFieldValue(42)));
    TEST_DO(serializeAndDeserializeNumber(ShortFieldValue(0x1234)));
    TEST_DO(serializeAndDeserializeNumber(IntFieldValue(0x12345678)));
    TEST_DO(serializeAndDeserializeNumber(DoubleFieldValue(34567890.123456)));
    TEST_DO(serializeAndDeserializeNumber(FloatFieldValue(3456.1234f)));
    TEST_DO(serializeAndDeserializeNumber(LongFieldValue(0x12345678123456LL)));
}

template <typename SizeType>
void checkLiteralFieldValue(nbostream &stream, const string &val) {
    uint8_t read_coding;
    SizeType size;
    stream >> read_coding >> size;
    EXPECT_EQUAL(0, read_coding);
    size &= (SizeType(-1) >> 1);  // Clear MSB.
    vespalib::string read_val;
    read_val.assign(stream.peek(), size);
    stream.adjustReadPos(size);
    EXPECT_EQUAL(val.size(), read_val.size());
    EXPECT_EQUAL(val, read_val);
}

template <typename SizeType>
void checkStringFieldValue(const string &val) {
    StringFieldValue value(val);
    nbostream stream;
    serializeAndDeserialize(value, stream);

    string val_with_null(val.c_str(), val.size() + 1);
    checkLiteralFieldValue<SizeType>(stream, val_with_null);
}

void setSpanTree(StringFieldValue & str, const SpanTree & tree) {
    nbostream os;
    AnnotationSerializer serializer(os);
    serializer.write(tree);
    str.setSpanTrees(vespalib::ConstBufferRef(os.peek(), os.size()), repo, VespaDocumentSerializer::getCurrentVersion(), false);
}

void checkStringFieldValueWithAnnotation() {
    StringFieldValue value("foo");
    Span::UP root(new Span(2, 3));
    SpanTree::UP tree(new SpanTree("test", std::move(root)));
    AnnotationType annotation_type(42, "foo_type");
    tree->annotate(tree->getRoot(), annotation_type);

    nbostream os;
    AnnotationSerializer serializer(os);
    serializer.write(*tree);

    value.setSpanTrees(vespalib::ConstBufferRef(os.peek(), os.size()), repo, VespaDocumentSerializer::getCurrentVersion(), true);
    nbostream stream;
    serializeAndDeserialize(value, stream);
}

TEST("requireThatStringFieldValueCanBeSerialized") {
    TEST_DO(checkStringFieldValue<uint8_t>("foo bar baz"));
    TEST_DO(checkStringFieldValue<uint32_t>(string(200, 'a')));
    TEST_DO(checkStringFieldValueWithAnnotation());
}

TEST("require that strings can be re-deserialized") {
    StringFieldValue value("foo");
    nbostream streamNotAnnotated;
    VespaDocumentSerializer serializer(streamNotAnnotated);
    serializer.write(value);

    Span::UP root(new Span(2, 3));
    SpanTree::UP tree(new SpanTree("test", std::move(root)));
    AnnotationType annotation_type(42, "foo_type");
    tree->annotate(tree->getRoot(), annotation_type);

    setSpanTree(value, *tree);

    nbostream streamAnnotated;
    VespaDocumentSerializer serializerAnnotated(streamAnnotated);
    serializerAnnotated.write(value);

    StringFieldValue deserialized;
    {
        VespaDocumentDeserializer deserializer(
                repo, streamAnnotated, serialization_version);
        deserializer.read(deserialized);
    }
    EXPECT_EQUAL("foo", deserialized.getValueRef());
    EXPECT_TRUE(deserialized.hasSpanTrees());
    {
        VespaDocumentDeserializer deserializer(
                repo, streamNotAnnotated, serialization_version);
        deserializer.read(deserialized);
    }
    EXPECT_EQUAL("foo", deserialized.getValueRef());
    EXPECT_FALSE(deserialized.hasSpanTrees());
}

template <typename SizeType>
void checkRawFieldValue(const string &val) {
    RawFieldValue value(val);
    nbostream stream;
    serializeAndDeserialize(value, stream);

    uint32_t size;
    stream >> size;
    string read_val(stream.peek(), size);
    stream.adjustReadPos(size);
    EXPECT_EQUAL(val, read_val);
}

TEST("requireThatRawFieldValueCanBeSerialized") {
    TEST_DO(checkRawFieldValue<uint8_t>("foo bar"));
    TEST_DO(checkRawFieldValue<uint32_t>(string(200, 'b')));
}

TEST("requireThatPredicateFieldValueCanBeSerialized") {
    PredicateSlimeBuilder builder;
    builder.neg().feature("foo").value("bar").value("baz");
    PredicateFieldValue value(builder.build());
    nbostream stream;
    serializeAndDeserialize(value, stream);
}

template <typename SizeType>
SizeType readSize1_2_4(nbostream &stream) {
    SizeType size;
    stream >> size;
    if (sizeof(SizeType) > 1) {
        size &= (SizeType(-1) >> 2);  // Clear MSBs
    }
    return size;
}

template <typename SizeType>
void checkArrayFieldValue(SizeType value_count) {
    ArrayDataType array_type(*DataType::INT);
    ArrayFieldValue value(array_type);
    for (uint32_t i = 0; i < value_count; ++i) {
        CollectionHelper(value).add(static_cast<int32_t>(i));
    }

    nbostream stream;
    serializeAndDeserialize(value, stream);

    SizeType size = readSize1_2_4<SizeType>(stream);
    EXPECT_EQUAL(value_count, size);
    uint32_t child;
    for (uint32_t i = 0; i < value_count; ++i) {
        stream >> child;
        EXPECT_EQUAL(i, child);
    }
}

TEST("requireThatArrayFieldValueCanBeSerialized") {
    TEST_DO(checkArrayFieldValue<uint8_t>(2));
    TEST_DO(checkArrayFieldValue<uint8_t>(0x7f));
    TEST_DO(checkArrayFieldValue<uint16_t>(0x80));
    TEST_DO(checkArrayFieldValue<uint16_t>(0x3fff));
    TEST_DO(checkArrayFieldValue<uint32_t>(0x4000));
}

template <typename SizeType>
void checkMapFieldValue(SizeType value_count, bool check_equal) {
    MapDataType map_type(*DataType::LONG, *DataType::BYTE);
    MapFieldValue value(map_type);
    for (SizeType i = 0; i < value_count; ++i) {
        value.push_back(LongFieldValue(i), ByteFieldValue(i));
    }

    nbostream stream;
    serializeAndDeserialize(value, stream, check_equal);

    SizeType size = readSize1_2_4<SizeType>(stream);
    EXPECT_EQUAL(value_count, size);
    uint64_t key;
    uint8_t val;
    for (SizeType i = 0; i < value_count; ++i) {
        stream >> key >> val;
        EXPECT_EQUAL(i, key);
        EXPECT_EQUAL(i % 256, val);
    }
}

TEST("requireThatMapFieldValueCanBeSerialized") {
    TEST_DO(checkMapFieldValue<uint8_t>(2, true));
    TEST_DO(checkMapFieldValue<uint8_t>(0x7f, true));
    TEST_DO(checkMapFieldValue<uint16_t>(0x80, true));
    TEST_DO(checkMapFieldValue<uint16_t>(0x3fff, false));
    TEST_DO(checkMapFieldValue<uint32_t>(0x4000, false));
}

TEST("requireThatWeightedSetFieldValueCanBeSerialized") {
    WeightedSetDataType ws_type(*DataType::DOUBLE, false, false);
    WeightedSetFieldValue value(ws_type);
    value.add(DoubleFieldValue(3.14), 2);
    value.add(DoubleFieldValue(2.71), 3);

    nbostream stream;
    serializeAndDeserialize(value, stream);

    uint32_t type_id;
    uint32_t size;
    stream >> type_id >> size;
    EXPECT_EQUAL(2u, size);
    double val;
    uint32_t weight;
    stream >> size >> val >> weight;
    EXPECT_EQUAL(12u, size);
    EXPECT_EQUAL(3.14, val);
    EXPECT_EQUAL(2u, weight);
    stream >> size >> val >> weight;
    EXPECT_EQUAL(12u, size);
    EXPECT_EQUAL(2.71, val);
    EXPECT_EQUAL(3u, weight);
}

const Field field1("field1", *DataType::INT);
const Field field2("field2", *DataType::STRING);

StructDataType getStructDataType() {
    StructDataType struct_type("struct");
    struct_type.addField(field1);
    struct_type.addField(field2);
    return struct_type;
}

StructFieldValue getStructFieldValue(const StructDataType& structType) {
    StructFieldValue value(structType);
    value.setValue(field1, IntFieldValue(42));
    value.setValue(field2, StringFieldValue("foooooooooooooooooooooobaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    return value;
}

void checkStructSerialization(const StructFieldValue &value,
                              CompressionConfig::Type comp_type) {
    nbostream stream;
    serializeAndDeserialize(value, stream);
    uint32_t data_size;
    uint8_t compression_type;
    uint16_t uncompressed_size;
    uint8_t field_count;
    int32_t element1_id, element2_id;
    uint16_t element1_size, element2_size;
    stream >> data_size >> compression_type;
    if (CompressionConfig::isCompressed(comp_type)) {
        stream >> uncompressed_size;
        EXPECT_EQUAL(24u, data_size);
        EXPECT_EQUAL(64u, uncompressed_size);
    } else {
        EXPECT_EQUAL(64u, data_size);
    }
    stream >> field_count
           >> element1_id >> element1_size
           >> element2_id >> element2_size;
    EXPECT_EQUAL(comp_type, compression_type);
    EXPECT_EQUAL(2u, field_count);
    EXPECT_EQUAL(field1.getId(), element1_id & 0x7fffffff);
    EXPECT_EQUAL(4u, element1_size);
    EXPECT_EQUAL(field2.getId(), element2_id & 0x7fffffff);
    EXPECT_EQUAL(60u, element2_size);
}

TEST("requireThatEmptyStructCanBeSerialized") {
    StructDataType structType(getStructDataType());
    StructFieldValue value(structType);
    nbostream stream;
    serializeAndDeserialize(value, stream);
    uint32_t data_size;
    uint8_t compression_type;
    uint8_t field_count;
    stream >> data_size >> compression_type >> field_count;
    EXPECT_EQUAL(0u, data_size);
    EXPECT_EQUAL(0u, field_count);
}

TEST("requireThatUncompressedStructFieldValueCanBeSerialized") {
    StructDataType structType(getStructDataType());
    StructFieldValue value = getStructFieldValue(structType);
    checkStructSerialization(value, CompressionConfig::NONE);
}

TEST("requireThatReserializationIsUnompressedIfUnmodified") {
    StructDataType structType(getStructDataType());
    StructFieldValue value = getStructFieldValue(structType);

    TEST_DO(checkStructSerialization(value, CompressionConfig::NONE));

    nbostream os;
    VespaDocumentSerializer serializer(os);
    serializer.write(value);

    StructDataType struct_type(getStructDataType());
    StructFieldValue value2(struct_type);
    VespaDocumentDeserializer deserializer(repo, os, serialization_version);
    deserializer.read(value2);
    TEST_DO(checkStructSerialization(value, CompressionConfig::NONE));
    // Lazy serialization of structs....
    TEST_DO(checkStructSerialization(value2, CompressionConfig::NONE));
    EXPECT_EQUAL(value, value2);
}

template <typename T, int N> int arraysize(const T (&)[N]) { return N; }

TEST("requireThatDocumentCanBeSerialized") {
    const DocumentType &type = repo.getDocumentType();

    DocumentId doc_id("id:ns:" + type.getName() + "::");
    Document value(repo.getDocumentTypeRepo(), type, doc_id);

    value.setValue(type.getField("header field"), IntFieldValue(42));
    value.setValue(type.getField("body field"), StringFieldValue("foobar"));

    nbostream stream;
    serializeAndDeserialize(value, stream);

    uint16_t read_version;
    uint32_t size;
    stream >> read_version >> size;
    EXPECT_EQUAL(serialization_version, read_version);
    EXPECT_EQUAL(64u, size);
    EXPECT_EQUAL(doc_id.getScheme().toString(), stream.peek());
    stream.adjustReadPos(doc_id.getScheme().toString().size() + 1);
    uint8_t content_code;
    stream >> content_code;
    EXPECT_EQUAL(0x03, content_code);
    EXPECT_EQUAL(type.getName(), stream.peek());
    stream.adjustReadPos(type.getName().size() + 1);
    stream >> read_version;
    EXPECT_EQUAL(0, read_version);
}

TEST("requireThatOldVersionDocumentCanNotBeDeserialized") {
    uint16_t old_version = 6;
    uint16_t data_size = 432;
    string doc_id = "id:ns:my_doctype::";
    uint8_t content_code = 0x01;
    uint32_t crc = 42;

    nbostream stream;
    stream << old_version << data_size;
    stream.write(doc_id.c_str(), doc_id.size() + 1);
    stream << content_code;
    stream.write(doc_name.c_str(), doc_name.size() + 1);
    stream << static_cast<uint16_t>(0) << crc;  // version (unused)

    Document value;
    VespaDocumentDeserializer deserializer(repo, stream, old_version);
    try {
        deserializer.read(value);
        ASSERT_FALSE(true);
    } catch (vespalib::Exception & e) {
        EXPECT_EQUAL("CORRUPT DATA: Unrecognized serialization version 6", e.getMessage());
    }
}

TEST("requireThatUnmodifiedDocumentRetainsUnknownFieldOnSerialization") {

    DocumenttypesConfigBuilderHelper builder1, builder2;
    builder1.document(doc_type_id, doc_name,
                      Struct("my_doctype.header")
                      .addField("field2", DataType::T_STRING),
                      Struct("my_doctype.body"));
    builder2.document(doc_type_id, doc_name,
                      Struct("my_doctype.header")
                      .addField("field1", DataType::T_INT)
                      .addField("field2", DataType::T_STRING),
                      Struct("my_doctype.body"));
		
    DocumentTypeRepo repo1Field(builder1.config());
    DocumentTypeRepo repo2Fields(builder2.config());

    DocumentId doc_id("id:ns:my_doctype::");
    Document value(repo2Fields, *repo2Fields.getDocumentType(doc_type_id), doc_id);

    value.setValue("field1", IntFieldValue(42));
    value.setValue("field2", StringFieldValue("megafoo"));

    nbostream stream;
    VespaDocumentSerializer serializer(stream);
    serializer.write(value);

    Document read_value;
    // Deserialize+serialize with type where field1 is not known.
    VespaDocumentDeserializer deserializer1(repo1Field, stream, serialization_version);
    deserializer1.read(read_value);
    EXPECT_EQUAL(0u, stream.size());

    EXPECT_EQUAL(1u, read_value.getSetFieldCount());

    stream.clear();
    serializer.write(read_value);

    Document read_value_2;
    // Field should not have vanished.
    VespaDocumentDeserializer deserializer2(repo2Fields, stream, serialization_version);
    deserializer2.read(read_value_2);
    EXPECT_EQUAL(value, read_value_2);
}

TEST("requireThatAnnotationReferenceFieldValueCanBeSerialized") {
    AnnotationType annotation_type(0, "atype");
    AnnotationReferenceDataType type(annotation_type, 0);
    int annotation_id = 420;
    AnnotationReferenceFieldValue value(type, annotation_id);
    nbostream stream;
    serializeAndDeserialize(value, stream);

    int read_id = readSize1_2_4<uint16_t>(stream);
    EXPECT_EQUAL(annotation_id, read_id);
}

TEST("requireThatDocumentWithDocumentCanBeSerialized") {
    const DocumentTypeRepo &my_repo = repo.getDocumentTypeRepo();
    const DocumentType *inner_type = my_repo.getDocumentType(inner_type_id);
    ASSERT_TRUE(inner_type);
    const AnnotationType *a_type =my_repo.getAnnotationType(*inner_type, a_id);
    StringFieldValue str("foo");
    auto tree = std::make_unique<SpanTree>("name", std::make_unique<Span>(0, 3));
    tree->annotate(Annotation(*a_type));


    setSpanTree(str, *tree);
    const Field str_field("str", *DataType::STRING);

    Document inner(my_repo, *inner_type, DocumentId("id:ns:" + inner_type->getName() + "::"));
    inner.setValue(str_field, str);
    const DocumentType *type = my_repo.getDocumentType(outer_type_id);
    ASSERT_TRUE(type);
    DocumentId doc_id("id:ns:" + type->getName() + "::");
    Document value(my_repo, *type, doc_id);
    const Field doc_field(inner_name, *inner_type);
    value.setValue(doc_field, inner);

    nbostream stream;
    serializeAndDeserialize(value, stream);
}

TEST("requireThatReadDocumentTypeThrowsIfUnknownType") {
    string my_type("my_unknown_type");
    nbostream stream;
    stream.write(my_type.c_str(), my_type.size() + 1);
    stream << static_cast<uint16_t>(0);  // version (unused)

    DocumentType value("invalid");
    VespaDocumentDeserializer deserializer(repo, stream, serialization_version);
    EXPECT_EXCEPTION(deserializer.read(value), DocumentTypeNotFoundException,
                "Document type " + my_type + " not found");
}

template <typename FieldValueT>
void serializeToFile(FieldValueT &value, const string &file_name,
                     const DocumentTypeRepo& my_repo,
                     const DocumentType *type, const string &field_name) {
    DocumentId doc_id("id:test:" + type->getName() + "::foo");
    Document doc(my_repo, *type, doc_id);
    doc.setValue(type->getField(field_name), value);

    nbostream stream;
    VespaDocumentSerializer serializer(stream);
    serializer.write(doc);

    File file(file_name);
    file.open(File::CREATE);
    file.write(stream.peek(), stream.size(), 0);
}

void serializeToFile(PredicateFieldValue &value, const string &file_name) {
    const DocumentType *type = doc_repo.getDocumentType(predicate_doc_type_id);
    serializeToFile(value, file_name, doc_repo, type, predicate_field_name);
}

template <typename FieldValueT>
void deserializeAndCheck(const string &file_name, FieldValueT &value,
                         const FixedTypeRepo &myrepo,
                         const string &field_name) {
    File file(file_name);
    file.open(File::READONLY);
    vector<char> content(file.getFileSize());
    size_t r = file.read(&content[0], content.size(), 0);
    ASSERT_EQUAL(content.size(), r);

    nbostream_longlivedbuf stream(&content[0], content.size());
    Document doc;
    VespaDocumentDeserializer deserializer(myrepo, stream, serialization_version);
    deserializer.read(doc);

    ASSERT_EQUAL(0, value.compare(*doc.getValue(field_name)));
}

void deserializeAndCheck(const string &file_name, PredicateFieldValue &value) {
    deserializeAndCheck(file_name, value, repo, predicate_field_name);
}

void checkDeserialization(const string &name, std::unique_ptr<Slime> slime) {
    const string data_dir = TEST_PATH("../../test/resources/predicates/");

    PredicateFieldValue value(std::move(slime));
    serializeToFile(value, data_dir + name + "__cpp.new");
    std::filesystem::rename(std::filesystem::path(data_dir + name + "__cpp.new"),
                            std::filesystem::path(data_dir + name + "__cpp"));

    deserializeAndCheck(data_dir + name + "__cpp", value);
    deserializeAndCheck(data_dir + name + "__java", value);
}

TEST("Require that predicate deserialization matches Java") {
    PredicateSlimeBuilder builder;

    builder.feature("foo").range(6, 9);
    checkDeserialization("foo_in_6_9", builder.build());

    builder.feature("foo").greaterEqual(6);
    checkDeserialization("foo_in_6_x", builder.build());

    builder.feature("foo").lessEqual(9);
    checkDeserialization("foo_in_x_9", builder.build());

    builder.feature("foo").value("bar");
    checkDeserialization("foo_in_bar", builder.build());

    builder.feature("foo").value("bar").value("baz");
    checkDeserialization("foo_in_bar_baz", builder.build());

    builder.neg().feature("foo").value("bar");
    checkDeserialization("not_foo_in_bar", builder.build());

    std::unique_ptr<Slime> slime(new Slime);
    Cursor *cursor = &slime->setObject();
    cursor->setString(Predicate::KEY, "foo");
    cursor->setLong(Predicate::NODE_TYPE, Predicate::TYPE_FEATURE_SET);
    checkDeserialization("foo_in_x", std::move(slime));

    slime.reset(new Slime);
    cursor = &slime->setObject();
    cursor->setString(Predicate::KEY, "foo");
    cursor->setLong(Predicate::NODE_TYPE, Predicate::TYPE_FEATURE_RANGE);
    checkDeserialization("foo_in_x_x", std::move(slime));

    std::unique_ptr<Slime> slime1 = builder.feature("foo").value("bar").build();
    std::unique_ptr<Slime> slime2 = builder.feature("baz").value("cox").build();
    builder.and_node(std::move(slime1), std::move(slime2));
    checkDeserialization("foo_in_bar_and_baz_in_cox", builder.build());

    slime1 = builder.feature("foo").value("bar").build();
    slime2 = builder.feature("baz").value("cox").build();
    builder.or_node(std::move(slime1), std::move(slime2));
    checkDeserialization("foo_in_bar_or_baz_in_cox", builder.build());

    builder.true_predicate();
    checkDeserialization("true", builder.build());

    builder.false_predicate();
    checkDeserialization("false", builder.build());
}

namespace
{

vespalib::eval::Value::UP createTensor(const TensorSpec &spec) {
    return SimpleValue::from_spec(spec);
}

}

TEST("Require that tensors can be serialized")
{
    TensorDataType xySparseTensorDataType(vespalib::eval::ValueType::from_spec("tensor(x{},y{})"));
    TensorFieldValue noTensorValue(xySparseTensorDataType);
    TensorFieldValue emptyTensorValue(xySparseTensorDataType);
    TensorFieldValue twoCellsTwoDimsValue(xySparseTensorDataType);
    nbostream stream;
    serializeAndDeserialize(noTensorValue, stream);
    stream.clear();
    emptyTensorValue = createTensor(TensorSpec("tensor(x{},y{})"));
    serializeAndDeserialize(emptyTensorValue, stream);
    stream.clear();
    twoCellsTwoDimsValue = createTensor(TensorSpec("tensor(x{},y{})")
                                        .add({{"x", ""}, {"y", "3"}}, 3)
                                        .add({{"x", "4"}, {"y", "5"}}, 7));
    serializeAndDeserialize(twoCellsTwoDimsValue, stream);
    EXPECT_NOT_EQUAL(noTensorValue, emptyTensorValue);
    EXPECT_NOT_EQUAL(noTensorValue, twoCellsTwoDimsValue);
    EXPECT_NOT_EQUAL(emptyTensorValue, twoCellsTwoDimsValue);
}



const int tensor_doc_type_id = 321;
const string tensor_field_name = "my_tensor";

DocumenttypesConfig getTensorDocTypesConfig(const vespalib::string &tensorType) {
    DocumenttypesConfigBuilderHelper builder;
    builder.document(tensor_doc_type_id, "my_type",
                     Struct("my_type.header"),
                     Struct("my_type.body")
                     .addTensorField(tensor_field_name, tensorType));
    return builder.config();
}

DocumenttypesConfig getTensorDocTypesConfig() {
    return getTensorDocTypesConfig("tensor(dimX{},dimY{})");
}

const DocumentTypeRepo tensor_doc_repo(getTensorDocTypesConfig());
const FixedTypeRepo tensor_repo(tensor_doc_repo,
                                *tensor_doc_repo.getDocumentType(tensor_doc_type_id));

const DocumentTypeRepo tensor_doc_repo1(getTensorDocTypesConfig("tensor(dimX{})"));

void serializeToFile(TensorFieldValue &value, const string &file_name) {
    const DocumentType *type =
        tensor_doc_repo.getDocumentType(tensor_doc_type_id);
    serializeToFile(value, file_name, tensor_doc_repo, type, tensor_field_name);
}

void deserializeAndCheck(const string &file_name, TensorFieldValue &value) {
    deserializeAndCheck(file_name, value, tensor_repo, tensor_field_name);
}

void checkDeserialization(const string &name, std::unique_ptr<vespalib::eval::Value> tensor) {
    const string data_dir = TEST_PATH("../../test/resources/tensor/");

    TensorDataType valueType(tensor ? tensor->type() : vespalib::eval::ValueType::error_type());
    TensorFieldValue value(valueType);
    if (tensor) {
        value = std::move(tensor);
    }
    serializeToFile(value, data_dir + name + "__cpp.new");
    std::filesystem::rename(std::filesystem::path(data_dir + name + "__cpp.new"),
                            std::filesystem::path(data_dir + name + "__cpp"));

    deserializeAndCheck(data_dir + name + "__cpp", value);
    deserializeAndCheck(data_dir + name + "__java", value);
}


TEST("Require that tensor deserialization matches Java") {
    checkDeserialization("non_existing_tensor", std::unique_ptr<vespalib::eval::Value>());
    checkDeserialization("empty_tensor", createTensor(TensorSpec("tensor(dimX{},dimY{})")));
    checkDeserialization("multi_cell_tensor",
                         createTensor(TensorSpec("tensor(dimX{},dimY{})")
                                 .add({{"dimX", "a"}, {"dimY", "bb"}}, 2.0)
                                 .add({{"dimX", "ccc"}, {"dimY", "dddd"}}, 3.0)
                                 .add({{"dimX", "e"}, {"dimY", "ff"}}, 5.0)));
}

struct TensorDocFixture {
    const DocumentTypeRepo &_docTypeRepo;
    const DocumentType *_docType;
    std::unique_ptr<vespalib::eval::Value> _tensor;
    Document _doc;
    vespalib::nbostream _blob;

    TensorDocFixture(const DocumentTypeRepo &docTypeRepo,
                     std::unique_ptr<vespalib::eval::Value> tensor);
    ~TensorDocFixture();
};

TensorDocFixture::TensorDocFixture(const DocumentTypeRepo &docTypeRepo,
                                   std::unique_ptr<vespalib::eval::Value> tensor)
    : _docTypeRepo(docTypeRepo),
      _docType(_docTypeRepo.getDocumentType(tensor_doc_type_id)),
      _tensor(std::move(tensor)),
      _doc(_docTypeRepo, *_docType, DocumentId("id:test:my_type::foo")),
      _blob()
{
    auto fv = _doc.getField(tensor_field_name).createValue();
    dynamic_cast<TensorFieldValue &>(*fv) = SimpleValue::from_value(*_tensor);
    _doc.setValue(tensor_field_name, *fv);
    _doc.serialize(_blob);
}

TensorDocFixture::~TensorDocFixture() = default;

struct DeserializedTensorDoc
{
    std::unique_ptr<Document> _doc;
    std::unique_ptr<FieldValue> _fieldValue;

    DeserializedTensorDoc();
    ~DeserializedTensorDoc();

    void setup(const DocumentTypeRepo &docTypeRepo, const vespalib::nbostream &blob);
    const vespalib::eval::Value *getTensor() const;
};

DeserializedTensorDoc::DeserializedTensorDoc()
    : _doc(),
      _fieldValue()
{
}

DeserializedTensorDoc::~DeserializedTensorDoc() = default;

void
DeserializedTensorDoc::setup(const DocumentTypeRepo &docTypeRepo, const vespalib::nbostream &blob)
{
    vespalib::nbostream wrapStream(blob.peek(), blob.size());
    _doc = std::make_unique<Document>(docTypeRepo, wrapStream);
    _fieldValue = _doc->getValue(tensor_field_name);
}

const vespalib::eval::Value *
DeserializedTensorDoc::getTensor() const
{
    return dynamic_cast<const TensorFieldValue &>(*_fieldValue).getAsTensorPtr();
}

TEST("Require that wrong tensor type hides tensor")
{
    TensorDocFixture f(tensor_doc_repo,
                       createTensor(TensorSpec("tensor(dimX{},dimY{})")
                                    .add({{"dimX", "a"}, {"dimY", "bb"}}, 2.0)
                                    .add({{"dimX", "ccc"}, {"dimY", "dddd"}}, 3.0)
                                    .add({{"dimX", "e"}, {"dimY", "ff"}}, 5.0)));
    TensorDocFixture f1(tensor_doc_repo1,
                        createTensor(TensorSpec("tensor(dimX{})")
                                .add({{"dimX", "a"}}, 20.0)
                                .add({{"dimX", "ccc"}}, 30.0)));
    DeserializedTensorDoc doc;
    doc.setup(tensor_doc_repo, f._blob);
    EXPECT_TRUE(doc.getTensor() != nullptr);
    EXPECT_TRUE((*doc.getTensor()) == (*f._tensor));
    doc.setup(tensor_doc_repo, f1._blob);
    EXPECT_TRUE(doc.getTensor() == nullptr);
    doc.setup(tensor_doc_repo1, f._blob);
    EXPECT_TRUE(doc.getTensor() == nullptr);
    doc.setup(tensor_doc_repo1, f1._blob);
    EXPECT_TRUE(doc.getTensor() != nullptr);
    EXPECT_TRUE((*doc.getTensor()) == (*f1._tensor));
}

struct RefFixture {
    const DocumentType* ref_doc_type{doc_repo.getDocumentType(doc_with_ref_type_id)};
    FixedTypeRepo fixed_repo{doc_repo, *ref_doc_type};

    const ReferenceDataType& ref_type() const {
        auto* raw_type = fixed_repo.getDataType(ref_type_id);
        assert(raw_type != nullptr);
        return dynamic_cast<const ReferenceDataType&>(*raw_type);
    }

    void roundtrip_serialize(const ReferenceFieldValue& src, ReferenceFieldValue& dest) {
        nbostream stream;
        VespaDocumentSerializer serializer(stream);
        serializer.write(src);

        VespaDocumentDeserializer deserializer(fixed_repo, stream, serialization_version);
        deserializer.read(dest);
    }

    void verify_cross_language_serialization(const string& file_base_name,
                                             const ReferenceFieldValue& value) {
        const string data_dir = TEST_PATH("../../test/resources/reference/");
        const string field_name = "ref_field";
        serializeToFile(value, data_dir + file_base_name + "__cpp.new",
                        fixed_repo.getDocumentTypeRepo(), ref_doc_type, field_name);
        std::filesystem::rename(std::filesystem::path(data_dir + file_base_name + "__cpp.new"),
                                std::filesystem::path(data_dir + file_base_name + "__cpp"));

        deserializeAndCheck(data_dir + file_base_name + "__cpp",
                            value, fixed_repo, field_name);
        deserializeAndCheck(data_dir + file_base_name + "__java",
                            value, fixed_repo, field_name);
    }
};

TEST_F("Empty ReferenceFieldValue can be roundtrip serialized", RefFixture) {
    ReferenceFieldValue empty_ref(f.ref_type());
    nbostream stream;
    serializeAndDeserialize(empty_ref, stream, f.fixed_repo);
}

TEST_F("ReferenceFieldValue with ID can be roundtrip serialized", RefFixture) {
    ReferenceFieldValue ref_with_id(f.ref_type(), DocumentId("id:ns:" + doc_name + "::foo"));
    nbostream stream;
    serializeAndDeserialize(ref_with_id, stream, f.fixed_repo);
}

TEST_F("Empty ReferenceFieldValue serialization matches Java", RefFixture) {
    ReferenceFieldValue value(f.ref_type());
    f.verify_cross_language_serialization("empty_reference", value);
}

TEST_F("ReferenceFieldValue with ID serialization matches Java", RefFixture) {
    ReferenceFieldValue value(f.ref_type(), DocumentId("id:ns:" + doc_name + "::bar"));
    f.verify_cross_language_serialization("reference_with_id", value);
}

struct AssociatedDocumentRepoFixture {
    const DocumentType& doc_type{repo.getDocumentType()};
    DocumentId doc_id{"id:ns:" + doc_type.getName() + "::"};
    Document source_doc{doc_repo, doc_type, doc_id};

    std::unique_ptr<Document> roundtrip_serialize_source_document() {
        nbostream stream;
        VespaDocumentSerializer serializer(stream);
        serializer.write(source_doc);

        auto deserialized_doc = std::make_unique<Document>();
        VespaDocumentDeserializer deserializer(repo, stream, serialization_version);
        deserializer.read(*deserialized_doc);
        return deserialized_doc;
    }
};

TEST_F("Deserializing non-empty document associates correct repo with document instance", AssociatedDocumentRepoFixture) {
    f.source_doc.setValue(f.doc_type.getField("header field"), IntFieldValue(42));
    auto deserialized_doc = f.roundtrip_serialize_source_document();
    EXPECT_EQUAL(&doc_repo, deserialized_doc->getRepo());
}

TEST_F("Deserializing empty document associates correct repo with document instance", AssociatedDocumentRepoFixture) {
    auto deserialized_doc = f.roundtrip_serialize_source_document();
    EXPECT_EQUAL(&doc_repo, deserialized_doc->getRepo());
}

}  // namespace

TEST_MAIN() { TEST_RUN_ALL(); }
