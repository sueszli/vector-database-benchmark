/*
 * Copyright (c) 2015-2021 Nicholas Fraser and the MPack authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "test-file.h"
#include "test-write.h"
#include "test-reader.h"
#include "test-node.h"

#if MPACK_STDIO

// the file tests currently all require the writer, since it
// is used to write the test data that is read back.
#if MPACK_WRITER

#ifdef _WIN32
#include <direct.h>
#define test_mkdir(dir, mode) ((void)mode, _mkdir(dir))
#define test_rmdir _rmdir
#else
#include <unistd.h>
#define test_mkdir mkdir
#define test_rmdir rmdir
#endif

#define MESSAGEPACK_FILES_PATH "test/messagepack/"
#define PSEUDOJSON_FILES_PATH "test/pseudojson/"

#if MPACK_EXTENSIONS
#define TEST_FILE_MESSAGEPACK (MESSAGEPACK_FILES_PATH "test-file-ext.mp")
#define TEST_FILE_PSEUDOJSON (PSEUDOJSON_FILES_PATH "test-file-ext.debug")
#else
#define TEST_FILE_MESSAGEPACK (MESSAGEPACK_FILES_PATH "test-file-noext.mp")
#define TEST_FILE_PSEUDOJSON (PSEUDOJSON_FILES_PATH "test-file-noext.debug")
#endif

static const char* test_blank_filename = "mpack-test-blank-file";
static const char* test_filename = "mpack-test-file";
static const char* test_dir = "mpack-test-dir";

static const int nesting_depth = 150;
static const char* quick_brown_fox = "The quick brown fox jumps over a lazy dog.";

static char* test_file_fetch(const char* filename, size_t* out_size) {
    *out_size = 0;

    // open the file
    FILE* file = fopen(filename, "rb");
    if (!file) {
        TEST_TRUE(false, "missing file %s", filename);
        return NULL;
    }

    // get the file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size < 0) {
        TEST_TRUE(false, "invalid file size %i for %s", (int)size, filename);
        fclose(file);
        return NULL;
    }

    // allocate the data
    if (size == 0) {
        fclose(file);
        return (char*)MPACK_MALLOC(1);
    }
    char* data = (char*)MPACK_MALLOC((size_t)size);

    // read the file
    long total = 0;
    while (total < size) {
        size_t count = fread(data + total, 1, (size_t)(size - total), file);
        if (count <= 0) {
            TEST_TRUE(false, "failed to read from file %s", filename);
            fclose(file);
            MPACK_FREE(data);
            return NULL;
        }
        total += (long)count;
    }

    fclose(file);
    *out_size = (size_t)size;
    return data;
}

static void test_file_write_bytes(mpack_writer_t* writer, mpack_tag_t tag) {
    mpack_write_tag(writer, tag);
    char buf[1024];
    memset(buf, 0, sizeof(buf));
    for (; tag.v.l > sizeof(buf); tag.v.l -= (uint32_t)sizeof(buf))
        mpack_write_bytes(writer, buf, sizeof(buf));
    mpack_write_bytes(writer, buf, tag.v.l);
    mpack_finish_type(writer, tag.type);
}

static void test_file_write_elements(mpack_writer_t* writer, mpack_tag_t tag) {
    mpack_write_tag(writer, tag);
    size_t i;
    for (i = 0; i < tag.v.n; ++i) {
        if (tag.type == mpack_type_map)
            mpack_write_nil(writer);
        mpack_write_nil(writer);
    }
    mpack_finish_type(writer, tag.type);
}

static void test_file_write_contents(mpack_writer_t* writer) {
    mpack_start_array(writer, 7);

    // write lipsum to test a large fill/seek
    mpack_write_cstr(writer, lipsum);

    // test compound types of various sizes
    mpack_start_array(writer, 5);
    test_file_write_bytes(writer, mpack_tag_str(0));
    test_file_write_bytes(writer, mpack_tag_str(MPACK_INT8_MAX));
    test_file_write_bytes(writer, mpack_tag_str(MPACK_UINT8_MAX));
    test_file_write_bytes(writer, mpack_tag_str(MPACK_UINT8_MAX + 1));
    test_file_write_bytes(writer, mpack_tag_str(MPACK_UINT16_MAX + 1));
    mpack_finish_array(writer);

    mpack_start_array(writer, 5);
    test_file_write_bytes(writer, mpack_tag_bin(0));
    test_file_write_bytes(writer, mpack_tag_bin(MPACK_INT8_MAX));
    test_file_write_bytes(writer, mpack_tag_bin(MPACK_UINT8_MAX));
    test_file_write_bytes(writer, mpack_tag_bin(MPACK_UINT8_MAX + 1));
    test_file_write_bytes(writer, mpack_tag_bin(MPACK_UINT16_MAX + 1));
    mpack_finish_array(writer);

    #if MPACK_EXTENSIONS
    mpack_start_array(writer, 10);
    test_file_write_bytes(writer, mpack_tag_ext(1, 0));
    test_file_write_bytes(writer, mpack_tag_ext(1, 1));
    test_file_write_bytes(writer, mpack_tag_ext(1, 2));
    test_file_write_bytes(writer, mpack_tag_ext(1, 4));
    test_file_write_bytes(writer, mpack_tag_ext(1, 8));
    test_file_write_bytes(writer, mpack_tag_ext(1, 16));
    test_file_write_bytes(writer, mpack_tag_ext(2, MPACK_INT8_MAX));
    test_file_write_bytes(writer, mpack_tag_ext(3, MPACK_UINT8_MAX));
    test_file_write_bytes(writer, mpack_tag_ext(4, MPACK_UINT8_MAX + 1));
    test_file_write_bytes(writer, mpack_tag_ext(5, MPACK_UINT16_MAX + 1));
    mpack_finish_array(writer);
    #else
    mpack_write_nil(writer);
    #endif

    mpack_start_array(writer, 5);
    test_file_write_elements(writer, mpack_tag_array(0));
    test_file_write_elements(writer, mpack_tag_array(MPACK_INT8_MAX));
    test_file_write_elements(writer, mpack_tag_array(MPACK_UINT8_MAX));
    test_file_write_elements(writer, mpack_tag_array(MPACK_UINT8_MAX + 1));
    test_file_write_elements(writer, mpack_tag_array(MPACK_UINT16_MAX + 1));
    mpack_finish_array(writer);

    mpack_start_array(writer, 5);
    test_file_write_elements(writer, mpack_tag_map(0));
    test_file_write_elements(writer, mpack_tag_map(MPACK_INT8_MAX));
    test_file_write_elements(writer, mpack_tag_map(MPACK_UINT8_MAX));
    test_file_write_elements(writer, mpack_tag_map(MPACK_UINT8_MAX + 1));
    test_file_write_elements(writer, mpack_tag_map(MPACK_UINT16_MAX + 1));
    mpack_finish_array(writer);

    // test deep nesting
    int i;
    for (i = 0; i < nesting_depth; ++i)
        mpack_start_array(writer, 1);
    mpack_write_nil(writer);
    for (i = 0; i < nesting_depth; ++i)
        mpack_finish_array(writer);

    mpack_finish_array(writer);
}

static void test_file_write_failures(void) {
    mpack_writer_t writer;

    // test invalid filename
    (void)test_mkdir(test_dir, 0700);
    mpack_writer_init_filename(&writer, test_dir);
    TEST_WRITER_DESTROY_ERROR(&writer, mpack_error_io);

    // test close and flush failure
    // (if we write more than libc's internal FILE buffer size, fwrite()
    // fails, otherwise fclose() fails. we test both here.)

    mpack_writer_init_filename(&writer, "/dev/full");
    mpack_write_cstr(&writer, quick_brown_fox);
    TEST_WRITER_DESTROY_ERROR(&writer, mpack_error_io);

    int count = MPACK_UINT16_MAX / 20;
    mpack_writer_init_filename(&writer, "/dev/full");
    mpack_start_array(&writer, (uint32_t)count);
    int i;
    for (i = 0; i < count; ++i)
        mpack_write_cstr(&writer, quick_brown_fox);
    mpack_finish_array(&writer);
    TEST_WRITER_DESTROY_ERROR(&writer, mpack_error_io);
}

static void test_file_write(void) {
    mpack_writer_t writer;
    mpack_writer_init_file(&writer, test_filename); // test the deprecated function
    TEST_TRUE(mpack_writer_error(&writer) == mpack_ok, "file open failed with %s",
            mpack_error_to_string(mpack_writer_error(&writer)));

    test_file_write_contents(&writer);
    TEST_WRITER_DESTROY_NOERROR(&writer);
}

static void test_file_write_helper_std_owned(void) {
    // test writing to a libc FILE, giving ownership
    FILE* file = test_fopen(test_filename, "wb");
    TEST_TRUE(file != NULL, "failed to open file for writing! filename %s", test_filename);

    mpack_writer_t writer;
    mpack_writer_init_stdfile(&writer, file, true);
    test_file_write_contents(&writer);
    TEST_WRITER_DESTROY_NOERROR(&writer);

    // the test harness will ensure no file is leaked
}

static void test_file_write_helper_std_unowned(void) {
    // test writing to a libc FILE, retaining ownership
    FILE* file = test_fopen(test_filename, "wb");
    TEST_TRUE(file != NULL, "failed to open file for writing! filename %s", test_filename);

    mpack_writer_t writer;
    mpack_writer_init_stdfile(&writer, file, false);
    test_file_write_contents(&writer);
    TEST_WRITER_DESTROY_NOERROR(&writer);

    // we retained ownership, so we close it ourselves
    test_fclose(file);
}

static bool test_file_write_failure(void) {

    // The write failure test may fail with either
    // mpack_error_memory or mpack_error_io. We write a
    // bunch of strs and bins to test the various expect
    // allocator modes.

    mpack_writer_t writer;
    mpack_writer_init_filename(&writer, test_filename);

    mpack_start_array(&writer, 2);
    mpack_start_array(&writer, 6);

    // write a large string near the start to cause a
    // more than double buffer size growth
    mpack_write_cstr(&writer, quick_brown_fox);

    mpack_write_cstr(&writer, "one");
    mpack_write_cstr(&writer, "two");
    mpack_write_cstr(&writer, "three");
    mpack_write_cstr(&writer, "four");
    mpack_write_cstr(&writer, "five");

    mpack_finish_array(&writer);

    // test deep nesting
    int i;
    for (i = 0; i < nesting_depth; ++i)
        mpack_start_array(&writer, 1);
    mpack_write_nil(&writer);
    for (i = 0; i < nesting_depth; ++i)
        mpack_finish_array(&writer);

    mpack_finish_array(&writer);

    mpack_error_t error = mpack_writer_destroy(&writer);
    if (error == mpack_error_io || error == mpack_error_memory)
        return false;
    TEST_TRUE(error == mpack_ok, "unexpected error state %i (%s)", (int)error,
            mpack_error_to_string(error));
    return true;

}

// compares the test filename to the expected debug output
static void test_compare_print(void) {
    size_t expected_size;
    char* expected_data = test_file_fetch(TEST_FILE_PSEUDOJSON, &expected_size);
    size_t actual_size;
    char* actual_data = test_file_fetch(test_filename, &actual_size);

    TEST_TRUE(actual_size == expected_size, "print length %i does not match expected length %i",
            (int)actual_size, (int)expected_size);
    TEST_TRUE(0 == memcmp(actual_data, expected_data, actual_size), "print does not match expected");

    MPACK_FREE(expected_data);
    MPACK_FREE(actual_data);
}

#if MPACK_READER && MPACK_DEBUG && MPACK_DOUBLE
static void test_print(void) {

    // miscellaneous print tests
    // (we're not actually checking the output; we just want to make
    // sure it doesn't crash under the below errors.)
    FILE* out = fopen(test_filename, "wb");
    mpack_print_data_to_file("\x91", 1, out); // truncated file
    mpack_print_data_to_file("\xa1", 1, out); // truncated str
    mpack_print_data_to_file("\x92\x00", 2, out); // truncated array
    mpack_print_data_to_file("\x81", 1, out); // truncated map key
    mpack_print_data_to_file("\x81\x00", 2, out); // truncated map value
    mpack_print_data_to_file("\x90\xc0", 2, out); // extra bytes
    mpack_print_data_to_file("\xca\x00\x00\x00\x00", 5, out); // float
    fclose(out);

    // print test string to stdout
    mpack_print("\xaatesting...", 11);

    // dump MessagePack to debug file

    size_t input_size;
    char* input_data = test_file_fetch(TEST_FILE_MESSAGEPACK, &input_size);

    out = fopen(test_filename, "wb");
    mpack_print_data_to_file(input_data, input_size, out);
    fclose(out);

    MPACK_FREE(input_data);
    test_compare_print();
}
#endif

#if MPACK_NODE && MPACK_DEBUG && MPACK_DOUBLE
static void test_node_print(void) {
    mpack_tree_t tree;

    // miscellaneous node print tests
    FILE* out = fopen(test_filename, "wb");
    mpack_tree_init(&tree, "\xca\x00\x00\x00\x00", 5); // float
    mpack_tree_parse(&tree);
    mpack_node_print_to_file(mpack_tree_root(&tree), out);
    mpack_tree_destroy(&tree);
    fclose(out);

    // print test string to stdout
    mpack_tree_init(&tree, "\xaatesting...", 11);
    mpack_tree_parse(&tree);
    mpack_node_print(mpack_tree_root(&tree));
    mpack_tree_destroy(&tree);

    // dump MessagePack to debug file

    mpack_tree_init_filename(&tree, TEST_FILE_MESSAGEPACK, 0);
    mpack_tree_parse(&tree);
    TEST_TRUE(mpack_ok == mpack_tree_error(&tree));

    out = fopen(test_filename, "wb");
    mpack_node_print_to_file(mpack_tree_root(&tree), out);
    fclose(out);

    TEST_TRUE(mpack_ok == mpack_tree_destroy(&tree));
    test_compare_print();
}
#endif

#if MPACK_READER
static void test_file_discard(void) {
    mpack_reader_t reader;
    mpack_reader_init_filename(&reader, test_filename);
    mpack_discard(&reader);
    TEST_READER_DESTROY_NOERROR(&reader);

    mpack_reader_init_filename(&reader, test_filename);
    reader.skip = NULL; // disable the skip callback to test skipping without it
    mpack_discard(&reader);
    TEST_READER_DESTROY_NOERROR(&reader);
}
#endif

#if MPACK_EXPECT
static void test_file_expect_bytes(mpack_reader_t* reader, mpack_tag_t tag) {
    mpack_expect_tag(reader, tag);
    TEST_TRUE(mpack_reader_error(reader) == mpack_ok, "got error %i (%s)", (int)mpack_reader_error(reader), mpack_error_to_string(mpack_reader_error(reader)));

    char expected[1024];
    memset(expected, 0, sizeof(expected));
    char buf[sizeof(expected)];
    while (tag.v.l > 0) {
        uint32_t count = (tag.v.l < (uint32_t)sizeof(buf)) ? tag.v.l : (uint32_t)sizeof(buf);
        mpack_read_bytes(reader, buf, count);
        TEST_TRUE(mpack_reader_error(reader) == mpack_ok, "got error %i (%s)", (int)mpack_reader_error(reader), mpack_error_to_string(mpack_reader_error(reader)));
        TEST_TRUE(memcmp(buf, expected, count) == 0, "data does not match!");
        tag.v.l -= count;
    }

    mpack_done_type(reader, tag.type);
}

static void test_file_expect_elements(mpack_reader_t* reader, mpack_tag_t tag) {
    mpack_expect_tag(reader, tag);
    size_t i;
    for (i = 0; i < tag.v.n; ++i) {
        if (tag.type == mpack_type_map)
            mpack_expect_nil(reader);
        mpack_expect_nil(reader);
    }
    mpack_done_type(reader, tag.type);
}

static void test_file_read_contents(mpack_reader_t* reader) {
    TEST_TRUE(mpack_reader_error(reader) == mpack_ok, "file open failed with %s",
            mpack_error_to_string(mpack_reader_error(reader)));

    TEST_TRUE(7 == mpack_expect_array(reader));

    // test matching a cstr larger than the buffer size
    mpack_expect_cstr_match(reader, lipsum);
    TEST_TRUE(mpack_reader_error(reader) == mpack_ok, "failed to match huge string!");

    TEST_TRUE(5 == mpack_expect_array(reader));
    test_file_expect_bytes(reader, mpack_tag_str(0));
    test_file_expect_bytes(reader, mpack_tag_str(MPACK_INT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_str(MPACK_UINT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_str(MPACK_UINT8_MAX + 1));
    test_file_expect_bytes(reader, mpack_tag_str(MPACK_UINT16_MAX + 1));
    mpack_done_array(reader);

    TEST_TRUE(5 == mpack_expect_array(reader));
    test_file_expect_bytes(reader, mpack_tag_bin(0));
    test_file_expect_bytes(reader, mpack_tag_bin(MPACK_INT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_bin(MPACK_UINT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_bin(MPACK_UINT8_MAX + 1));
    test_file_expect_bytes(reader, mpack_tag_bin(MPACK_UINT16_MAX + 1));
    mpack_done_array(reader);

    #if MPACK_EXTENSIONS
    TEST_TRUE(10 == mpack_expect_array(reader));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 0));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 1));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 2));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 4));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 8));
    test_file_expect_bytes(reader, mpack_tag_ext(1, 16));
    test_file_expect_bytes(reader, mpack_tag_ext(2, MPACK_INT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_ext(3, MPACK_UINT8_MAX));
    test_file_expect_bytes(reader, mpack_tag_ext(4, MPACK_UINT8_MAX + 1));
    test_file_expect_bytes(reader, mpack_tag_ext(5, MPACK_UINT16_MAX + 1));
    mpack_done_array(reader);
    #else
    mpack_expect_nil(reader);
    #endif

    TEST_TRUE(5 == mpack_expect_array(reader));
    test_file_expect_elements(reader, mpack_tag_array(0));
    test_file_expect_elements(reader, mpack_tag_array(MPACK_INT8_MAX));
    test_file_expect_elements(reader, mpack_tag_array(MPACK_UINT8_MAX));
    test_file_expect_elements(reader, mpack_tag_array(MPACK_UINT8_MAX + 1));
    test_file_expect_elements(reader, mpack_tag_array(MPACK_UINT16_MAX + 1));
    mpack_done_array(reader);

    TEST_TRUE(5 == mpack_expect_array(reader));
    test_file_expect_elements(reader, mpack_tag_map(0));
    test_file_expect_elements(reader, mpack_tag_map(MPACK_INT8_MAX));
    test_file_expect_elements(reader, mpack_tag_map(MPACK_UINT8_MAX));
    test_file_expect_elements(reader, mpack_tag_map(MPACK_UINT8_MAX + 1));
    test_file_expect_elements(reader, mpack_tag_map(MPACK_UINT16_MAX + 1));
    mpack_done_array(reader);

    int i;
    for (i = 0; i < nesting_depth; ++i)
        mpack_expect_array_match(reader, 1);
    mpack_expect_nil(reader);
    for (i = 0; i < nesting_depth; ++i)
        mpack_done_array(reader);

    mpack_done_array(reader);
}

static void test_file_read_missing(void) {
    // test missing file
    mpack_reader_t reader;
    mpack_reader_init_filename(&reader, "invalid-filename");
    TEST_READER_DESTROY_ERROR(&reader, mpack_error_io);
}

static void test_file_read_helper(void) {
    // test reading with the default file reader
    mpack_reader_t reader;
    mpack_reader_init_file(&reader, test_filename); // test the deprecated function
    test_file_read_contents(&reader);
    TEST_READER_DESTROY_NOERROR(&reader);
}

static void test_file_read_helper_std_owned(void) {
    // test reading from a libc FILE, giving ownership
    FILE* file = test_fopen(test_filename, "rb");
    TEST_TRUE(file != NULL, "failed to open file! filename %s", test_filename);

    mpack_reader_t reader;
    mpack_reader_init_stdfile(&reader, file, true);
    test_file_read_contents(&reader);
    TEST_READER_DESTROY_NOERROR(&reader);

    // the test harness will ensure no file is leaked
}

static void test_file_read_helper_std_unowned(void) {
    // test reading from a libc FILE, retaining ownership
    FILE* file = test_fopen(test_filename, "rb");
    TEST_TRUE(file != NULL, "failed to open file! filename %s", test_filename);

    mpack_reader_t reader;
    mpack_reader_init_stdfile(&reader, file, false);
    test_file_read_contents(&reader);
    TEST_READER_DESTROY_NOERROR(&reader);

    // we retained ownership, so we close it ourselves
    test_fclose(file);
}

typedef struct test_file_streaming_t {
    FILE* file;
    size_t read_size;
} test_file_streaming_t;

static size_t test_file_read_streaming_fill(mpack_reader_t* reader, char* buffer, size_t count) {
    test_file_streaming_t* context = (test_file_streaming_t*)reader->context;
    if (count > context->read_size)
        count = context->read_size;
    return fread((void*)buffer, 1, count, context->file);
}

static void test_file_read_streaming(void) {
    // We test reading from a file using a streaming function
    // that returns a small number of bytes each time (as though
    // it is slowly receiving data through a socket.) This tests
    // that the reader correctly handles streams, and that it
    // can continue asking for data even when it needs more bytes
    // than read by a single call to the fill function.

    size_t sizes[] = {1, 2, 3, 5, 7, 11};
    size_t i;
    for (i = 0; i < sizeof(sizes) / sizeof(*sizes); ++i) {

        FILE* file = fopen(test_filename, "rb");
        TEST_TRUE(file != NULL, "failed to open file! filename %s", test_filename);

        test_file_streaming_t context = {file, sizes[i]};
        mpack_reader_t reader;
        char buffer[MPACK_READER_MINIMUM_BUFFER_SIZE];
        mpack_reader_init(&reader, buffer, sizeof(buffer), 0);
        mpack_reader_set_context(&reader, &context);
        mpack_reader_set_fill(&reader, &test_file_read_streaming_fill);

        test_file_read_contents(&reader);
        TEST_READER_DESTROY_NOERROR(&reader);
        fclose(file);
    }
}

static bool test_file_expect_failure(void) {

    // The expect failure test may fail with either
    // mpack_error_memory or mpack_error_io.

    mpack_reader_t reader;

    #define TEST_POSSIBLE_FAILURE() do { \
        mpack_error_t error = mpack_reader_error(&reader); \
        if (error == mpack_error_memory || error == mpack_error_io) { \
            mpack_reader_destroy(&reader); \
            return false; \
        } \
    } while (0)

    mpack_reader_init_filename(&reader, test_filename);
    mpack_expect_array_match(&reader, 2);

    uint32_t count;
    char** strings = mpack_expect_array_alloc(&reader, char*, 50, &count);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(strings != NULL);
    TEST_TRUE(count == 6);
    MPACK_FREE(strings);

    char* str = mpack_expect_cstr_alloc(&reader, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    if (str) {
        TEST_TRUE(strcmp(str, quick_brown_fox) == 0);
        MPACK_FREE(str);
    }

    str = mpack_expect_utf8_cstr_alloc(&reader, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    if (str) {
        TEST_TRUE(strcmp(str, "one") == 0);
        MPACK_FREE(str);
    }

    str = mpack_expect_cstr_alloc(&reader, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    if (str) {
        TEST_TRUE(strcmp(str, "two") == 0);
        MPACK_FREE(str);
    }

    str = mpack_expect_utf8_cstr_alloc(&reader, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    if (str) {
        TEST_TRUE(strcmp(str, "three") == 0);
        MPACK_FREE(str);
    }

    mpack_discard(&reader);
    mpack_discard(&reader);
    mpack_done_array(&reader);

    mpack_discard(&reader); // discard the deep nested arrays
    mpack_done_array(&reader);

    #undef TEST_POSSIBLE_FAILURE

    mpack_error_t error = mpack_reader_destroy(&reader);
    if (error == mpack_error_io || error == mpack_error_memory)
        return false;
    TEST_TRUE(error == mpack_ok, "unexpected error state %i (%s)", (int)error,
            mpack_error_to_string(error));
    return true;

}

static void test_file_read_eof(void) {
    mpack_reader_t reader;
    mpack_reader_init_filename(&reader, test_filename);
    TEST_TRUE(mpack_reader_error(&reader) == mpack_ok, "file open failed with %s",
            mpack_error_to_string(mpack_reader_error(&reader)));

    while (mpack_reader_error(&reader) != mpack_error_eof)
        mpack_discard(&reader);

    mpack_error_t error = mpack_reader_destroy(&reader);
    TEST_TRUE(error == mpack_error_eof, "unexpected error state %i (%s)", (int)error,
            mpack_error_to_string(error));
}

#endif

#if MPACK_NODE
static void test_file_node_bytes(mpack_node_t node, mpack_tag_t tag) {
    TEST_TRUE(mpack_tag_equal(tag, mpack_node_tag(node)));
    const char* data = mpack_node_data(node);
    uint32_t length = mpack_node_data_len(node);
    TEST_TRUE(mpack_node_error(node) == mpack_ok);

    char expected[1024];
    memset(expected, 0, sizeof(expected));
    while (length > 0) {
        uint32_t count = (length < (uint32_t)sizeof(expected)) ? length : (uint32_t)sizeof(expected);
        TEST_TRUE(memcmp(data, expected, count) == 0);
        length -= count;
        data += count;
    }
}

static void test_file_node_elements(mpack_node_t node, mpack_tag_t tag) {
    TEST_TRUE(mpack_tag_equal(tag, mpack_node_tag(node)));
    size_t i;
    for (i = 0; i < tag.v.n; ++i) {
        if (tag.type == mpack_type_map) {
            mpack_node_nil(mpack_node_map_key_at(node, i));
            mpack_node_nil(mpack_node_map_value_at(node, i));
        } else {
            mpack_node_nil(mpack_node_array_at(node, i));
        }
    }
}

static void test_file_node_contents(mpack_node_t root) {
    TEST_TRUE(mpack_node_array_length(root) == 7);

    mpack_node_t lipsum_node = mpack_node_array_at(root, 0);
    const char* lipsum_str = mpack_node_str(lipsum_node);
    TEST_TRUE(lipsum_str != NULL);
    if (lipsum_str) {
        TEST_TRUE(mpack_node_strlen(lipsum_node) == strlen(lipsum));
        TEST_TRUE(memcmp(lipsum_str, lipsum, strlen(lipsum)) == 0);
    }

    mpack_node_t node = mpack_node_array_at(root, 1);
    TEST_TRUE(mpack_node_array_length(node) == 5);
    test_file_node_bytes(mpack_node_array_at(node, 0), mpack_tag_str(0));
    test_file_node_bytes(mpack_node_array_at(node, 1), mpack_tag_str(MPACK_INT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 2), mpack_tag_str(MPACK_UINT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 3), mpack_tag_str(MPACK_UINT8_MAX + 1));
    test_file_node_bytes(mpack_node_array_at(node, 4), mpack_tag_str(MPACK_UINT16_MAX + 1));

    node = mpack_node_array_at(root, 2);
    TEST_TRUE(5 == mpack_node_array_length(node));
    test_file_node_bytes(mpack_node_array_at(node, 0), mpack_tag_bin(0));
    test_file_node_bytes(mpack_node_array_at(node, 1), mpack_tag_bin(MPACK_INT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 2), mpack_tag_bin(MPACK_UINT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 3), mpack_tag_bin(MPACK_UINT8_MAX + 1));
    test_file_node_bytes(mpack_node_array_at(node, 4), mpack_tag_bin(MPACK_UINT16_MAX + 1));

    node = mpack_node_array_at(root, 3);
    #if MPACK_EXTENSIONS
    TEST_TRUE(10 == mpack_node_array_length(node));
    test_file_node_bytes(mpack_node_array_at(node, 0), mpack_tag_ext(1, 0));
    test_file_node_bytes(mpack_node_array_at(node, 1), mpack_tag_ext(1, 1));
    test_file_node_bytes(mpack_node_array_at(node, 2), mpack_tag_ext(1, 2));
    test_file_node_bytes(mpack_node_array_at(node, 3), mpack_tag_ext(1, 4));
    test_file_node_bytes(mpack_node_array_at(node, 4), mpack_tag_ext(1, 8));
    test_file_node_bytes(mpack_node_array_at(node, 5), mpack_tag_ext(1, 16));
    test_file_node_bytes(mpack_node_array_at(node, 6), mpack_tag_ext(2, MPACK_INT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 7), mpack_tag_ext(3, MPACK_UINT8_MAX));
    test_file_node_bytes(mpack_node_array_at(node, 8), mpack_tag_ext(4, MPACK_UINT8_MAX + 1));
    test_file_node_bytes(mpack_node_array_at(node, 9), mpack_tag_ext(5, MPACK_UINT16_MAX + 1));
    #else
    mpack_node_nil(node);
    #endif

    node = mpack_node_array_at(root, 4);
    TEST_TRUE(5 == mpack_node_array_length(node));
    test_file_node_elements(mpack_node_array_at(node, 0), mpack_tag_array(0));
    test_file_node_elements(mpack_node_array_at(node, 1), mpack_tag_array(MPACK_INT8_MAX));
    test_file_node_elements(mpack_node_array_at(node, 2), mpack_tag_array(MPACK_UINT8_MAX));
    test_file_node_elements(mpack_node_array_at(node, 3), mpack_tag_array(MPACK_UINT8_MAX + 1));
    test_file_node_elements(mpack_node_array_at(node, 4), mpack_tag_array(MPACK_UINT16_MAX + 1));

    node = mpack_node_array_at(root, 5);
    TEST_TRUE(5 == mpack_node_array_length(node));
    test_file_node_elements(mpack_node_array_at(node, 0), mpack_tag_map(0));
    test_file_node_elements(mpack_node_array_at(node, 1), mpack_tag_map(MPACK_INT8_MAX));
    test_file_node_elements(mpack_node_array_at(node, 2), mpack_tag_map(MPACK_UINT8_MAX));
    test_file_node_elements(mpack_node_array_at(node, 3), mpack_tag_map(MPACK_UINT8_MAX + 1));
    test_file_node_elements(mpack_node_array_at(node, 4), mpack_tag_map(MPACK_UINT16_MAX + 1));

    node = mpack_node_array_at(root, 6);
    int i;
    for (i = 0; i < nesting_depth; ++i)
        node = mpack_node_array_at(node, 0);
    TEST_TRUE(mpack_ok == mpack_node_error(node));
    mpack_node_nil(node);
}

static void test_file_tree_successful_parse(mpack_tree_t* tree) {
    mpack_tree_parse(tree);
    TEST_TRUE(mpack_tree_error(tree) == mpack_ok, "file tree parsing failed: %s",
            mpack_error_to_string(mpack_tree_error(tree)));
    test_file_node_contents(mpack_tree_root(tree));
    mpack_error_t error = mpack_tree_destroy(tree);
    TEST_TRUE(error == mpack_ok, "file tree failed with error %s", mpack_error_to_string(error));
}

static void test_file_node(void) {
    mpack_tree_t tree;

    // test maximum size
    mpack_tree_init_file(&tree, test_filename, 100);
    TEST_TREE_DESTROY_ERROR(&tree, mpack_error_too_big);

    // test blank file
    mpack_tree_init_file(&tree, test_blank_filename, 0);
    TEST_TREE_DESTROY_ERROR(&tree, mpack_error_invalid);

    // test successful parse from filename
    mpack_tree_init_file(&tree, test_filename, 0); // test the deprecated function
    test_file_tree_successful_parse(&tree);

    // test file size out of bounds
    #if MPACK_DEBUG
    if (sizeof(size_t) >= sizeof(long)) {
        TEST_BREAK((mpack_tree_init_filename(&tree, "invalid-filename", ((size_t)LONG_MAX) + 1), true));
        TEST_TREE_DESTROY_ERROR(&tree, mpack_error_bug);
    }
    #endif

    // test missing file
    mpack_tree_init_filename(&tree, "invalid-filename", 0);
    TEST_TREE_DESTROY_ERROR(&tree, mpack_error_io);

    // test successful parse from FILE with auto-close
    FILE* file = test_fopen(test_filename, "rb");
    TEST_TRUE(file != NULL);
    mpack_tree_init_stdfile(&tree, file, 0, true);
    test_file_tree_successful_parse(&tree);

    // test successful parse from FILE with no-close
    file = test_fopen(test_filename, "rb");
    TEST_TRUE(file != NULL);
    mpack_tree_init_stdfile(&tree, file, 0, false);
    test_file_tree_successful_parse(&tree);
    test_fclose(file);
}

typedef struct test_file_stream_t {
    size_t length;
    char* data;
    size_t pos;
    size_t step;
} test_file_stream_t;

// The test stream reader loops over the data in the test file. It returns at
// most the step size so we can test reading from very small to very large
// chunks.
static size_t test_file_stream_read(mpack_tree_t* tree, char* buffer, size_t count) {
    test_file_stream_t* stream = (test_file_stream_t*)tree->context;

    if (count > stream->step)
        count = stream->step;

    size_t left = count;
    while (left > 0) {
        size_t n = stream->length - stream->pos;
        if (n > left)
            n = left;

        mpack_memcpy(buffer, stream->data + stream->pos, n);

        buffer += n;
        stream->pos += n;
        left -= n;

        if (stream->pos == stream->length)
            stream->pos = 0;
    }

    return count;
}

static void test_file_node_stream(void) {
    test_file_stream_t stream;
    stream.data = test_file_fetch(test_filename, &stream.length);

    size_t steps[] = {11, 23, 32, 127, 369, 4096, SIZE_MAX};

    size_t i;
    for (i = 0; i < sizeof(steps) / sizeof(steps[0]); ++i) {
        stream.pos = 0;
        stream.step = steps[i];

        // We use a max_size a bit larger than the file, that way some extra
        // data is read from the next tree.
        size_t max_size = stream.length * 4 / 3;

        mpack_tree_t tree;
        mpack_tree_init_stream(&tree, &test_file_stream_read, &stream, max_size, max_size);
        TEST_TRUE(mpack_tree_error(&tree) == mpack_ok, "tree initialization failed: %s",
                mpack_error_to_string(mpack_tree_error(&tree)));

        // We try parsing the same tree a dozen times repeatedly with this step size.
        int j;
        for (j = 0; j < 12; ++j) {
            mpack_tree_parse(&tree);
            TEST_TRUE(mpack_tree_error(&tree) == mpack_ok, "tree parsing failed: %s",
                    mpack_error_to_string(mpack_tree_error(&tree)));
            test_file_node_contents(mpack_tree_root(&tree));
            TEST_TRUE(mpack_tree_error(&tree) == mpack_ok, "tree contents failed: %s",
                    mpack_error_to_string(mpack_tree_error(&tree)));
        }

        mpack_error_t error = mpack_tree_destroy(&tree);
        TEST_TRUE(error == mpack_ok, "tree stream failed with error %s", mpack_error_to_string(error));
    }

    MPACK_FREE(stream.data);
}

static bool test_file_node_failure(void) {

    // The node failure test may fail with either
    // mpack_error_memory or mpack_error_io.

    mpack_tree_t tree;

    #define TEST_POSSIBLE_FAILURE() do { \
        mpack_error_t error = mpack_tree_error(&tree); \
        TEST_TRUE(test_tree_error == error); \
        if (error == mpack_error_memory || error == mpack_error_io) { \
            test_tree_error = mpack_ok; \
            mpack_tree_destroy(&tree); \
            return false; \
        } \
    } while (0)

    mpack_tree_init_filename(&tree, test_filename, 0);
    mpack_tree_parse(&tree);
    if (mpack_tree_error(&tree) == mpack_error_memory || mpack_tree_error(&tree) == mpack_error_io) {
        mpack_tree_destroy(&tree);
        return false;
    }
    mpack_tree_set_error_handler(&tree, test_tree_error_handler);


    mpack_node_t root = mpack_tree_root(&tree);

    mpack_node_t strings = mpack_node_array_at(root, 0);
    size_t length = mpack_node_array_length(strings);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(6 == length);

    mpack_node_t node = mpack_node_array_at(strings, 0);
    char* str = mpack_node_data_alloc(node, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    const char* expected = quick_brown_fox;
    TEST_TRUE(mpack_node_strlen(node) == strlen(expected));
    if (str) {
        TEST_TRUE(memcmp(str, expected, mpack_node_strlen(node)) == 0);
        MPACK_FREE(str);
    }

    node = mpack_node_array_at(strings, 1);

    str = mpack_node_cstr_alloc(node, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    expected = "one";
    if (str) {
        TEST_TRUE(strlen(str) == strlen(expected));
        TEST_TRUE(strcmp(str, expected) == 0);
        MPACK_FREE(str);
    }

    str = mpack_node_utf8_cstr_alloc(node, 100);
    TEST_POSSIBLE_FAILURE();
    TEST_TRUE(str != NULL);
    if (str) {
        TEST_TRUE(strlen(str) == strlen(expected));
        TEST_TRUE(strcmp(str, expected) == 0);
        MPACK_FREE(str);
    }

    node = mpack_node_array_at(root, 1);
    int i;
    for (i = 0; i < nesting_depth; ++i)
        node = mpack_node_array_at(node, 0);
    TEST_TRUE(mpack_ok == mpack_node_error(node));
    mpack_node_nil(node);

    #undef TEST_POSSIBLE_FAILURE

    mpack_error_t error = mpack_tree_destroy(&tree);
    if (error == mpack_error_io || error == mpack_error_memory)
        return false;
    TEST_TRUE(error == mpack_ok, "unexpected error state %i (%s)", (int)error,
            mpack_error_to_string(error));
    return true;

}
#endif

void test_file(void) {
    // write a blank file for test purposes
    FILE* blank = fopen(test_blank_filename, "wb");
    fclose(blank);

    #if MPACK_READER && MPACK_DEBUG && MPACK_DOUBLE
    test_print();
    #endif
    #if MPACK_NODE && MPACK_DEBUG && MPACK_DOUBLE
    test_node_print();
    #endif

    #if MPACK_WRITER
    test_file_write_failures();
    test_file_write_helper_std_owned();
    test_file_write_helper_std_unowned();
    test_file_write();
    #endif

    #if MPACK_READER
    test_file_discard();
    #endif
    #if MPACK_EXPECT
    test_file_read_missing();
    test_file_read_helper();
    test_file_read_helper_std_owned();
    test_file_read_helper_std_unowned();
    test_file_read_streaming();
    test_file_read_eof();
    #endif
    #if MPACK_NODE
    test_file_node();
    test_file_node_stream();
    #endif

    #if MPACK_WRITER
    test_system_fail_until_ok(&test_file_write_failure);
    #endif
    #if MPACK_EXPECT
    test_system_fail_until_ok(&test_file_expect_failure);
    #endif
    #if MPACK_NODE
    test_system_fail_until_ok(&test_file_node_failure);
    #endif

    TEST_TRUE(remove(test_filename) == 0, "failed to delete %s", test_filename);
    TEST_TRUE(remove(test_blank_filename) == 0, "failed to delete %s", test_blank_filename);
    TEST_TRUE(test_rmdir(test_dir) == 0, "failed to delete %s", test_dir);

    (void)&test_compare_print;
}

#else

void test_file(void) {
    // if we don't have the writer, nothing to do
}

#endif
#endif

