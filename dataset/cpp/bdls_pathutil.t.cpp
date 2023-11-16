// bdls_pathutil.t.cpp                                                -*-C++-*-

// ----------------------------------------------------------------------------
//                                   NOTICE
//
// This component is not up to date with current BDE coding standards, and
// should not be used as an example for new development.
// ----------------------------------------------------------------------------

#include <bdls_pathutil.h>

#include <bslim_testutil.h>

#include <bslma_deallocatorguard.h>
#include <bslma_testallocator.h>

#include <bsls_assert.h>
#include <bsls_asserttest.h>
#include <bsls_platform.h>
#include <bsls_review.h>

#include <bsl_cstdlib.h>
#include <bsl_cstring.h>
#include <bsl_iostream.h>
#include <bsl_vector.h>

using namespace BloombergLP;
using namespace bsl;  // automatically added by script

//=============================================================================
//                              TEST PLAN
//-----------------------------------------------------------------------------
//                              Overview
//                              --------
// 'bdlt::PathUtil' provides a suite of functions for manipulating strings that
// represent paths in the filesystem.  This test driver falls short of accepted
// standards and needs to be rewritten.
//
//-----------------------------------------------------------------------------
// CLASS METHODS
// [ 4] int appendIfValid(bsl::string *, const bsl::string_view& );
// [ 5] void splitFilename(string_view*, string_view*, const string_view&, int);
// ----------------------------------------------------------------------------
// [ 6] USAGE EXAMPLE

// ============================================================================
//                     STANDARD BDE ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        cout << "Error " __FILE__ "(" << line << "): " << message
             << "    (failed)" << endl;

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace

// ============================================================================
//               STANDARD BDE TEST DRIVER MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT       BSLIM_TESTUTIL_ASSERT
#define ASSERTV      BSLIM_TESTUTIL_ASSERTV

#define LOOP_ASSERT  BSLIM_TESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLIM_TESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLIM_TESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLIM_TESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLIM_TESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLIM_TESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLIM_TESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLIM_TESTUTIL_LOOP6_ASSERT

#define Q            BSLIM_TESTUTIL_Q   // Quote identifier literally.
#define P            BSLIM_TESTUTIL_P   // Print identifier and value.
#define P_           BSLIM_TESTUTIL_P_  // P(X) without '\n'.
#define T_           BSLIM_TESTUTIL_T_  // Print a tab (w/o newline).
#define L_           BSLIM_TESTUTIL_L_  // current Line number

// ============================================================================
//                     NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------
#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

// ============================================================================
//                   GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------

typedef bdls::PathUtil Obj;

struct Parameters {
    int d_line;
    const char* d_path;
    const char* d_root;
    const char* d_leaf;
    const char* d_dirName;
    int d_numLeaves;
    bool d_isRelative;
    bool d_isValidRaw;
} parameters[] = {
#ifdef BSLS_PLATFORM_OS_WINDOWS
   {L_, "", "", 0, 0,
         0, 1, 0},
   {L_, "\\", "\\", 0, 0,
         0, 0, 0},
   {L_, "c:", "c:", 0, 0,
         0, 0, 1},
   {L_, "c:\\", "c:\\", 0, 0,
         0, 0, 1},
   {L_, "c:\\\\", "c:\\\\", 0, 0,
         0, 0, 1},
   {L_, "c:\\one", "c:\\", "one", "c:\\",
         1, 0, 1},
   {L_, "c:\\one\\", "c:\\", "one", "c:\\",
         1, 0, 0},
   {L_, "c:\\one\\two", "c:\\", "two", "c:\\one",
         2, 0, 1},
   {L_, "c:\\one\\two\\", "c:\\", "two", "c:\\one",
         2, 0, 0},
   {L_, "c:\\\\one", "c:\\\\", "one", "c:\\\\",
         1, 0, 1},
   {L_, "c:\\\\one\\", "c:\\\\", "one", "c:\\\\",
         1, 0, 0},
   {L_, "c:\\\\one\\two", "c:\\\\", "two", "c:\\\\one",
         2, 0, 1},
   {L_, "c:\\\\one\\two\\", "c:\\\\", "two", "c:\\\\one",
         2, 0, 0},
   {L_, "c:one", "c:", "one", "c:",
         1, 0, 1},
   {L_, "one", "", "one", "",
         1, 1, 1},
   {L_, "one\\", "", "one", "",
         1, 1, 0},
   {L_, "one\\two", "", "two", "one",
         2, 1, 1},
   {L_, "one\\two\\", "", "two", "one",
         2, 1, 0},
   {L_, "one\\two\\\\", "", "two", "one",
         2, 1, 1},
   {L_, "c:one\\two\\\\", "c:", "two", "c:one",
         2, 0, 0},
   {L_, "\\\\server\\dir", "\\\\server\\dir", 0, 0,
         0, 0, 1},
   {L_, "\\\\server\\dir\\", "\\\\server\\dir\\", 0, 0,
         0, 0, 1},
   {L_, "\\\\server\\dir\\\\", "\\\\server\\dir\\\\", 0, 0,
         0, 0, 1},
   {L_, "\\\\server\\dir\\one",
        "\\\\server\\dir\\", "one", "\\\\server\\dir\\",
        1, 0, 1},
   {L_, "\\\\server\\dir\\one\\",
        "\\\\server\\dir\\", "one", "\\\\server\\dir\\",
         1, 0, 0},
   {L_, "\\\\server\\dir\\one\\two",
         "\\\\server\\dir\\", "two", "\\\\server\\dir\\one",
         2, 0, 1},
   {L_, "\\\\.\\dir\\one\\two\\", "\\\\.\\dir\\", "two", "\\\\.\\dir\\one",
         2, 0, 0},
   {L_, "\\\\server\\dir\\\\one",
        "\\\\server\\dir\\\\", "one", "\\\\server\\dir\\\\",
         1, 0, 1},
   {L_, "\\\\server\\dir\\\\one\\",
        "\\\\server\\dir\\\\", "one", "\\\\server\\dir\\\\",
         1, 0, 0},
   {L_, "\\\\server\\dir\\\\one\\two",
        "\\\\server\\dir\\\\", "two", "\\\\server\\dir\\\\one",
         2, 0, 1},
   {L_, "\\\\server\\dir\\\\one\\two\\",
        "\\\\server\\dir\\\\", "two", "\\\\server\\dir\\\\one",
         2, 0, 0},
   {L_, "\\\\server\\dir\\one\\two\\\\",
        "\\\\server\\dir\\", "two", "\\\\server\\dir\\one",
         2, 0, 0},
   {L_, "\\\\?\\UNC\\server\\dir\\one\\two\\\\",
        "\\\\?\\UNC\\server\\dir\\", "two", "\\\\?\\UNC\\server\\dir\\one",
         2, 0, 0},
   {L_, "\\\\?\\c:one\\two\\\\", "\\\\?\\c:", "two", "\\\\?\\c:one",
         2, 0, 0},
   {L_, "\\\\?\\c:\\one\\two\\\\", "\\\\?\\c:\\", "two", "\\\\?\\c:\\one",
         2, 0, 0},
   {L_, "/", "/", 0, 0,
         0, 0, 0},
   {L_, "c:/", "c:/", 0, 0,
         0, 0, 1},
   {L_, "c://", "c://", 0, 0,
         0, 0, 1},
   {L_, "c:/one", "c:/", "one", "c:/",
         1, 0, 1},
   {L_, "c:/one/", "c:/", "one", "c:/",
         1, 0, 0},
   {L_, "c:/one/two", "c:/", "two", "c:/one",
         2, 0, 1},
   {L_, "c:/one/two/", "c:/", "two", "c:/one",
         2, 0, 0},
   {L_, "c://one", "c://", "one", "c://",
         1, 0, 1},
   {L_, "c://one/", "c://", "one", "c://",
         1, 0, 0},
   {L_, "c://one/two", "c://", "two", "c://one",
         2, 0, 1},
   {L_, "c://one/two/", "c://", "two", "c://one",
         2, 0, 0},
   {L_, "one/", "", "one", "",
         1, 1, 0},
   {L_, "one/two", "", "two", "one",
         2, 1, 1},
   {L_, "one/two/", "", "two", "one",
         2, 1, 0},
   {L_, "one/two//", "", "two", "one",
         2, 1, 1},
   {L_, "c:one/two//", "c:", "two", "c:one",
         2, 0, 0},
   {L_, "\\\\server/dir", "\\\\server/dir", 0, 0,
         0, 0, 1},
   {L_, "\\\\server/dir/", "\\\\server/dir/", 0, 0,
         0, 0, 1},
   {L_, "\\\\server/dir//", "\\\\server/dir//", 0, 0,
         0, 0, 1},
   {L_, "\\\\server/dir/one",
        "\\\\server/dir/", "one", "\\\\server/dir/",
        1, 0, 1},
   {L_, "\\\\server/dir/one/",
        "\\\\server/dir/", "one", "\\\\server/dir/",
         1, 0, 0},
   {L_, "\\\\server/dir/one/two",
         "\\\\server/dir/", "two", "\\\\server/dir/one",
         2, 0, 1},
   {L_, "\\\\./dir/one/two/", "\\\\./dir/", "two", "\\\\./dir/one",
         2, 0, 0},
   {L_, "\\\\server/dir//one",
        "\\\\server/dir//", "one", "\\\\server/dir//",
         1, 0, 1},
   {L_, "\\\\server/dir//one/",
        "\\\\server/dir//", "one", "\\\\server/dir//",
         1, 0, 0},
   {L_, "\\\\server/dir//one/two",
        "\\\\server/dir//", "two", "\\\\server/dir//one",
         2, 0, 1},
   {L_, "\\\\server/dir//one/two/",
        "\\\\server/dir//", "two", "\\\\server/dir//one",
         2, 0, 0},
   {L_, "\\\\server/dir/one/two/",
        "\\\\server/dir/", "two", "\\\\server/dir/one",
         2, 0, 0}
#else
   {L_, "", "", 0, 0,
         0, 1, 0},
   {L_, "/", "/", 0, 0,
         0, 0, 1},
   {L_, "//", "//", 0, 0,
         0, 0, 1},
   {L_, "/one", "/", "one", "/",
         1, 0, 1},
   {L_, "/one/", "/", "one", "/",
         1, 0, 0},
   {L_, "/one/two", "/", "two", "/one",
         2, 0, 1},
   {L_, "/one/two/", "/", "two", "/one",
         2, 0, 0},
   {L_, "//one", "//", "one", "//",
         1, 0, 1},
   {L_, "//one/", "//", "one", "//",
         1, 0, 0},
   {L_, "//one/two", "//", "two", "//one",
         2, 0, 1},
   {L_, "//one/two/", "//", "two", "//one",
         2, 0, 0},
   {L_, "one", "", "one", "",
         1, 1, 1},
   {L_, "one/", "", "one", "",
         1, 1, 0},
   {L_, "one/two", "", "two", "one",
         2, 1, 1},
   {L_, "one/two/", "", "two", "one",
         2, 1, 0},
   {L_, "one/two//", "", "two", "one",
         2, 1, 0}
#endif
};

template <class STRING_TYPE>
void convertToWindowsSeparator(STRING_TYPE *path)
    // Replace each occurrence of '/' with '\\' in the specified 'path'.
{
    bsl::string::size_type position = path->find('/');
    for ( ; position != bsl::string::npos; position = path->find('/')) {
        (*path)[position] = '\\';
    }
}

void convertToUnixSeparator(bsl::string *path)
    // Replace each occurrence of '\\' with '\' in the specified 'path'.
{
    bsl::string::size_type position = path->find('\\');
    for ( ;
         position != bsl::string::npos;
         position = path->find('\\', position)) {
        (*path)[position] = '/';
    }
}

template <class STRING_TYPE>
void usageExample()
{
///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: Basic Syntax
///- - - - - - - - - - - -
// We start with strings representing an absolute native path and a relative
// native path, respectively:
//..
    #ifdef BSLS_PLATFORM_OS_WINDOWS
    STRING_TYPE tempPath  = "c:\\windows\\temp";
    STRING_TYPE otherPath = "22jan08\\log.txt";
    #else
    STRING_TYPE tempPath  = "/var/tmp";
    STRING_TYPE otherPath = "22jan08/log.txt";
    #endif
//..
// 'tempPath' is an absolute path, since it has a root.  It also has a leaf
// element ("temp"):
//..
    ASSERT(false == bdls::PathUtil::isRelative(tempPath));
    ASSERT(true  == bdls::PathUtil::isAbsolute(tempPath));
    ASSERT(true  == bdls::PathUtil::hasLeaf(tempPath));
//..
// We can add filenames to the path one at a time, or we can add another path
// if is relative.  We can also remove filenames from the end of the path one
// at a time:
//..
    bdls::PathUtil::appendRaw(&tempPath, "myApp");
    bdls::PathUtil::appendRaw(&tempPath, "logs");

    ASSERT(true == bdls::PathUtil::isRelative(otherPath));
    ASSERT(0    == bdls::PathUtil::appendIfValid(&tempPath, otherPath));
    ASSERT(true == bdls::PathUtil::hasLeaf(tempPath));

    bdls::PathUtil::popLeaf(&tempPath);
    bdls::PathUtil::appendRaw(&tempPath, "log2.txt");

    #ifdef BSLS_PLATFORM_OS_WINDOWS
    ASSERT("c:\\windows\\temp\\myApp\\logs\\22jan08\\log2.txt" == tempPath);
    #else
    ASSERT("/var/tmp/myApp/logs/22jan08/log2.txt"              == tempPath);
    #endif
//..
// A relative path may be appended to any other path, even itself.  An absolute
// path may not be appended to any path, or undefined behavior will result:
//..
    ASSERT(0 == bdls::PathUtil::appendIfValid(&otherPath, otherPath));  // OK
    /* bdls::PathUtil::append(&otherPath, tempPath); */ // UNDEFINED BEHAVIOR!
//..
// Note that there is no attempt to distinguish filenames that are regular
// files from filenames that are directories, or to verify the existence of
// paths in the filesystem.
//..
    #ifdef BSLS_PLATFORM_OS_WINDOWS
    ASSERT("c:\\windows\\temp\\myApp\\logs\\22jan08\\log2.txt" == tempPath);
    #else
    ASSERT("/var/tmp/myApp/logs/22jan08/log2.txt"              == tempPath);
    #endif
//..
//
///Example 2: Parsing a path using 'splitFilename'
///- - - - - - - - - - - - - - - - - - - - - - - -
// Suppose we need to obtain all filenames from the path.
//
// First, we create a path for splitting and a storage for filenames:
//..
    #ifdef BSLS_PLATFORM_OS_WINDOWS
    const char                     *splitPath = "c:\\one\\two\\three\\four";
    #else
    const char                     *splitPath = "//one/two/three/four";
    #endif
    bsl::vector<bsl::string_view>  filenames;
//..
// Then, we run a cycle to sever filenames from the end one by one:
//..
    bsl::string_view head;
    bsl::string_view tail;
    bsl::string_view path(splitPath);

    do {
        bdls::PathUtil::splitFilename(&head, &tail, path);
        filenames.push_back(tail);
        path = head;
    } while (!tail.empty());
//..
// Now, verify the resulting values:
//..
    ASSERT(5           == filenames.size());

    ASSERT("four"      == filenames[0]);
    ASSERT("three"     == filenames[1]);
    ASSERT("two"       == filenames[2]);
    ASSERT("one"       == filenames[3]);
    ASSERT(""          == filenames[4]);
//..
// Finally, make sure that only the root remains of the original value:
//..
    #ifdef BSLS_PLATFORM_OS_WINDOWS
    ASSERT("c:\\"      == head);
    #else
    ASSERT("//"        == head);
    #endif
//..
}

template <class STRING_TYPE>
void test_getExtension(int verbose, int veryVerbose, int veryVeryVerbose)
{
    (void) verbose; (void) veryVerbose; (void) veryVeryVerbose;

    static const struct {
        int d_line;
        const char* d_path;
        int d_root;
        bool d_success;
        const char *d_extension;
    } DATA [] = {
// 1. Relative and absolute paths
{L_, "hello.txt",           0, true,  ".txt"   },
{L_, "hello",               0, false, ""       },
{L_, "/a/b/c.txt",          1, true,  ".txt"   },
{L_, "/a/b/c",              1, false, ""       },
// 2. Empty path
{L_, "",                    0, false, ""       },
// 3. Paths with multiple dots
{L_, "/a.c/b.d/aa.jpg.txt", 1, true,  ".txt"   },
{L_, "a/b/./d.dd/c.txt",    0, true,  ".txt"   },
{L_, "/a/b.c.d.e.a.m.cpp",  1, true,  ".cpp"   },
// 4. "." and ".."
{L_, "a/..",                0, false, ""       },
{L_, ".",                   0, false, ""       },
{L_, "/a/b.txt/.",          0, false, ""       },
{L_, "/a.txt/b.txt/..",     0, false, ""       },
// 5. First '.' is ignored
{L_, ".profile",            0, false, ""       },
{L_, ".profile.backup",     0, true,  ".backup"},
{L_, "/a/.txt",             1, false, ""       },
// 6. Dots in parent directories
{L_, "/a/b.c/a",            1, false, ""       },
{L_, "a.txt/b",             0, false, ""       },
// 7. Consistency with 'getLeaf'
{L_, "/a.txt/",             1, true,  ".txt"   },
{L_, "a.exe/",              0, true,  ".exe"   },
// 8. Empty extensions
{L_, "hello.",              0, true,  "."      },
{L_, "/a/b/c/d.",           1, true,  "."      },
#ifdef BSLS_PLATFORM_OS_WINDOWS
// 1. [Windows] Relative and absolute paths
{L_, "c:\\\\b\\a.txt",      3, true,  ".txt"   },
{L_, "\\\\serv\\c",         7, false, ""       },
// 3. [Windows] Paths with multiple dots
{L_, "\\\\a.txt",           8, false, ""       },
{L_, "c:\\\\a.b\\b.txt",    3, true,  ".txt"   },
// 4. [Windows] "." and ".."
{L_, "a\\..",               0, false, ""       },
{L_, "c:\\\\a.txt\\.",      3, false, ""       },
{L_, "c:\\\\a.txt\\..",     3, false, ""       },
// 5. [Windows] First '.' is ignored
{L_, "d:\\\\b\\.txt",       3, false, ""       },
// 6. [Windows] Dots in parent directories
{L_, "\\\\a\\b.c\\a",       4, false, ""       },
{L_, "a.txt\\b",            0, false, ""       },
// 7. [Windows] Consistency with 'getLeaf'
{L_, "c:\\\\a.txt\\",       3, true,  ".txt"   },
{L_, "a.exe\\",             0, true,  ".exe"   },
// 8. [Windows] Empty extensions
{L_, "\\\\a\\b\\c\\d.",     4, true,  "."      }
#endif
    };

    const size_t NUM_DATA = sizeof DATA / sizeof *DATA;

    for (size_t i = 0; i < NUM_DATA; ++i) {
        const int              LINE      = DATA[i].d_line;
        const bsl::string_view PATH      = DATA[i].d_path;
        const int              ROOT      = DATA[i].d_root;
        const bool             SUCCESS   = DATA[i].d_success;
        const bsl::string_view EXTENSION = DATA[i].d_extension;

        (void) PATH;

        // Explicit negative 'rootEnd'
        STRING_TYPE extension;
        int         result =
            bdls::PathUtil::getExtension(&extension, DATA[i].d_path, -1);
        ASSERTV(LINE, SUCCESS, result == 0, SUCCESS == (result == 0));
        ASSERTV(LINE, EXTENSION, extension, EXTENSION == extension);

        // Implicit negative 'rootEnd' (default value)
        extension.clear();
        result = bdls::PathUtil::getExtension(&extension, DATA[i].d_path);
        ASSERTV(LINE, SUCCESS, result == 0, SUCCESS == (result == 0));
        ASSERTV(LINE, EXTENSION, extension, EXTENSION == extension);

        // Explicit non-negative 'rootEnd'
        extension.clear();
        result =
            bdls::PathUtil::getExtension(&extension, DATA[i].d_path, ROOT);
        ASSERTV(LINE, SUCCESS, result == 0, SUCCESS == (result == 0));
        ASSERTV(LINE, EXTENSION, extension, EXTENSION == extension);
    }
}

template <class STRING_TYPE>
void test_splitFilename(int verbose, int veryVerbose, int veryVeryVerbose)
{
    (void) verbose; (void) veryVerbose; (void) veryVeryVerbose;

    static const struct {
            int         d_line;  // line
            const char *d_path;  // original path to split
            int         d_root;  // length of root
            const char *d_head;  // expected head
            const char *d_tail;  // expected tail
    } DATA [] = {
//v--------^
//L  ORIGINAL PATH                       ROOT  HEAD                     TAIL
//-  ----------------------------------  ----  -----------------------  -----
#ifdef BSLS_PLATFORM_OS_WINDOWS
{L_, "",                                   0,  "",                      ""   },
{L_, "\\",                                 1,  "\\",                    ""   },
{L_, "\\\\",                               2,  "\\\\",                  ""   },
{L_, "\\\\\\\\",                           4,  "\\\\\\\\",              ""   },
// Relative paths.
{L_, "one",                                0,  "",                      "one"},
{L_, "one\\",                              0,  "one",                   ""   },
{L_, "one\\\\",                            0,  "one",                   ""   },
{L_, "one\\\\\\",                          0,  "one",                   ""   },
{L_, "one\\\\\\\\",                        0,  "one",                   ""   },
{L_, "one\\two",                           0,  "one",                   "two"},
{L_, "one\\\\two",                         0,  "one",                   "two"},
{L_, "one\\\\\\two",                       0,  "one",                   "two"},
{L_, "one\\\\\\\\two",                     0,  "one",                   "two"},
// LFS.
{L_, "c:",                                 2,  "c:",                    ""   },
{L_, "c:\\",                               3,  "c:\\",                  ""   },
{L_, "c:\\\\",                             4,  "c:\\\\",                ""   },
{L_, "c:\\\\\\",                           5,  "c:\\\\\\",              ""   },
{L_, "c:\\\\\\\\",                         6,  "c:\\\\\\\\",            ""   },
{L_, "c:one",                              2,  "c:",                    "one"},
{L_, "c:\\one",                            3,  "c:\\",                  "one"},
{L_, "c:\\\\one",                          4,  "c:\\\\",                "one"},
{L_, "c:\\\\\\one",                        5,  "c:\\\\\\",              "one"},
{L_, "c:\\\\\\\\one",                      6,  "c:\\\\\\\\",            "one"},
{L_, "c:\\one\\",                          3,  "c:\\one",               ""   },
{L_, "c:\\\\one\\",                        4,  "c:\\\\one",             ""   },
{L_, "c:\\\\\\one\\",                      5,  "c:\\\\\\one",           ""   },
{L_, "c:\\\\\\\\one\\",                    6,  "c:\\\\\\\\one",         ""   },
{L_, "c:\\one\\two",                       3,  "c:\\one",               "two"},
{L_, "c:\\one\\\\two",                     3,  "c:\\one",               "two"},
{L_, "c:\\one\\\\\\two",                   3,  "c:\\one",               "two"},
{L_, "c:\\one\\\\\\\\two",                 3,  "c:\\one",               "two"},
// UNC.
{L_, "\\\\serv\\",                         7,  "\\\\serv\\",            ""   },
{L_, "\\\\serv\\dir",                      10, "\\\\serv\\dir",         ""   },
{L_, "\\\\serv\\dir\\",                    11, "\\\\serv\\dir\\",       ""   },
{L_, "\\\\serv\\dir\\one",                 11, "\\\\serv\\dir\\",       "one"},
{L_, "\\\\serv\\dir\\\\one",               12, "\\\\serv\\dir\\\\",     "one"},
{L_, "\\\\serv\\dir\\\\\\one",             13, "\\\\serv\\dir\\\\\\",   "one"},
{L_, "\\\\serv\\dir\\\\\\\\one",           14, "\\\\serv\\dir\\\\\\\\", "one"},
{L_, "\\\\serv\\dir\\one\\",               11, "\\\\serv\\dir\\one",    ""   },
{L_, "\\\\serv\\dir\\one\\\\",             11, "\\\\serv\\dir\\one",    ""   },
{L_, "\\\\serv\\dir\\one\\\\\\",           11, "\\\\serv\\dir\\one",    ""   },
{L_, "\\\\serv\\dir\\one\\\\\\\\",         11, "\\\\serv\\dir\\one",    ""   },
{L_, "\\\\.\\",                            4,  "\\\\.\\",               ""   },
{L_, "\\\\.\\dir",                         7,  "\\\\.\\dir",            ""   },
{L_, "\\\\.\\dir\\",                       8,  "\\\\.\\dir\\",          ""   },
{L_, "\\\\.\\dir\\one",                    8,  "\\\\.\\dir\\",          "one"},
{L_, "\\\\.\\dir\\\\one",                  9,  "\\\\.\\dir\\\\",        "one"},
{L_, "\\\\.\\dir\\\\\\one",                10, "\\\\.\\dir\\\\\\",      "one"},
{L_, "\\\\.\\dir\\\\\\\\one",              11, "\\\\.\\dir\\\\\\\\",    "one"},
{L_, "\\\\.\\dir\\one\\",                  8,  "\\\\.\\dir\\one",       ""   },
{L_, "\\\\.\\dir\\one\\\\",                8,  "\\\\.\\dir\\one",       ""   },
{L_, "\\\\.\\dir\\one\\\\\\",              8,  "\\\\.\\dir\\one",       ""   },
{L_, "\\\\.\\dir\\one\\\\\\\\",            8,  "\\\\.\\dir\\one",       ""   },
// LUNC.
{L_, "\\\\?\\c:",                          6,  "\\\\?\\c:",             ""   },
{L_, "\\\\?\\c:\\",                        7,  "\\\\?\\c:\\",           ""   },
{L_, "\\\\?\\c:\\\\",                      8,  "\\\\?\\c:\\\\",         ""   },
{L_, "\\\\?\\c:\\\\\\",                    9,  "\\\\?\\c:\\\\\\",       ""   },
{L_, "\\\\?\\c:\\\\\\\\",                  10, "\\\\?\\c:\\\\\\\\",     ""   },
{L_, "\\\\?\\c:one",                       6,  "\\\\?\\c:",             "one"},
{L_, "\\\\?\\c:\\one",                     7,  "\\\\?\\c:\\",           "one"},
{L_, "\\\\?\\c:\\\\one",                   8,  "\\\\?\\c:\\\\",         "one"},
{L_, "\\\\?\\c:\\\\\\one",                 9,  "\\\\?\\c:\\\\\\",       "one"},
{L_, "\\\\?\\c:\\\\\\\\one",               10, "\\\\?\\c:\\\\\\\\",     "one"},
{L_, "\\\\?\\c:\\one\\",                   7,  "\\\\?\\c:\\one",        ""   },
{L_, "\\\\?\\c:\\\\one\\",                 8,  "\\\\?\\c:\\\\one",      ""   },
{L_, "\\\\?\\c:\\\\\\one\\",               9,  "\\\\?\\c:\\\\\\one",    ""   },
{L_, "\\\\?\\c:\\\\\\\\one\\",             10, "\\\\?\\c:\\\\\\\\one",  ""   },
{L_, "\\\\?\\c:\\one\\two",                7,  "\\\\?\\c:\\one",        "two"},
{L_, "\\\\?\\c:\\one\\\\two",              7,  "\\\\?\\c:\\one",        "two"},
{L_, "\\\\?\\c:\\one\\\\\\two",            7,  "\\\\?\\c:\\one",        "two"},
{L_, "\\\\?\\c:\\one\\\\\\\\two",          7,  "\\\\?\\c:\\one",        "two"},
{L_, "\\\\?\\UNC\\serv\\",                 13, "\\\\?\\UNC\\serv\\",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir",              16, "\\\\?\\UNC\\serv\\dir",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir\\",            17, "\\\\?\\UNC\\serv\\dir\\",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir\\one",         17, "\\\\?\\UNC\\serv\\dir\\",
                                                                    "one"},
{L_, "\\\\?\\UNC\\serv\\dir\\\\one",       18, "\\\\?\\UNC\\serv\\dir\\\\",
                                                                    "one"},
{L_, "\\\\?\\UNC\\serv\\dir\\\\\\one",     19, "\\\\?\\UNC\\serv\\dir\\\\\\",
                                                                    "one"},
{L_, "\\\\?\\UNC\\serv\\dir\\\\\\\\one",   20, "\\\\?\\UNC\\serv\\dir\\\\\\\\",
                                                                    "one"},

{L_, "\\\\?\\UNC\\serv\\dir\\one\\",       19, "\\\\?\\UNC\\serv\\dir\\one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir\\one\\\\",     19, "\\\\?\\UNC\\serv\\dir\\one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir\\one\\\\\\",   19, "\\\\?\\UNC\\serv\\dir\\one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv\\dir\\one\\\\\\\\", 19, "\\\\?\\UNC\\serv\\dir\\one",
                                                                    ""   },
//L  ORIGINAL PATH                       ROOT  HEAD                     TAIL
//-  ----------------------------------  ----  -----------------------  -----
// Forward slash.
{L_, "/",                                  1,  "/",                     ""   },
{L_, "//",                                 2,  "//",                    ""   },
{L_, "///",                                3,  "///",                   ""   },
{L_, "////",                               4,  "////",                  ""   },
// Forward slash relative paths.
{L_, "one/",                               0,  "one",                   ""   },
{L_, "one//",                              0,  "one",                   ""   },
{L_, "one///",                             0,  "one",                   ""   },
{L_, "one////",                            0,  "one",                   ""   },
{L_, "one/two",                            0,  "one",                   "two"},
{L_, "one//two",                           0,  "one",                   "two"},
{L_, "one///two",                          0,  "one",                   "two"},
{L_, "one////two",                         0,  "one",                   "two"},
// Forward slash LFS.
{L_, "c:/",                                3,  "c:/",                   ""   },
{L_, "c://",                               4,  "c://",                  ""   },
{L_, "c:/one",                             3,  "c:/",                   "one"},
{L_, "c://one",                            4,  "c://",                  "one"},
{L_, "c:///one",                           5,  "c:///",                 "one"},
{L_, "c:////one",                          6,  "c:////",                "one"},
{L_, "c:/one/",                            3,  "c:/one",                ""   },
{L_, "c:/one//",                           3,  "c:/one",                ""   },
{L_, "c:/one///",                          3,  "c:/one",                ""   },
{L_, "c:/one/////",                        3,  "c:/one",                ""   },
{L_, "c:/one/two",                         3,  "c:/one",                "two"},
{L_, "c:/one//two",                        3,  "c:/one",                "two"},
{L_, "c:/one///two",                       3,  "c:/one",                "two"},
{L_, "c:/one////two",                      3,  "c:/one",                "two"},
// Forward slash UNC.
{L_, "\\\\serv/",                          7,  "\\\\serv/",             ""   },
{L_, "\\\\serv/dir",                       10, "\\\\serv/dir",          ""   },
{L_, "\\\\serv/dir/",                      11, "\\\\serv/dir/",         ""   },
{L_, "\\\\serv/dir/one",                   11, "\\\\serv/dir/",         "one"},
{L_, "\\\\serv/dir//one",                  12, "\\\\serv/dir//",        "one"},
{L_, "\\\\serv/dir///one",                 13, "\\\\serv/dir///",       "one"},
{L_, "\\\\serv/dir////one",                14, "\\\\serv/dir////",      "one"},
{L_, "\\\\serv/dir/one/",                  11, "\\\\serv/dir/one",      ""   },
{L_, "\\\\serv/dir/one//",                 11, "\\\\serv/dir/one",      ""   },
{L_, "\\\\serv/dir/one///",                11, "\\\\serv/dir/one",      ""   },
{L_, "\\\\serv/dir/one////",               11, "\\\\serv/dir/one",      ""   },
{L_, "\\\\./",                             4,  "\\\\./",                ""   },
{L_, "\\\\./dir",                          7,  "\\\\./dir",             ""   },
{L_, "\\\\./dir/",                         8,  "\\\\./dir/",            ""   },
{L_, "\\\\./dir/one",                      8,  "\\\\./dir/",            "one"},
{L_, "\\\\./dir//one",                     9,  "\\\\./dir//",           "one"},
{L_, "\\\\./dir///one",                    10, "\\\\./dir///",          "one"},
{L_, "\\\\./dir////one",                   11, "\\\\./dir////",         "one"},
{L_, "\\\\./dir/one/",                     8,  "\\\\./dir/one",         ""   },
{L_, "\\\\./dir/one//",                    8,  "\\\\./dir/one",         ""   },
{L_, "\\\\./dir/one///",                   8,  "\\\\./dir/one",         ""   },
{L_, "\\\\./dir/one////",                  8,  "\\\\./dir/one",         ""   },
// Forward slash LUNC.
{L_, "\\\\?\\c:",                          6,  "\\\\?\\c:",             ""   },
{L_, "\\\\?\\c:/",                         7,  "\\\\?\\c:/",            ""   },
{L_, "\\\\?\\c://",                        8,  "\\\\?\\c://",           ""   },
{L_, "\\\\?\\c:///",                       9,  "\\\\?\\c:///",          ""   },
{L_, "\\\\?\\c:////",                      10, "\\\\?\\c:////",         ""   },
{L_, "\\\\?\\c:one",                       6,  "\\\\?\\c:",             "one"},
{L_, "\\\\?\\c:/one",                      7,  "\\\\?\\c:/",            "one"},
{L_, "\\\\?\\c://one",                     8,  "\\\\?\\c://",           "one"},
{L_, "\\\\?\\c:///one",                    9,  "\\\\?\\c:///",          "one"},
{L_, "\\\\?\\c:////one",                   10, "\\\\?\\c:////",         "one"},
{L_, "\\\\?\\c:/one/",                     7,  "\\\\?\\c:/one",         ""   },
{L_, "\\\\?\\c://one/",                    8,  "\\\\?\\c://one",        ""   },
{L_, "\\\\?\\c:///one/",                   9,  "\\\\?\\c:///one",       ""   },
{L_, "\\\\?\\c:////one/",                  10, "\\\\?\\c:////one",      ""   },
{L_, "\\\\?\\c:/one/two",                  7,  "\\\\?\\c:/one",         "two"},
{L_, "\\\\?\\c:/one//two",                 7,  "\\\\?\\c:/one",         "two"},
{L_, "\\\\?\\c:/one///two",                7,  "\\\\?\\c:/one",         "two"},
{L_, "\\\\?\\c:/one////two",               7,  "\\\\?\\c:/one",         "two"},
{L_, "\\\\?\\UNC\\serv/",                  13, "\\\\?\\UNC\\serv/",     ""   },
{L_, "\\\\?\\UNC\\serv/dir",               16, "\\\\?\\UNC\\serv/dir",  ""   },
{L_, "\\\\?\\UNC\\serv/dir/",              17, "\\\\?\\UNC\\serv/dir/", ""   },
{L_, "\\\\?\\UNC\\serv/dir/one",           17, "\\\\?\\UNC\\serv/dir/", "one"},
{L_, "\\\\?\\UNC\\serv/dir//one",          18, "\\\\?\\UNC\\serv/dir//",
                                                                    "one"},
{L_, "\\\\?\\UNC\\serv/dir///one",         19, "\\\\?\\UNC\\serv/dir///",
                                                                    "one"},
{L_, "\\\\?\\UNC\\serv/dir////one",        20, "\\\\?\\UNC\\serv/dir////",
                                                                    "one"},

{L_, "\\\\?\\UNC\\serv/dir/one/",          19, "\\\\?\\UNC\\serv/dir/one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv/dir/one//",         19, "\\\\?\\UNC\\serv/dir/one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv/dir/one///",        19, "\\\\?\\UNC\\serv/dir/one",
                                                                    ""   },
{L_, "\\\\?\\UNC\\serv/dir/one////",       19, "\\\\?\\UNC\\serv/dir/one",
                                                                    ""   },
#else
//L  ORIGINAL PATH                       ROOT  HEAD                     TAIL
//-  ----------------------------------  ----  -----------------------  -----
{L_, "",                                   0,  "",                      ""   },
{L_, "/",                                  1,  "/",                     ""   },
{L_, "//",                                 2,  "//",                    ""   },
{L_, "///",                                3,  "///",                   ""   },
{L_, "////",                               4,  "////",                  ""   },
{L_, "one",                                0,  "",                      "one"},
{L_, "one/",                               0,  "one",                   ""   },
{L_, "one//",                              0,  "one",                   ""   },
{L_, "one///",                             0,  "one",                   ""   },
{L_, "one////",                            0,  "one",                   ""   },
{L_, "one/two",                            0,  "one",                   "two"},
{L_, "one//two",                           0,  "one",                   "two"},
{L_, "one///two",                          0,  "one",                   "two"},
{L_, "one////two",                         0,  "one",                   "two"},
{L_, "one/two/",                           0,  "one/two",               ""   },
{L_, "one//two/",                          0,  "one//two",              ""   },
{L_, "one///two/",                         0,  "one///two",             ""   },
{L_, "one////two/",                        0,  "one////two",            ""   },
{L_, "/one",                               1,  "/",                     "one"},
{L_, "/one/",                              1,  "/one",                  ""   },
{L_, "/one//",                             1,  "/one",                  ""   },
{L_, "/one///",                            1,  "/one",                  ""   },
{L_, "/one////",                           1,  "/one",                  ""   },
{L_, "/one/two",                           1,  "/one",                  "two"},
{L_, "/one//two",                          1,  "/one",                  "two"},
{L_, "/one///two",                         1,  "/one",                  "two"},
{L_, "/one////two",                        1,  "/one",                  "two"},
{L_, "/one/two/",                          1,  "/one/two",              ""   },
{L_, "/one//two/",                         1,  "/one//two",             ""   },
{L_, "/one///two/",                        1,  "/one///two",            ""   },
{L_, "/one////two/",                       1,  "/one////two",           ""   },
{L_, "//one",                              2,  "//",                    "one"},
{L_, "//one/",                             2,  "//one",                 ""   },
{L_, "//one//",                            2,  "//one",                 ""   },
{L_, "//one///",                           2,  "//one",                 ""   },
{L_, "//one////",                          2,  "//one",                 ""   },
{L_, "//one/two",                          2,  "//one",                 "two"},
{L_, "//one//two",                         2,  "//one",                 "two"},
{L_, "//one///two",                        2,  "//one",                 "two"},
{L_, "//one////two",                       2,  "//one",                 "two"},
{L_, "//one/two/",                         2,  "//one/two",             ""   },
{L_, "//one//two/",                        2,  "//one//two",            ""   },
{L_, "//one///two/",                       2,  "//one///two",           ""   },
{L_, "//one////two/",                      2,  "//one////two",          ""   },
{L_, "///one",                             3,  "///",                   "one"},
{L_, "///one/",                            3,  "///one",                ""   },
{L_, "///one//",                           3,  "///one",                ""   },
{L_, "///one///",                          3,  "///one",                ""   },
{L_, "///one////",                         3,  "///one",                ""   },
{L_, "///one/two",                         3,  "///one",                "two"},
{L_, "///one//two",                        3,  "///one",                "two"},
{L_, "///one///two",                       3,  "///one",                "two"},
{L_, "///one////two",                      3,  "///one",                "two"},
{L_, "///one/two/",                        3,  "///one/two",            ""   },
{L_, "///one//two/",                       3,  "///one//two",           ""   },
{L_, "///one///two/",                      3,  "///one///two",          ""   },
{L_, "///one////two/",                     3,  "///one////two",         ""   },
#endif
//^--------v
    };
    const size_t NUM_DATA = sizeof DATA / sizeof *DATA;

    const bsl::string_view   emptyRefOrView;

    if (verbose)
        cout << "\nBehavior Testing." << endl;

    for (size_t ti = 0; ti < NUM_DATA; ++ti) {
        const int              LINE     = DATA[ti].d_line;
        const bsl::string_view PATH     = DATA[ti].d_path;
        const int              ROOT_END = DATA[ti].d_root;
        const STRING_TYPE      EXP_HEAD = DATA[ti].d_head;
        const STRING_TYPE      EXP_TAIL = DATA[ti].d_tail;

        if (veryVerbose) { T_; P_(ti); P(PATH); }

        // Explicit negative 'rootEnd'.

        bsl::string_view head;
        bsl::string_view tail;

        Obj::splitFilename(&head, &tail, PATH, -1);

        ASSERTV(LINE, EXP_HEAD, head, EXP_HEAD == head);
        if (EXP_HEAD != head) {
            cout << __FILE__ << ":" << __LINE__
                 << "\n\thead    : \"" << head << "\""
                 << "\n\tEXP_HEAD: \"" << EXP_HEAD << "\""
                 << endl;
        }

        ASSERTV(LINE, EXP_TAIL, tail, EXP_TAIL == tail);
        if (EXP_TAIL != tail) {
            cout << __FILE__ << ":" << __LINE__
                 << "\n\ttail    : \"" << tail << "\""
                 << "\n\tEXP_TAIL: \"" << EXP_TAIL << "\""
                 << endl;
        }

        // Implicit negative 'rootEnd' (default value).

        head = emptyRefOrView;
        tail = emptyRefOrView;

        Obj::splitFilename(&head, &tail, PATH);

        ASSERTV(LINE, EXP_HEAD, head, EXP_HEAD == head);
        ASSERTV(LINE, EXP_TAIL, tail, EXP_TAIL == tail);

        // Explicit non-negative 'rootEnd'.

        head = emptyRefOrView;
        tail = emptyRefOrView;

        Obj::splitFilename(&head, &tail, PATH, ROOT_END);

        ASSERTV(LINE, EXP_HEAD, head, EXP_HEAD == head);
        ASSERTV(LINE, EXP_TAIL, tail, EXP_TAIL == tail);

        // 'head' is alias of 'path'.

        bsl::string_view aliasPath = DATA[ti].d_path;
        tail = emptyRefOrView;

        Obj::splitFilename(&aliasPath, &tail, aliasPath);

        ASSERTV(LINE, EXP_HEAD, head,      EXP_HEAD == head);
        ASSERTV(LINE, EXP_HEAD, aliasPath, EXP_HEAD == aliasPath);
        ASSERTV(LINE, EXP_TAIL, tail,      EXP_TAIL == tail);

        // 'tail' is alias of 'path'.

        aliasPath = DATA[ti].d_path;
        head = emptyRefOrView;

        Obj::splitFilename(&head, &aliasPath, aliasPath);

        ASSERTV(LINE, EXP_HEAD, head,      EXP_HEAD == head);
        ASSERTV(LINE, EXP_TAIL, tail,      EXP_TAIL == tail);
        ASSERTV(LINE, EXP_HEAD, aliasPath, EXP_TAIL == aliasPath);
    }

    if (verbose) cout << "\nNegative Testing." << endl;
    {
        bsls::AssertTestHandlerGuard hG;

        bsl::string_view head;
        bsl::string_view tail;
        bsl::string_view path;

        ASSERT_FAIL(Obj::splitFilename(0,         0, path));
        ASSERT_FAIL(Obj::splitFilename(0,     &tail, path));
        ASSERT_FAIL(Obj::splitFilename(&head,     0, path));
        ASSERT_FAIL(Obj::splitFilename(&head, &head, path));

        ASSERT_PASS(Obj::splitFilename(&head, &tail, path));
    }

}

template <class STRING_TYPE>
void test_appendIfValid(int verbose, int veryVerbose, int veryVeryVerbose)
{
    (void) verbose; (void) veryVerbose; (void) veryVeryVerbose;

    if (verbose) {
        cout << "\tUse table of distinct object values." << endl;
    }

    {
        struct TestData {
                int         d_line;
                const char *d_path;
                const char *d_filename;
                bool        d_expectSuccess;
                const char *d_expectedResult;
        } VALUES [] = {
            { L_,  "" , "" , true, ""    },
            { L_,  "a", "" , true, "a"   },
            { L_,  "" , "a", true, "a"   },
            { L_,  "a", "b", true, "a/b" },

            { L_,  "a/" , "b"   , true,  "a/b" },
            { L_,  "a//", "b"   , true,  "a/b" },
            { L_,  "a"  , "b/"  , true,  "a/b" },
            { L_,  "a"  , "b//" , true,  "a/b" },
            { L_,  "a//", "b//" , true,  "a/b" },
            { L_,  "a//", "/b//", false, "" },

            { L_,  "/a/" , "b"   , true , "/a/b" },
            { L_,  "/a//", "b"   , true , "/a/b" },
            { L_,  "/a"  , "b/"  , true , "/a/b" },
            { L_,  "/a"  , "b//" , true , "/a/b" },
            { L_,  "/a//", "b//" , true , "/a/b" },
            { L_,  "/a//", "/b//", false, "" },


            { L_,  "/"     , "b" , true, "/b" },
            { L_,  "//////", "b" , true, "/b" },
            { L_,  "//////", "b/", true, "/b" },

            { L_,  "/a/b/c/" , "e/f/g"   , true,  "/a/b/c/e/f/g" },
            { L_,  "/a/b/c//", "e/f/g"   , true,  "/a/b/c/e/f/g" },
            { L_,  "/a/b/c"  , "e/f/g/"  , true,  "/a/b/c/e/f/g" },
            { L_,  "/a/b/c"  , "e/f/g//" , true,  "/a/b/c/e/f/g" },
            { L_,  "/a/b/c//", "e/f/g//" , true,  "/a/b/c/e/f/g" },
            { L_,  "/a/b/c//", "/e/f/g//", false, "" },


#ifdef BSLS_PLATFORM_OS_WINDOWS
            // Test a path starting with a drive letter.
            { L_,  "z:"    , "b" , true ,  "z:/b" },
            { L_,  "z:/"   , "b" , true ,  "z:/b" },
            { L_,  "z://"  , "b" , true ,  "z:/b" },
            { L_,  "z://"  , "b/", true ,  "z:/b" },
            { L_,  "z:/a"  , "b" , true ,  "z:/a/b" },
            { L_,  "z:/a/" , "b" , true ,  "z:/a/b" },
            { L_,  "z:/a//", "b" , true ,  "z:/a/b" },
            { L_,  "z:/"   , "/b", false,  "z:/b" },

            // Test UNC paths
            { L_,  "//UNC"    , "b" , true ,  "//UNC/b" },
            { L_,  "//UNC/"   , "b" , true ,  "//UNC/b" },
            { L_,  "//UNC//"  , "b" , true ,  "//UNC/b" },
            { L_,  "//UNC//"  , "b/", true ,  "//UNC/b" },
            { L_,  "//UNC/a"  , "b" , true ,  "//UNC/a/b" },
            { L_,  "//UNC/a/" , "b" , true ,  "//UNC/a/b" },
            { L_,  "//UNC/a//", "b" , true ,  "//UNC/a/b" },
            { L_,  "//UNC/"   , "/b", false,  "//UNC/b" },

            // Test device paths
            { L_,  "//?/"   , "b" , true ,  "//?/b" },
            { L_,  "//?//"  , "b" , true ,  "//?/b" },
            { L_,  "//?//"  , "b/", true ,  "//?/b" },
            { L_,  "//?/a"  , "b" , true ,  "//?/a/b" },
            { L_,  "//?/a/" , "b" , true ,  "//?/a/b" },
            { L_,  "//?/a//", "b" , true ,  "//?/a/b" },
            { L_,  "//?/"   , "/b", false,  "//?/b" },
#endif

        };
        const int NUM_VALUES = sizeof(VALUES) / sizeof(*VALUES);

        for (int i = 0; i < NUM_VALUES; ++i) {
            STRING_TYPE path(VALUES[i].d_path);
            STRING_TYPE filename(VALUES[i].d_filename);
            STRING_TYPE expectedResult(VALUES[i].d_expectedResult);

            const int  LINE    = VALUES[i].d_line;
            const bool success = VALUES[i].d_expectSuccess;


#ifdef BSLS_PLATFORM_OS_WINDOWS
            convertToWindowsSeparator(&path);
            convertToWindowsSeparator(&filename);
            convertToWindowsSeparator(&expectedResult);
#endif

            STRING_TYPE originalPath(path);

            int rc = Obj::appendIfValid(&path, filename);

            if (veryVeryVerbose) {
                P_(LINE); P_(originalPath); P_(filename);
                P_(path); P(expectedResult);
            }

            if (!success) {
                LOOP_ASSERT(LINE, 0 != rc);
                continue;
            }

            LOOP_ASSERT(LINE, 0 == rc);
            LOOP4_ASSERT(LINE, originalPath, filename, path,
                         expectedResult == path);
        }
    }
    if (verbose) {
        cout << "\tTest for aliasing." << endl;
    }

    {
        const char *VALUES[] = {
            "",
            "a",
            "abc",
            "thisisalongpathnamexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        };
        const int NUM_VALUES = sizeof(VALUES) / sizeof(*VALUES);

        for (int i = 0; i < NUM_VALUES; ++i) {
            const size_t pathLen = bsl::strlen(VALUES[i]);

            for (size_t subStrLen = 0; subStrLen < pathLen; ++subStrLen) {
                for (size_t offset = 0; offset < pathLen - subStrLen + 1;
                     ++offset) {
                    STRING_TYPE      path(VALUES[i]);
                    bsl::string_view filename(path.c_str() + offset,
                                              subStrLen);

                    if (filename.length() > 0 && '/' == filename[0]) {
                        continue;
                    }

                    STRING_TYPE expectedResult(path);
                    if (path.size() > 0 && filename.length() > 0) {
                        expectedResult += "/";
                    }
                    expectedResult +=
                        STRING_TYPE(filename.data(), filename.length());

#ifdef BSLS_PLATFORM_OS_WINDOWS
                    convertToWindowsSeparator(&path);
                    convertToWindowsSeparator(&expectedResult);
#endif
                    STRING_TYPE originalPath(path);
                    STRING_TYPE originalFilename(filename.data(),
                                                 filename.length());

                    int rc = Obj::appendIfValid(&path, filename);

                    if (veryVeryVerbose) {
                        P_(originalPath); P(originalFilename);
                        P_(path); P(expectedResult);
                    }

                    ASSERT(0 == rc);
                    LOOP4_ASSERT(originalPath,
                                 originalFilename,
                                 path,
                                 expectedResult,
                                 expectedResult == path);
                }
            }
        }
    }
}

template <class STRING_TYPE>
void test_leaflessAppend(int verbose, int veryVerbose, int veryVeryVerbose)
{
    (void) verbose; (void) veryVerbose; (void) veryVeryVerbose;

    enum {
       LEAF_ORIGINAL = 0,
       LEAF_EXPECTED = 1,
       LEAFLESS_ORIGINAL = 2,
       LEAFLESS_EXPECTED = 3
    };
    const char *absolute[] = {
#ifdef BSLS_PLATFORM_OS_WINDOWS
        "\\\\machine\\dir\\logs",
        "\\\\machine\\dir\\logs\\hello",
        "\\\\machine\\dir\\",
        "\\\\machine\\dir\\hello",
        "\\\\machine/dir/logs",
        "\\\\machine/dir/logs/hello",
        "\\\\machine/dir/",
        "\\\\machine/dir/hello"
#else
        "/logs",
        "/logs/hello",
        "/",
        "/hello"
#endif
    };
    const char *relative[] = {
#ifdef BSLS_PLATFORM_OS_WINDOWS
        "logs",
        "logs\\hello",
        "",
        "hello",
        "logs",
        "logs/hello",
        "",
        "hello"
#else
        "logs",
        "logs/hello",
        "",
        "hello"
#endif
    };
    const char** arrays[] = {absolute, relative};
    for (int i = 0; i < 2; ++i) {
       STRING_TYPE leaf(arrays[i][LEAF_ORIGINAL]);
       bdls::PathUtil::appendRaw(&leaf, "hello");
       ASSERT(leaf == STRING_TYPE(arrays[i][LEAF_EXPECTED]));

       STRING_TYPE leafless(arrays[i][LEAFLESS_ORIGINAL]);
       bdls::PathUtil::appendRaw(&leafless, "hello");
       LOOP3_ASSERT(
              arrays[i][LEAFLESS_ORIGINAL],
              arrays[i][LEAFLESS_EXPECTED],
              leafless, leafless == STRING_TYPE(arrays[i][LEAFLESS_EXPECTED]));
    }

    STRING_TYPE leafless2;
#ifdef BSLS_PLATFORM_OS_WINDOWS
    leafless2 = "\\\\.\\pipe";
#else
    leafless2 = "/var/tmp/"; // not really leafless but let's do SOMEthing
                             // on unix for this part
#endif
    ASSERT( 0 == bdls::PathUtil::appendIfValid(&leafless2, "hello") );
#ifdef BSLS_PLATFORM_OS_WINDOWS
    LOOP_ASSERT(leafless2, "\\\\.\\pipe\\hello" == leafless2);
#else
    LOOP_ASSERT(leafless2, "/var/tmp/hello" == leafless2);
#endif
}

template <class STRING_TYPE>
void test_nativeParsingTest(int verbose, int veryVerbose, int veryVeryVerbose)
{
    (void) verbose; (void) veryVerbose; (void) veryVeryVerbose;

    const int NUM_PARAMETERS = sizeof(parameters) / sizeof(Parameters);
    for (int i = 0; i < NUM_PARAMETERS; ++i) {
        const Parameters& pi = parameters[i];

        STRING_TYPE iTest(pi.d_path);

        // Verify root

        STRING_TYPE root;
        if (pi.d_isRelative) {
            ASSERT(0 != bdls::PathUtil::getRoot(&root, iTest));
        }
        else {
            ASSERT(0 == bdls::PathUtil::getRoot(&root, iTest));
            ASSERTV(root, root == pi.d_root);
        }

        // Verify getLeaf, getDirName

        STRING_TYPE basename, dirname;
        if (pi.d_numLeaves) {
            ASSERT(0 == bdls::PathUtil::getLeaf(&basename, iTest));
            ASSERT(0 == bdls::PathUtil::getDirname(&dirname, iTest));

            //test invariant:

            ASSERT(pi.d_leaf && pi.d_dirName);

            LOOP2_ASSERT(pi.d_line, basename, basename == pi.d_leaf);
            LOOP2_ASSERT(pi.d_line, dirname, dirname == pi.d_dirName);
        }
        else {
            ASSERT(0 != bdls::PathUtil::getLeaf(&basename, iTest));
            ASSERT(0 != bdls::PathUtil::getDirname(&dirname, iTest));
        }

        // Count leaves by removing them iteratively...When we're done,
        // verify the root of the path (we should not be able to change it
        // by removing leaves)

        int count;
        for (count = 0; bdls::PathUtil::hasLeaf(iTest); ++count) {
            bdls::PathUtil::popLeaf(&iTest);
        }
        LOOP2_ASSERT(pi.d_line, count, count == pi.d_numLeaves);
        ASSERT(bdls::PathUtil::isRelative(iTest) == pi.d_isRelative);
        if (!pi.d_isRelative) {
            ASSERT(0 == bdls::PathUtil::getRoot(&root, iTest));
            LOOP_ASSERT(root, root == pi.d_root);
        }

        for (int j = 0; j < NUM_PARAMETERS; ++j) {
            const Parameters& pj = parameters[j];

            int referenceCount;
            if (pj.d_isRelative) {
                ASSERT(0 ==
                         bdls::PathUtil::appendIfValid(&iTest, pj.d_path));
                referenceCount = pj.d_numLeaves;
            }
            else {
                ASSERT(0 !=
                         bdls::PathUtil::appendIfValid(&iTest, pj.d_path));
                referenceCount = 0;
            }
            for (count = 0;
                 bdls::PathUtil::hasLeaf(iTest);
                 ++count) {
                bdls::PathUtil::popLeaf(&iTest);
            }
            LOOP2_ASSERT(pi.d_path, pj.d_path, count == referenceCount);
            LOOP2_ASSERT(pi.d_path, iTest,
                       bdls::PathUtil::isRelative(iTest)==pi.d_isRelative);
        }
    }
}

int main(int argc, char *argv[])
{
    int             test = argc > 1 ? bsl::atoi(argv[1]) : 0;
    bool         verbose = argc > 2;
    bool     veryVerbose = argc > 3;
    bool veryVeryVerbose = argc > 4;

    (void)veryVerbose;

    // CONCERN: 'BSLS_REVIEW' failures should lead to test failures.
    bsls::ReviewFailureHandlerGuard reviewGuard(&bsls::Review::failByAbort);

    switch(test) { case 0:
      case 7: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
        //   Extracted from component header file.
        //
        // Concerns:
        //: 1 The usage example provided in the component header file compiles,
        //:   links, and runs as shown.
        //
        // Plan:
        //: 1 Incorporate usage example from header into test driver, remove
        //:   leading comment characters, and replace 'assert' with 'ASSERT'.
        //:   (C-1)
        //
        // Testing:
        //   USAGE EXAMPLE
        // --------------------------------------------------------------------

        if (verbose) cout << endl
                          << "TESTING USAGE EXAMPLE" << endl
                          << "=====================" << endl;

        usageExample<bsl::string>();
        usageExample<std::string>();
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        usageExample<std::pmr::string>();
#endif
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // TESTING: 'getExtension'
        //
        // Concerns:
        //: 1 The 'getExtension' method accepts absolute and relative paths.
        //:
        //: 2 The 'getExtension' method accepts empty paths and returns an
        //:   empty extension in that case.
        //:
        //: 3 The 'getExtension' method is able to find the extension even
        //:   in filenames that contain multiple dots.
        //:
        //: 4 The 'getExtension' method does not find extensions in the special
        //:   files '.' and '..'
        //:
        //: 5 In the case that the leaf of the path begins with a dot ('.'),
        //:   it is ignored as a character for considering what the path is.
        //:
        //: 6 The 'getExtension' method is not tricked by dots in the
        //:     directory names containing the path
        //:
        //: 7 The 'getExtension' method behaviour is consistent with that of
        //:   the 'getLeaf' method ("a.txt/" *has* an extension)
        //:
        //: 8 The 'getExtension' method correctly identifies empty extensions
        //:
        //: 9 Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Create a table of test input values and expected results
        //:
        //: 2 Iterate over this table verifying that 'getExtension' produces
        //:   the expected results with
        //:     - the explicit negative value of the 'rootEnd'
        //:     - the default value of the 'rootEnd'
        //:     - the explicit correct value of the 'rootEnd
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid attribute values, but not triggered for
        //:   adjacent valid ones.  (C-9)
        //
        // Testing:
        //   void getExtension(bsl::string*, const string_view&, int);
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "TESTING: getExtension" << endl
                 << "=====================" << endl;
        }

        test_getExtension<bsl::string>(verbose, veryVerbose, veryVeryVerbose);
        test_getExtension<std::string>(verbose, veryVerbose, veryVeryVerbose);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        test_getExtension<std::pmr::string>(
            verbose, veryVerbose, veryVeryVerbose);
#endif

    }; break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING: 'splitFilename'
        //
        // Concerns:
        //: 1 The 'splitFilename' method accepts absolute and relative paths.
        //:
        //: 2 The 'splitFilename' method accepts empty paths and returns empty
        //:   'head' and 'tail' in such cases.
        //:
        //: 3 The resulting 'head' always contains the root of the original
        //:   path.
        //:
        //: 4 The resulting 'head' does not contain trailing seperators.
        //:
        //: 5 The resulting 'tail' does not contain seperators.
        //:
        //: 6 The 'splitFilename' method properly handles Windows paths
        //:   containing forward and backward slashes.
        //:
        //: 7 The 'splitFilename' method correctly identifies the root end of
        //:   the passed path.
        //:
        //: 8 The 'splitFilename' method correctly handles passed 'head' or
        //:   'tail' in the case when they are aliases of 'path'
        //:   ('head == &path' or 'tail == &path').
        //:
        //: 9 Asserted precondition violations are detected when enabled.
        //
        // Plan:
        //: 1 Create a table of test input values and expected results.  Input
        //:   values are graded in the following way:
        //:
        //:     Windows:
        //:     --------
        //:     - empty path
        //:     - slashes only
        //:     - LFS  root
        //:     - UNC  root
        //:     - LUNC root
        //:
        //:   As Windows OS supports both backward and forward slashes, we
        //:   check both variants separately.
        //:
        //:     Unix:
        //:     -----
        //:     - empty path
        //:     - one slash root
        //:     - two slashes root
        //:     - three slashes root
        //:
        //:   Whithin each group input values are graded in the following way:
        //:
        //:     - root + delimiter(s)
        //:     - root + delimiter(s) + file
        //:     - root + delimiter(s) + folder
        //:     - root + delimiter    + folder + delimiter(s) + file
        //:
        //:   As paths can contain multiple delimiters, we add up to 4 of them
        //:   to check that they are handled correctly.
        //:
        //: 2 Iterate over this table verifying that 'splitFilename' produces
        //:   the expected results with
        //:     - the explicit negative value of the 'rootEnd'
        //:     - the default value of the 'rootEnd'
        //:     - the explicit correct value of the 'rootEnd
        //:     - the 'head' being an address of 'path'
        //:     - the 'tail' being an address of 'path'  (C-1..8)
        //:
        //: 3 Verify that, in appropriate build modes, defensive checks are
        //:   triggered for invalid attribute values, but not triggered for
        //:   adjacent valid ones.  (C-9)
        //
        // Testing:
        //   void splitFilename(string_view*, string_view*,
        //                      const string_view&, int);
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "TESTING: splitFilename" << endl
                 << "======================" << endl;
        }

        test_splitFilename<bsl::string>(verbose, veryVerbose, veryVeryVerbose);
        test_splitFilename<std::string>(verbose, veryVerbose, veryVeryVerbose);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        test_splitFilename<std::pmr::string>(
            verbose, veryVerbose, veryVeryVerbose);
#endif
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING: 'appendIfValid'
        //
        // Concerns:
        //: 1 Return an error code is the appended path is an absolute path.
        //:
        //: 2 If 'path' does not contain trailing separators, 'filename' is
        //:   appended with the appropriate separator.
        //:
        //: 3 If 'path' does contain trailing separators, 'filename' is
        //:   appended with only a single separator.
        //:
        //: 4 If 'filename' contains trailing separators, they are not in
        //:   appended to 'path'.
        //:
        //: 5 If 'path' contains only separators it is simplified to a single
        //:   separator.
        //:
        //: 6 If 'filename' is an alias for any element in 'path', it is still
        //:   correctly appended.
        //:
        //: 7 Windows file names properly handle drive names in the absolute
        //:   path.
        //
        // Plan:
        //: 1 Create a table of test input values and expected results, and
        //:   iterate over this table verifying that 'appendIfValue' produces
        //:   the expected results.  (C-1, C-2, C-3, C-4, C-5, C-6)
        //:
        //: 1 Iterate over a series of simple test paths, and for each
        //:   path, iterate over a series of sub-string within that path.
        //:   For each sub-string, create a bsl::string_view aliasing that
        //:   sub-string, create an expected result value, and verify
        //:   that 'appendIfValid' also generates that expected value (C-7).
        //
        // Testing:
        //  int appendIfValid(bsl::string *, const bsl::string_view& );
        // --------------------------------------------------------------------

        if (verbose) {
            cout << "TESTING: appendIfValid" << endl
                 << "======================" << endl;
        }

        test_appendIfValid<bsl::string>(verbose, veryVerbose, veryVeryVerbose);
        test_appendIfValid<std::string>(verbose, veryVerbose, veryVeryVerbose);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        test_appendIfValid<std::pmr::string>(
            verbose, veryVerbose, veryVeryVerbose);
#endif
      } break;
      case 3: {
        ///////////////////////////////////////////////////////////////////////
        // Special path parsing test
        //
        // Concern: "root-end" determination for certain paths (WIN)
        //
        // Plan: Using the getRootEnd method, check parsing correctness
        //       for a few specific paths which caused issues during
        //       development
        ///////////////////////////////////////////////////////////////////////

        if (verbose) {
            cout << "Special path parsing test" << endl;
        }
#ifdef BSLS_PLATFORM_OS_WINDOWS
        int rootEnd = bdls::PathUtil::getRootEnd("\\\\machine\\dir\\");
        LOOP_ASSERT(rootEnd, 14 == rootEnd);
#endif
      } break;
      case 2: {
        ///////////////////////////////////////////////////////////////////////
        // Leafless append test
        //
        // Concern: Appending to the end of a path will not add a separator iff
        // there are no leaves.
        ///////////////////////////////////////////////////////////////////////

        if (verbose) {
           cout << "Leafless Append Test" << endl;
        }

        test_leaflessAppend<bsl::string>(
            verbose, veryVerbose, veryVeryVerbose);
        test_leaflessAppend<std::string>(
            verbose, veryVerbose, veryVeryVerbose);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        test_leaflessAppend<std::pmr::string>(
            verbose, veryVerbose, veryVeryVerbose);
#endif
      } break;
      case 1: {
        //////////////////////////////////////////////////////////////////////
        // Native Parsing Test
        //
        // Concerns: Functionality of native path parsing logic
        //
        // Plan: Construct objects from various native paths and verify
        // properties.  Also verify the parser by invoking it through the
        // appendIfValid code path in addition to the assignment path.  Also
        // verify correct parsing of the root.
        //////////////////////////////////////////////////////////////////////

        if (verbose) {
            cout << "Native Parsing Test" << endl;
        }

        test_nativeParsingTest<bsl::string>(
            verbose, veryVerbose, veryVeryVerbose);
        test_nativeParsingTest<std::string>(
            verbose, veryVerbose, veryVeryVerbose);
#ifdef BSLS_LIBRARYFEATURES_HAS_CPP17_PMR
        test_nativeParsingTest<std::pmr::string>(
            verbose, veryVerbose, veryVeryVerbose);
#endif
      } break;
      default: {
        cerr << "WARNING: CASE `" << test << "' NOT FOUND." << endl;
        testStatus = -1;
      }
    }

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "."
             << endl;
    }
    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2015 Bloomberg Finance L.P.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------- END-OF-FILE ----------------------------------
