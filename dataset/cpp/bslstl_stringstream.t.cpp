// bslstl_stringstream.t.cpp                                          -*-C++-*-
#include <bslstl_stringstream.h>

#include <bslstl_string.h>

#include <bslma_allocator.h>
#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bslma_stdallocator.h>
#include <bslma_stdtestallocator.h>
#include <bslma_testallocator.h>
#include <bslma_testallocatormonitor.h>

#include <bslmf_assert.h>

#include <bsls_bsltestutil.h>
#include <bsls_platform.h>

#include <bsltf_stdtestallocator.h>

#include <ios>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

//=============================================================================
//                                 TEST PLAN
//-----------------------------------------------------------------------------
//                                  Overview
//                                  --------
// 'bsl::basic_stringstream' IS-A 'std::basic_iostream' that uses
// 'bsl::basic_stringbuf' as an input/output buffer.  All of the functionality
// and state of a 'bsl::basic_stringstream' object are provided by base class
// sub-objects.  Therefore, it is not necessary to test all of the
// functionality derived from 'std::basic_iostream'.  It is sufficient to test
// that the various constructors initialize object state as expected, and that
// the manipulator and two accessors, all of which are trivial, work as
// expected.
//
// Note that the 'str' accessor may use the default allocator since it returns
// its string result by value; therefore, tests of allocator usage must take
// this into account.
//-----------------------------------------------------------------------------
// CREATORS
// [ 2] stringstream(const A& a = A());
// [ 3] stringstream(const A&& a);
// [ 5] stringstream(openmode mask, const A& a = A());
// [ 6] stringstream(const STRING& s, const A& a = A());
// [ 6] stringstream(MovableRef<STRING> s, const A& a = A());
// [ 7] stringstream(const STRING& s, openmode mask, const A& a = A());
// [ 7] stringstream(MovableRef<STRING> s, openmode mask, const A& a = A());
// [ 7] stringstream(const OTHER_STRING& s, const A& a = A());
// [ 7] stringstream(const OTHER_STRING& s, openmode mask, const A& a = A());
//
// MANIPULATORS
// [ 3] operator=(const A&& a);
// [ 4] void str(const StringType& value);
// [ 4] void str(MovableRef<StringType> value);
// [ 4] void str(const StringType<CHAR_TYPE, CHAR_TRAITS, SALLOC>& value);
//
// ACCESSORS
// [ 4] StringType str() const;
// [ 4] StringType str() &&;
// [ 4] StringType str(const SALLOC&);
// [ 4] ViewType view() const;
// [ 2] StreamBufType *rdbuf() const;
// ----------------------------------------------------------------------------
// [ 1] BREATHING TEST
// [ 9] USAGE EXAMPLE
// [ 8] CONCERN: Standard allocator can be used
// [ *] CONCERN: In no case does memory come from the global allocator.

//=============================================================================
//                  STANDARD BDE ASSERT TEST MACRO
//-----------------------------------------------------------------------------
// NOTE: THIS IS A LOW-LEVEL COMPONENT AND MAY NOT USE ANY C++ LIBRARY
// FUNCTIONS, INCLUDING IOSTREAMS.
static int testStatus = 0;

namespace {

void aSsErT(bool b, const char *s, int i) {
    if (b) {
        printf("Error " __FILE__ "(%d): %s    (failed)\n", i, s);
        if (testStatus >= 0 && testStatus <= 100) ++testStatus;
    }
}

}  // close unnamed namespace

//=============================================================================
//                       STANDARD BDE TEST DRIVER MACROS
//-----------------------------------------------------------------------------

#define ASSERT       BSLS_BSLTESTUTIL_ASSERT
#define LOOP_ASSERT  BSLS_BSLTESTUTIL_LOOP_ASSERT
#define LOOP0_ASSERT BSLS_BSLTESTUTIL_LOOP0_ASSERT
#define LOOP1_ASSERT BSLS_BSLTESTUTIL_LOOP1_ASSERT
#define LOOP2_ASSERT BSLS_BSLTESTUTIL_LOOP2_ASSERT
#define LOOP3_ASSERT BSLS_BSLTESTUTIL_LOOP3_ASSERT
#define LOOP4_ASSERT BSLS_BSLTESTUTIL_LOOP4_ASSERT
#define LOOP5_ASSERT BSLS_BSLTESTUTIL_LOOP5_ASSERT
#define LOOP6_ASSERT BSLS_BSLTESTUTIL_LOOP6_ASSERT
#define ASSERTV      BSLS_BSLTESTUTIL_ASSERTV

#define Q   BSLS_BSLTESTUTIL_Q   // Quote identifier literally.
#define P   BSLS_BSLTESTUTIL_P   // Print identifier and value.
#define P_  BSLS_BSLTESTUTIL_P_  // P(X) without '\n'.
#define T_  BSLS_BSLTESTUTIL_T_  // Print a tab (w/o newline).
#define L_  BSLS_BSLTESTUTIL_L_  // current Line number

#define RUN_EACH_TYPE BSLTF_TEMPLATETESTFACILITY_RUN_EACH_TYPE

// ============================================================================
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

// ============================================================================
//                       GLOBAL TEST VALUES
// ----------------------------------------------------------------------------

static bool             verbose;
static bool         veryVerbose;
static bool     veryVeryVerbose;
static bool veryVeryVeryVerbose;

//=============================================================================
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
//-----------------------------------------------------------------------------

typedef bsl::stringstream  Obj;
typedef bsl::wstringstream WObj;

typedef std::ios_base      IosBase;
typedef IosBase::openmode  Mode;

// ============================================================================
//                             GLOBAL TEST DATA
// ----------------------------------------------------------------------------

// Define the length of a 'bsl::string' value long enough to ensure dynamic
// memory allocation.

const int LENGTH_OF_SUFFICIENTLY_LONG_STRING =
#ifdef BSLS_PLATFORM_CPU_32_BIT
                                               33;
#else                 // 64_BIT
                                               65;
#endif

BSLMF_ASSERT(LENGTH_OF_SUFFICIENTLY_LONG_STRING >
                                        static_cast<int>(sizeof(bsl::string)));

struct StrlenDataRow {
    int  d_line;    // source line number
    int  d_length;  // string length
    char d_mem;     // expected allocation: 'Y', 'N', '?'
};

static
const StrlenDataRow STRLEN_DATA[] =
{
    //LINE  LENGTH                              MEM
    //----  ----------------------------------  ---
    { L_,   0,                                  'N'   },
    { L_,   1,                                  'N'   },
    { L_,   LENGTH_OF_SUFFICIENTLY_LONG_STRING, 'Y'   }
};
const int NUM_STRLEN_DATA = sizeof STRLEN_DATA / sizeof *STRLEN_DATA;

struct OpenModeDataRow {
    int  d_line;  // source line number
    Mode d_mode;
};

static
const OpenModeDataRow OPENMODE_DATA[] =
{
    //LINE  OPENMODE
    //----  ------------------------------------------
    { L_,   IosBase::in                                },
    { L_,   IosBase::out                               },
    { L_,   IosBase::in  | IosBase::out                },
    { L_,   IosBase::ate                               },
    { L_,   IosBase::in  | IosBase::ate                },
    { L_,   IosBase::out | IosBase::ate                },
    { L_,   IosBase::in  | IosBase::out | IosBase::ate }
};
const int NUM_OPENMODE_DATA = sizeof OPENMODE_DATA / sizeof *OPENMODE_DATA;

//=============================================================================
//                               TEST FACILITIES
//-----------------------------------------------------------------------------

namespace {

template <class StringT>
void loadString(StringT *value, int length)
    // Load into the specified 'value' a character string having the specified
    // 'length'.  The behavior is undefined unless 'length >= 0'.
{
    value->resize(length);

    for (int i = 0; i < length; ++i) {
        (*value)[i] =
                     static_cast<typename StringT::value_type>('a' + (i % 26));
    }
}

template <class StringT>
bool stringCouldBeMovedFrom(const StringT&                          s,
                            const typename StringT::allocator_type& otherAlloc)
    // Return 'true' if the specified string 's' could be moved to another
    // string that uses the specified allocator 'otherAlloc', and 'false'
    // otherwise.
{
    return s.size() >= LENGTH_OF_SUFFICIENTLY_LONG_STRING &&
           s.get_allocator() == otherAlloc;
}

template <class StringT>
bool stringWasMovedFrom(const StringT& s)
    // Return 'true' if the specified string 's' was possibly moved from, and
    // 'false' otherwise.
{
    return 0 == s.size();
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase2()
{
    // ------------------------------------------------------------------------
    // DEFAULT CTOR
    //   Ensure that we can use the default constructor to create an object
    //   having the expected default-constructed state, and use the primary
    //   manipulator to put that object into a state relevant for testing.
    //
    // Concerns:
    //: 1 An object created with the constructor under test has the specified
    //:   allocator.
    //:
    //: 2 The constructor allocates no memory.
    //:
    //: 3 Excepting the 'str' accessor, any memory allocation is from the
    //:   object allocator.
    //:
    //: 4 Excepting the 'str' accessor, there is no temporary allocation from
    //:   any allocator.
    //:
    //: 5 The 'rdbuf' accessor returns the expected (non-null) value.
    //:
    //: 6 The string buffer is initially empty.
    //:
    //: 7 The object is created with mode 'ios_base::in | ios_base::out' in
    //:   effect.
    //:
    //: 8 Every object releases any allocated memory at destruction.
    //
    // Plan:
    //   The table-driven technique is used.
    //
    //: 1 For each value, 'S', in a small set of string values:  (C-1..8)
    //:
    //:   1 For each allocator configuration:  (C-1..8)
    //:
    //:     1 Create an object using the default constructor and verify no
    //:       memory is allocated.  (C-2)
    //:
    //:     2 Use the 'rdbuf' accessor to verify that it returns the expected
    //:       (non-null) value.  (C-5)
    //:
    //:     3 Use the 'str' accessor to verify the initial value of the buffer
    //:       (empty).  (C-6)
    //:
    //:     4 For non-empty string values, invoke 'str(S)' and verify that the
    //:       correct allocator was used (if memory allocation is expected).
    //:       Additionally, verify the new value of the string buffer.
    //:       (C-1, 3)
    //:
    //:     5 Stream a character to the object (using the inherited '<<'
    //:       operator), then stream a character from the object (using the
    //:       inherited '>>' operator); verify that the results are as
    //:       expected.  (C-7)
    //:
    //:     6 Verify no temporary memory is allocated from the object allocator
    //:       when supplied.  (C-4)
    //:
    //:     7 Delete the object and verify all memory is deallocated.  (C-8)
    //
    // Testing:
    //   stringstream(const A& a = A());
    //   StreamBufType *rdbuf() const;
    // ------------------------------------------------------------------------

    if (verbose) printf("\nDEFAULT CTOR"
                        "\n============\n");

    using namespace BloombergLP;

    for (int ti = 0; ti < NUM_STRLEN_DATA; ++ti) {
        const int  LENGTH = STRLEN_DATA[ti].d_length;
        const char MEM    = STRLEN_DATA[ti].d_mem;

        for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
            const char CONFIG = cfg;  // how we specify the allocator

            bslma::TestAllocator da("default",   veryVeryVeryVerbose);
            bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
            bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

            bslma::DefaultAllocatorGuard dag(&da);

            if (veryVerbose) {
                printf("\nTesting with various allocator configurations.\n");
            }

            StreamT              *objPtr;
            bslma::TestAllocator *objAllocatorPtr;

            switch (CONFIG) {
              case 'a': {
                objPtr = new (fa) StreamT();
                objAllocatorPtr = &da;
              } break;
              case 'b': {
                objPtr = new (fa) StreamT(typename StreamT::allocator_type(0));
                objAllocatorPtr = &da;
              } break;
              case 'c': {
                objPtr = new (fa) StreamT(&sa);
                objAllocatorPtr = &sa;
              } break;
              default: {
                ASSERTV(CONFIG, !"Bad allocator config.");
                return;                                               // RETURN
              } break;
            }

            StreamT& mX = *objPtr;  const StreamT& X = mX;
                                    const BaseT&   B =  X;

            bslma::TestAllocator&  oa = *objAllocatorPtr;
            bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

            // Verify no allocation from the object/non-object allocators.

            ASSERTV(CONFIG,  oa.numBlocksTotal(), 0 ==  oa.numBlocksTotal());
            ASSERTV(CONFIG, noa.numBlocksTotal(), 0 == noa.numBlocksTotal());

            ASSERTV(CONFIG, X.rdbuf());
            ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
            ASSERTV(CONFIG, X.str().empty());

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            StringT mS(&scratch);  const StringT& S = mS;
            loadString(&mS, LENGTH);

            if (LENGTH > 0) {
                bslma::TestAllocatorMonitor oam(&oa);

                mX.str(S);

                ASSERTV(CONFIG, ('Y' == MEM) == oam.isInUseUp());

                oam.reset();

                ASSERTV(CONFIG, X.str() == S);

                if ('c' == CONFIG) {
                    ASSERTV(CONFIG, oam.isInUseSame());
                }
            }

            mX << 'X';

            if (S.empty()) { mS.resize(1); }
            mS[0] = static_cast<CharT>('X');

            ASSERTV(CONFIG, X.str() == S);

            CharT c = 'Z';
            mX >> c;

            ASSERTV(CONFIG, 'X'     == c);
            ASSERTV(CONFIG, X.str() == S);

            // Verify no temporary memory is allocated from the object
            // allocator when supplied.

            if ('c' == CONFIG) {
                ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                        oa.numBlocksTotal() == oa.numBlocksInUse());
            }

            // Reclaim dynamically allocated object under test.

            fa.deleteObject(objPtr);

            // Verify all memory is released on object destruction.

            ASSERTV(CONFIG,  fa.numBlocksInUse(), 0 ==  fa.numBlocksInUse());
            ASSERTV(CONFIG,  oa.numBlocksInUse(), 0 ==  oa.numBlocksInUse());
            ASSERTV(CONFIG, noa.numBlocksInUse(), 0 == noa.numBlocksInUse());
        }
    }
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase3()
{
    // ------------------------------------------------------------------------
    // MOVE CTOR AND ASSIGNMENT
    //
    // Concerns:
    //: 1 An object created with the move constructor or assigned with the
    //:   move-assignment operator has the state of the moved-from object.
    //:
    //: 2 Move construction/assignment allocates no memory.
    //:
    //: 3 The moved-from object is in a valid state.
    //
    // Plan:
    //:  1 Create a stream in input mode with a string to read from.
    //:
    //:  2 Read some from the string to have a state change in the 'BaseStream'
    //:    ('basic_iostream') and 'BaseType' ('StringBufContainer').
    //:
    //:  3 Move-construct/assign a stream.
    //:
    //:  4 Verify that the 'BaseType' ('StringBufContainer') has been moved.
    //:    The verification is done using the 'str()' member.
    //:
    //:  5 Verify that the 'BaseStream' ('basic_iostream') has been moved.  The
    //:    verification is done using the 'fail()' and 'tellg()' members.
    //:
    //:  6 Create a stream in output mode.
    //:
    //:  7 Write some string to have a state change in the 'BaseStream'
    //:    ('basic_iostream') and 'BaseType' ('StringBufContainer').
    //:
    //:  8 Move-construct/assign a stream.
    //:
    //:  9 Verify that the 'BaseType' ('StringBufContainer') has been moved.
    //:    The verification is done using the 'str()' member.
    //:
    //: 10 Verify that the 'BaseStream' ('basic_iostream') has been moved.  The
    //:    verification is done using the 'fail()' and 'tellp()' members.
    //
    // Testing:
    //   stringstream(const A&&);
    //   operator=(const A&& a);
    // ------------------------------------------------------------------------

    if (verbose) printf("\nMOVE CTOR AND ASSIGNMENT"
                        "\n========================\n");

#ifdef BSLS_LIBRARYFEATURES_HAS_CPP11_STREAM_MOVE
    using namespace BloombergLP;

    if (veryVerbose) {
        printf("\nTesting move construction with short string.\n");
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        static const size_t n = 3;
        StringT s(&sa);
        loadString(&s, n);
        StreamT movedFrom(s, IosBase::in, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        CharT b[n];
        movedFrom.read(b, 1);                 // state change in 'BaseType'
        ASSERTV(movedFrom.tellg(), 1 == movedFrom.tellg());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(std::move(movedFrom));

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellg(), 1 == movedTo.tellg());
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        StreamT movedFrom(IosBase::out, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        static const size_t n = 3;
        StringT s(&sa);
        loadString(&s, n);
        movedFrom.write(s.data(), n);         // state change in 'BaseType'
        ASSERTV(movedFrom.tellp(), n == movedFrom.tellp());

        static const typename StreamT::pos_type seekPos = 1;

        movedFrom.seekp(seekPos);
        ASSERTV(movedFrom.tellp(), seekPos == movedFrom.tellp());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(std::move(movedFrom));

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellp(), seekPos == movedTo.tellp());
    }

    if (veryVerbose) {
        printf("\nTesting move assignment with short string.\n");
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        static const size_t n = 3;
        StringT s(&sa);
        loadString(&s, n);
        StreamT movedFrom(s, IosBase::in, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        CharT b[n];
        movedFrom.read(b, 1);                 // state change in 'BaseType'
        ASSERTV(movedFrom.tellg(), 1 == movedFrom.tellg());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(&oa);
        movedTo = std::move(movedFrom);

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellg(), 1 == movedTo.tellg());
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        StreamT movedFrom(IosBase::out, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        static const size_t n = 3;
        StringT s(&sa);
        loadString(&s, n);
        movedFrom.write(s.data(), n);         // state change in 'BaseType'
        ASSERTV(movedFrom.tellp(), n == movedFrom.tellp());

        static const typename StreamT::pos_type seekPos = 1;

        movedFrom.seekp(seekPos);
        ASSERTV(movedFrom.tellp(), seekPos == movedFrom.tellp());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(&oa);
        movedTo = std::move(movedFrom);

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellp(), seekPos == movedTo.tellp());
    }

    if (veryVerbose) {
        printf("\nTesting move construction with long string.\n");
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        static const size_t n = LENGTH_OF_SUFFICIENTLY_LONG_STRING;
        StringT s(&sa);
        loadString(&s, n);
        StreamT movedFrom(s, IosBase::in, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        CharT b[n];
        movedFrom.read(b, 1);                 // state change in 'BaseType'
        ASSERTV(movedFrom.tellg(), 1 == movedFrom.tellg());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(std::move(movedFrom));

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellg(), 1 == movedTo.tellg());
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        StreamT movedFrom(IosBase::out, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        static const size_t n = LENGTH_OF_SUFFICIENTLY_LONG_STRING;
        StringT s(&sa);
        loadString(&s, n);
        movedFrom.write(s.data(), n);         // state change in 'BaseType'
        ASSERTV(movedFrom.tellp(), n == movedFrom.tellp());

        static const typename StreamT::pos_type seekPos = n - 4;

        movedFrom.seekp(seekPos);
        ASSERTV(movedFrom.tellp(), seekPos == movedFrom.tellp());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(std::move(movedFrom));

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellp(), seekPos == movedTo.tellp());
    }

    if (veryVerbose) {
        printf("\nTesting move assignment with long string.\n");
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        static const size_t n = LENGTH_OF_SUFFICIENTLY_LONG_STRING;
        StringT s(&sa);
        loadString(&s, n);
        StreamT movedFrom(s, IosBase::in, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        CharT b[n];
        movedFrom.read(b, 1);                 // state change in 'BaseType'
        ASSERTV(movedFrom.tellg(), 1 == movedFrom.tellg());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(&oa);
        movedTo = std::move(movedFrom);

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellg(), 1 == movedTo.tellg());
    }

    {
        bslma::TestAllocator oa("object", veryVeryVeryVerbose);
        bslma::TestAllocator sa("scratch", veryVeryVeryVerbose);
        StreamT movedFrom(IosBase::out, &oa);

        // Set some state for both 'BaseType' and 'BaseStream'

        static const size_t n = LENGTH_OF_SUFFICIENTLY_LONG_STRING;
        StringT s(&sa);
        loadString(&s, n);
        movedFrom.write(s.data(), n);         // state change in 'BaseType'
        ASSERTV(movedFrom.tellp(), n == movedFrom.tellp());

        static const typename StreamT::pos_type seekPos = n - 4;

        movedFrom.seekp(seekPos);
        ASSERTV(movedFrom.tellp(), seekPos == movedFrom.tellp());

        movedFrom.setstate(IosBase::failbit); // state change in 'BaseStream'
        ASSERT(movedFrom.fail());

        // Move the stream

        StreamT movedTo(&oa);
        movedTo = std::move(movedFrom);

        ASSERTV(movedTo.str().c_str(), s == movedTo.str());

        ASSERT(movedTo.fail());
        movedTo.clear();
        ASSERTV(movedTo.tellp(), seekPos == movedTo.tellp());
    }
#endif
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase4()
{
    // ------------------------------------------------------------------------
    // 'str' MANIPULATOR AND 'str' ACCESSOR
    //   Ensure that the 'str' manipulator and 'str' accessor work as expected.
    //
    // Concerns:
    //: 1 The 'str' manipulator has the expected effect on the contents of the
    //:   string buffer.
    //:
    //: 2 The 'str' accessor correctly reports the contents of the string
    //:   buffer.
    //
    // Plan:
    //   The table-driven technique is used.
    //
    //: 1 For each value, 'S', in a small set of string values:  (C-1..2)
    //:
    //:   1 Create an object using the default constructor (tested in case 2).
    //:
    //:   2 Use the 'str' accessor to verify the initial value of the buffer
    //:     (empty).  (C-2)
    //:
    //:   3 Use the 'str' manipulator to set the buffer value to 'S'.  (C-1)
    //:
    //:   4 Use the 'str' accessor to verify the new expected value of the
    //:     string buffer.  (C-2)
    //:
    //:   5 For each value, 'T', in a small set of string values:  (C-1..2)
    //:
    //:     1 Use the 'str' manipulator to set the buffer value to 'T'.  (C-1)
    //:
    //:     2 Use the 'str' accessor to verify the new expected value of the
    //:     string buffer.  (C-2)
    //
    // Testing:
    //   void str(const StringType& value);
    //   void str(MovableRef<StringType> value);
    //   void str(const StringType<CHAR_TYPE, CHAR_TRAITS, SALLOC>& value);
    //   StringType str() const;
    //   StringType str() &&;
    //   StringType str(const SALLOC&);
    //   ViewType view() const;
    // ------------------------------------------------------------------------

    if (verbose) printf("\n'str' MANIPULATOR AND 'str' ACCESSOR"
                        "\n====================================\n");

    using namespace BloombergLP;

    bslma::TestAllocator         da("default", veryVeryVeryVerbose);
    bslma::DefaultAllocatorGuard dag(&da);

    for (int ti = 0; ti < NUM_STRLEN_DATA; ++ti) {
        const int LENGTH_TI = STRLEN_DATA[ti].d_length;

        bslma::TestAllocator sa("supplied", veryVeryVeryVerbose);

        StreamT mX(&sa);  const StreamT& X = mX;   ASSERT(X.str().empty());

        StringT mS(&da);  const StringT& S = mS;
        loadString(&mS, LENGTH_TI);

        mX.str(S);
        ASSERT(X.str()  == S);
        ASSERT(X.view() == S);

        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            const int LENGTH_TJ = STRLEN_DATA[tj].d_length;

            StringT mT(&da);  const StringT& T = mT;
            loadString(&mT, LENGTH_TJ);

            mX.str(T);
            ASSERT(X.str()  == T);
            ASSERT(X.view() == T);
        }

        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            const int LENGTH_TJ = STRLEN_DATA[tj].d_length;

            typedef bsl::basic_string<
                                   CharT,
                                   typename StringT::traits_type,
                                   bslma::StdTestAllocator<CharT> > TestString;

            TestString oS;       const TestString& OS = oS;
            loadString(&oS, LENGTH_TJ);

            // same string as 'oS'; but easier to compare
            StringT mT(&da);  const StringT& T = mT;
            loadString(&mT, LENGTH_TJ);

            mX.str(OS);
            ASSERT(X.str()  == T);
            ASSERT(X.view() == T);
        }

#ifndef BSLS_PLATFORM_CMP_SUN
        // The call 'str(OtherAllocator) is not supported on SunOS, because the
        // std::allocator there does not support rebind.  This limitation can
        // be lifted once we fully support C++20, where rebind is removed, and
        // always goes through 'allocator_traits'.  See {DRQS 168075157} and
        // https://github.com/bloomberg/bde/pull/268
        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            typedef std::allocator<CharT> OtherAllocator;
            typedef bsl::basic_string<CharT,
                              typename StringT::traits_type,
                              OtherAllocator>                      OtherString;

            const int LENGTH_TJ = STRLEN_DATA[tj].d_length;

            OtherString oS;       const OtherString& OS = oS;
            loadString(&oS, LENGTH_TJ);

            // same string as 'oS'; but easier to compare
            StringT mT(&da);  const StringT& T = mT;
            loadString(&mT, LENGTH_TJ);

            mX.str(OS);
            ASSERT(X.str()  == T);
            ASSERT(X.view() == T);

            OtherString oS2 = X.str(OtherAllocator());
            ASSERT(oS2 == oS);
        }
#endif

        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            const int LENGTH_TJ = STRLEN_DATA[tj].d_length;

            StringT mT(&sa);     StringT& T = mT;
            StringT oT(&sa);
            loadString(&mT, LENGTH_TJ);
            loadString(&oT, LENGTH_TJ);

            mX.str(BloombergLP::bslmf::MovableRefUtil::move(T));
            ASSERT(X.str() == oT);
            ASSERT(X.view() == oT);
            // test strings that will be changed by a move
            if (LENGTH_TJ >= LENGTH_OF_SUFFICIENTLY_LONG_STRING) {
                ASSERT(T != oT);
            }
        }

#ifdef BSLS_COMPILERFEATURES_SUPPORT_REF_QUALIFIERS
        {
            StringT oT = X.str();
            ASSERT(X.str() == oT);
            ASSERT(X.view() == oT);
            StringT mT = std::move(mX).str();
            ASSERT(mT == oT);
            ASSERT(X.view().empty());
        }
#endif
    }
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase5()
{
    // ------------------------------------------------------------------------
    // OPENMODE CTOR
    //   Ensure that an object created using the constructor that takes an
    //   'openmode' mask has the expected initial state, and use the primary
    //   manipulator to put that object into a state relevant for testing.
    //
    // Concerns:
    //: 1 An object created with the constructor under test has the specified
    //:   allocator.
    //:
    //: 2 The constructor allocates no memory.
    //:
    //: 3 Excepting the 'str' accessor, any memory allocation is from the
    //:   object allocator.
    //:
    //: 4 Excepting the 'str' accessor, there is no temporary allocation from
    //:   any allocator.
    //:
    //: 5 The 'rdbuf' accessor returns the expected (non-null) value.
    //:
    //: 6 The string buffer is initially empty.
    //:
    //: 7 The object is created with the mode as specified in 'modeBitMask' in
    //:   effect.
    //:
    //: 8 Every object releases any allocated memory at destruction.
    //
    // Plan:
    //   The table-driven technique is used.
    //
    //: 1 For each value, 'M', in a representative set of mode bit-masks:
    //:   (C-1..8)
    //:
    //:   1 For each value, 'S', in a small set of string values:  (C-1..8)
    //:
    //:     1 For each allocator configuration:  (C-1..8)
    //:
    //:       1 Create an object, supplying 'M' to the constructor, and verify
    //:         no memory is allocated.  (C-2)
    //:
    //:       2 Use the 'rdbuf' accessor to verify that it returns the expected
    //:         (non-null) value.  (C-5)
    //:
    //:       3 Use the 'str' accessor to verify the initial value of the
    //:         buffer (empty).  (C-6)
    //:
    //:       4 For non-empty string values, invoke 'str(S)' and verify that
    //:         the correct allocator was used (if memory allocation is
    //:         expected).  Additionally, verify the new value of the string
    //:         buffer.  (C-1, 3)
    //:
    //:       5 Stream a character to the object (using the inherited '<<'
    //:         operator), then stream a character from the object (using the
    //:         inherited '>>' operator); verify that the results are as
    //:         expected.  (C-7)
    //:
    //:       6 Verify no temporary memory is allocated from the object
    //:         allocator when supplied.  (C-4)
    //:
    //:       7 Delete the object and verify all memory is deallocated.  (C-8)
    //
    // Testing:
    //   stringstream(openmode mask, const A& a = A());
    // ------------------------------------------------------------------------

    if (verbose) printf("\nOPENMODE CTOR"
                        "\n=============\n");

    using namespace BloombergLP;

    for (int ti = 0; ti < NUM_OPENMODE_DATA; ++ti) {
        const Mode MODE   = OPENMODE_DATA[ti].d_mode;
        const bool IN_OUT = (MODE & IosBase::in) || (MODE & IosBase::out);

        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            const int  LENGTH = STRLEN_DATA[tj].d_length;
            const char MEM    = STRLEN_DATA[tj].d_mem;

            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                if (veryVerbose) {
                    printf(
                         "\nTesting with various allocator configurations.\n");
                }

                StreamT              *objPtr;
                bslma::TestAllocator *objAllocatorPtr;

                switch (CONFIG) {
                  case 'a': {
                    objPtr = new (fa) StreamT(MODE);
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objPtr = new (fa) StreamT(
                                          MODE,
                                          typename StreamT::allocator_type(0));
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objPtr = new (fa) StreamT(MODE, &sa);
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    ASSERTV(CONFIG, !"Bad allocator config.");
                    return;                                           // RETURN
                  } break;
                }

                StreamT& mX = *objPtr;  const StreamT& X = mX;
                                        const BaseT&   B =  X;

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

                // Verify no allocation from the object/non-object allocators.

                ASSERTV(CONFIG,  oa.numBlocksTotal(),
                        0 ==  oa.numBlocksTotal());
                ASSERTV(CONFIG, noa.numBlocksTotal(),
                        0 == noa.numBlocksTotal());

                ASSERTV(CONFIG, X.rdbuf());
                ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
                ASSERTV(CONFIG, X.str().empty());

                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

                StringT mS(&scratch);  const StringT& S = mS;
                loadString(&mS, LENGTH);

                if (LENGTH > 0) {
                    bslma::TestAllocatorMonitor oam(&oa);

                    mX.str(S);

                    ASSERTV(CONFIG, ('Y' == MEM) == oam.isInUseUp());

                    oam.reset();

                    ASSERTV(CONFIG, !IN_OUT || X.str() == S);

                    if ('c' == CONFIG) {
                        ASSERTV(CONFIG, oam.isInUseSame());
                    }
                }

                if ((MODE & IosBase::out) || !(MODE & IosBase::in)) {
                    mX << 'X';  // '>>' test is perturbed if 'out' not set
                }

                if (MODE & IosBase::out) {
                    if (MODE & IosBase::ate) {
                        mS.push_back(static_cast<CharT>('X'));
                    }
                    else {
                        if (S.empty()) { mS.resize(1); }
                        mS[0] = static_cast<CharT>('X');
                    }
                }

                ASSERTV(CONFIG, !IN_OUT || X.str() == S);

                CharT c = 'Z';
                mX >> c;

                CharT EXPECTED;
                if (MODE & IosBase::in) {
                    EXPECTED = (0 == LENGTH) && !(MODE & IosBase::out)
                               ? 'Z'
                               : S[0];
                }
                else {
                    EXPECTED = 'Z';
                }

                ASSERTV(CONFIG, EXPECTED == c);
                ASSERTV(CONFIG, !IN_OUT || X.str()  == S);

                // Verify no temporary memory is allocated from the object
                // allocator when supplied.

                if ('c' == CONFIG && !(MODE & IosBase::ate)) {
                    ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                            oa.numBlocksTotal() == oa.numBlocksInUse());
                }

                // Reclaim dynamically allocated object under test.

                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.

                ASSERTV(CONFIG,  fa.numBlocksInUse(),
                        0 ==  fa.numBlocksInUse());
                ASSERTV(CONFIG,  oa.numBlocksInUse(),
                        0 ==  oa.numBlocksInUse());
                ASSERTV(CONFIG, noa.numBlocksInUse(),
                        0 == noa.numBlocksInUse());
            }
        }
    }
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase6()
{
    // ------------------------------------------------------------------------
    // STRING CTOR
    //   Ensure that an object created using the constructor that takes a
    //   string value has the expected initial state.
    //
    // Concerns:
    //: 1 An object created with the constructor under test has the specified
    //:   allocator.
    //:
    //: 2 The constructor allocates memory if and only if the string supplied
    //:   at construction is sufficiently long to guarantee memory allocation.
    //:
    //: 3 Excepting the 'str' accessor, any memory allocation is from the
    //:   object allocator.
    //:
    //: 4 Excepting the 'str' accessor, there is no temporary allocation from
    //:   any allocator.
    //:
    //: 5 The 'rdbuf' accessor returns the expected (non-null) value.
    //:
    //: 6 The string buffer initially has the same value as that of the string
    //:   supplied at construction.
    //:
    //: 7 The object is created with mode 'ios_base::in | ios_base::out' in
    //:   effect.
    //:
    //: 8 Every object releases any allocated memory at destruction.
    //
    // Plan:
    //   The table-driven technique is used.
    //
    //: 1 For each value, 'S', in a small set of string values:  (C-1..8)
    //:
    //:   1 For each allocator configuration:  (C-1..8)
    //:
    //:     1 Create an object, supplying 'S' to the constructor, and verify
    //:       that the correct allocator was used (if memory allocation is
    //:       expected).  Additionally, verify the initial value of the string
    //:       buffer.  (C-1..3, 6)
    //:
    //:     2 Use the 'rdbuf' accessor to verify that it returns the expected
    //:       (non-null) value.  (C-5)
    //:
    //:     3 Stream a character to the object (using the inherited '<<'
    //:       operator), then stream a character from the object (using the
    //:       inherited '>>' operator); verify that the results are as
    //:       expected.  (C-7)
    //:
    //:     4 Verify no temporary memory is allocated from the object allocator
    //:       when supplied.  (C-4)
    //:
    //:     5 Delete the object and verify all memory is deallocated.  (C-8)
    //
    // Testing:
    //   stringstream(const STRING& s, const A& a = A());
    //   stringstream(MovableRef<STRING> s, const A& a = A());
    //   stringstream(const OTHER_STRING& s, const A& a = A());
    // ------------------------------------------------------------------------

    if (verbose) printf("\nSTRING CTOR"
                        "\n===========\n");

    using namespace BloombergLP;

    for (int ti = 0; ti < NUM_STRLEN_DATA; ++ti) {
        const int  LENGTH = STRLEN_DATA[ti].d_length;
        const char MEM    = STRLEN_DATA[ti].d_mem;

        for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
            const char CONFIG = cfg;  // how we specify the allocator

            bslma::TestAllocator da("default",   veryVeryVeryVerbose);
            bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
            bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

            bslma::DefaultAllocatorGuard dag(&da);

            if (veryVerbose) {
                printf("\nTesting with various allocator configurations.\n");
            }

            StreamT              *objPtr;
            bslma::TestAllocator *objAllocatorPtr;

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            StringT mS(&scratch);  const StringT& S = mS;
            loadString(&mS, LENGTH);

            switch (CONFIG) {
              case 'a': {
                objPtr = new (fa) StreamT(S);
                objAllocatorPtr = &da;
              } break;
              case 'b': {
                objPtr = new (fa) StreamT(S,
                                          typename StreamT::allocator_type(0));
                objAllocatorPtr = &da;
              } break;
              case 'c': {
                objPtr = new (fa) StreamT(S, &sa);
                objAllocatorPtr = &sa;
              } break;
              default: {
                ASSERTV(CONFIG, !"Bad allocator config.");
                return;                                               // RETURN
              } break;
            }

            StreamT& mX = *objPtr;  const StreamT& X = mX;
                                    const BaseT&   B =  X;

            bslma::TestAllocator&  oa = *objAllocatorPtr;
            bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

            // Verify allocations from the object/non-object allocators.

            if ('N' == MEM) {
                ASSERTV(CONFIG,  oa.numBlocksTotal(),
                        0 ==  oa.numBlocksTotal());
                ASSERTV(CONFIG, noa.numBlocksTotal(),
                        0 == noa.numBlocksTotal());
            }
            else {
                ASSERTV(CONFIG,  oa.numBlocksTotal(),
                        0 !=  oa.numBlocksTotal());
            }

            ASSERTV(CONFIG, X.rdbuf());
            ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
            ASSERTV(CONFIG, X.str()   == S);

            mX << 'X';

            if (S.empty()) { mS.resize(1); }
            mS[0] = static_cast<CharT>('X');

            ASSERTV(CONFIG, X.str() == S);

            CharT c = 'Z';
            mX >> c;

            ASSERTV(CONFIG, 'X'     == c);
            ASSERTV(CONFIG, X.str() == S);

            // Verify no temporary memory is allocated from the object
            // allocator when supplied.

            if ('c' == CONFIG) {
                ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                        oa.numBlocksTotal() == oa.numBlocksInUse());
            }

            // Reclaim dynamically allocated object under test.

            fa.deleteObject(objPtr);

            // Verify all memory is released on object destruction.

            ASSERTV(CONFIG,  fa.numBlocksInUse(), 0 ==  fa.numBlocksInUse());
            ASSERTV(CONFIG,  oa.numBlocksInUse(), 0 ==  oa.numBlocksInUse());
            ASSERTV(CONFIG, noa.numBlocksInUse(), 0 == noa.numBlocksInUse());
        }

        // Once again, only with MovableRef
        for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
           typedef BloombergLP::bslmf::MovableRefUtil MoveUtil;

            const char CONFIG = cfg;  // how we specify the allocator

            bslma::TestAllocator da("default",   veryVeryVeryVerbose);
            bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
            bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

            bslma::DefaultAllocatorGuard dag(&da);

            if (veryVerbose) {
                printf("\nTesting with various allocator configurations.\n");
            }

            StreamT              *objPtr;
            bslma::TestAllocator *objAllocatorPtr;

            StringT mS(&da);  const StringT& S = mS;
            loadString(&mS, LENGTH);
            StringT mO(&da);  const StringT& O = mO;
            loadString(&mO, LENGTH);

            switch (CONFIG) {
              case 'a': {
                objPtr = new (fa) StreamT(MoveUtil::move(mS));
                objAllocatorPtr = &da;
              } break;
              case 'b': {
                objPtr = new (fa) StreamT(MoveUtil::move(mS),
                                          typename StreamT::allocator_type(0));
                objAllocatorPtr = &da;
              } break;
              case 'c': {
                objPtr = new (fa) StreamT(MoveUtil::move(mS), &sa);
                objAllocatorPtr = &sa;
              } break;
              default: {
                ASSERTV(CONFIG, !"Bad allocator config.");
                return;                                               // RETURN
              } break;
            }

            StreamT& mX = *objPtr;  const StreamT& X = mX;
                                    const BaseT&   B =  X;

            ASSERTV(CONFIG, X.rdbuf());
            ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
            ASSERTV(CONFIG, X.str()   == O);

            if (stringCouldBeMovedFrom(S, objAllocatorPtr)) {
                ASSERTV(CONFIG, O.size(), S.size(), stringWasMovedFrom(S));
            }

            mX << 'X';

            if (O.empty()) { mO.resize(1); }
            mO[0] = static_cast<CharT>('X');

            ASSERTV(CONFIG, X.str() == O);

            CharT c = 'Z';
            mX >> c;

            ASSERTV(CONFIG, 'X'     == c);
            ASSERTV(CONFIG, X.str() == O);

            // Reclaim dynamically allocated object under test.
            fa.deleteObject(objPtr);
        }

        // And with a different string allocator
        for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
            typedef bsl::basic_string<
                CharT,
                bsl::char_traits<CharT>,
                bslma::StdTestAllocator<CharT> > SallocString;

            const char CONFIG = cfg;  // how we specify the allocator

            bslma::TestAllocator da("default",   veryVeryVeryVerbose);
            bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
            bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

            bslma::DefaultAllocatorGuard dag(&da);

            if (veryVerbose) {
                printf(
                     "\nTesting with various allocator configurations.\n");
            }

            StreamT              *objPtr;
            bslma::TestAllocator *objAllocatorPtr;

            bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

            SallocString        mSallocStr(&scratch);
            const SallocString& sallocStr = mSallocStr;
            loadString(&mSallocStr, LENGTH);

            StringT mS(&scratch);  const StringT& S = mS;
            loadString(&mS, LENGTH);

            switch (CONFIG) {
              case 'a': {
                objPtr = new (fa) StreamT(sallocStr);
                objAllocatorPtr = &da;
              } break;
              case 'b': {
                objPtr = new (fa) StreamT(
                                      sallocStr,
                                      typename StreamT::allocator_type(0));
                objAllocatorPtr = &da;
              } break;
              case 'c': {
                objPtr = new (fa) StreamT(sallocStr, &sa);
                objAllocatorPtr = &sa;
              } break;
              default: {
                ASSERTV(CONFIG, !"Bad allocator config.");
                return;                                               // RETURN
              } break;
            }

            StreamT& mX = *objPtr;  const StreamT& X = mX;
                                    const BaseT&   B =  X;

            bslma::TestAllocator&  oa = *objAllocatorPtr;
            bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

            // Verify allocations from the object/non-object allocators.

            if ('N' == MEM) {
                ASSERTV(CONFIG,  oa.numBlocksTotal(),
                        0 ==  oa.numBlocksTotal());
                ASSERTV(CONFIG, noa.numBlocksTotal(),
                        0 == noa.numBlocksTotal());
            }
            else {
                ASSERTV(CONFIG,  oa.numBlocksTotal(),
                        0 !=  oa.numBlocksTotal());
            }

            ASSERTV(CONFIG, X.rdbuf());
            ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
            ASSERTV(CONFIG, X.str()   == S);

            mX << 'X';

            if (S.empty()) { mS.resize(1); }
            mS[0] = static_cast<CharT>('X');

            ASSERTV(CONFIG, X.str() == S);

            CharT c = 'Z';
            mX >> c;

            ASSERTV(CONFIG, 'X'     == c);
            ASSERTV(CONFIG, X.str() == S);

            // Verify no temporary memory is allocated from the object
            // allocator when supplied.

            if ('c' == CONFIG) {
                ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                        oa.numBlocksTotal() == oa.numBlocksInUse());
            }

            // Reclaim dynamically allocated object under test.

            fa.deleteObject(objPtr);

            // Verify all memory is released on object destruction.

            ASSERTV(CONFIG,  fa.numBlocksInUse(),
                    0 ==  fa.numBlocksInUse());
            ASSERTV(CONFIG,  oa.numBlocksInUse(),
                    0 ==  oa.numBlocksInUse());
            ASSERTV(CONFIG, noa.numBlocksInUse(),
                    0 == noa.numBlocksInUse());
        }
    }
}

template <class StreamT, class BaseT, class StringT, class CharT>
void testCase7()
{
    // ------------------------------------------------------------------------
    // STRING & OPENMODE CTOR
    //   Ensure that an object created using the constructor that takes a
    //   string value and 'openmode' mask has the expected initial state.
    //
    // Concerns:
    //: 1 An object created with the constructor under test has the specified
    //:   allocator.
    //:
    //: 2 The constructor allocates memory if and only if the string supplied
    //:   at construction is sufficiently long to guarantee memory allocation.
    //:
    //: 3 Excepting the 'str' accessor, any memory allocation is from the
    //:   object allocator.
    //:
    //: 4 Excepting the 'str' accessor, there is no temporary allocation from
    //:   any allocator.
    //:
    //: 5 The 'rdbuf' accessor returns the expected (non-null) value.
    //:
    //: 6 The string buffer initially has the same value as that of the string
    //:   supplied at construction.
    //:
    //: 7 The object is created with the mode as specified in 'modeBitMask' in
    //:   effect.
    //:
    //: 8 Every object releases any allocated memory at destruction.
    //
    // Plan:
    //   The table-driven technique is used.
    //
    //: 1 For each value, 'M', in a representative set of mode bit-masks:
    //:   (C-1..8)
    //:
    //:   1 For each value, 'S', in a small set of string values:  (C-1..8)
    //:
    //:     1 For each allocator configuration:  (C-1..8)
    //:
    //:       1 Create an object, supplying 'S' and 'M' to the constructor, and
    //:         verify that the correct allocator was used (if memory
    //:         allocation is expected).  Additionally, verify the initial
    //:         value of the string buffer.  (C-1..3, 6)
    //:
    //:       2 Use the 'rdbuf' accessor to verify that it returns the expected
    //:         (non-null) value.  (C-5)
    //:
    //:       3 Stream a character to the object (using the inherited '<<'
    //:         operator), then stream a character from the object (using the
    //:         inherited '>>' operator); verify that the results are as
    //:         expected.  (C-7)
    //:
    //:       4 Verify no temporary memory is allocated from the object
    //:         allocator when supplied.  (C-4)
    //:
    //:       5 Delete the object and verify all memory is deallocated.  (C-8)
    //
    // Testing:
    //   stringstream(const STRING& s, openmode mask, const A& a = A());
    //   stringstream(MovableRef<STRING> s, openmode mask, const A& a = A());
    //   stringstream(const OTHER_STRING& s, openmode mask, const A& a = A());
    // ------------------------------------------------------------------------

    if (verbose) printf("\nSTRING & OPENMODE CTOR"
                        "\n======================\n");

    using namespace BloombergLP;

    for (int ti = 3; ti < NUM_OPENMODE_DATA; ++ti) {
        const Mode MODE   = OPENMODE_DATA[ti].d_mode;
        const bool IN_OUT = (MODE & IosBase::in) || (MODE & IosBase::out);

        for (int tj = 0; tj < NUM_STRLEN_DATA; ++tj) {
            const int  LENGTH = STRLEN_DATA[tj].d_length;
            const char MEM    = STRLEN_DATA[tj].d_mem;

            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                if (veryVerbose) {
                    printf(
                         "\nTesting with various allocator configurations.\n");
                }

                StreamT              *objPtr;
                bslma::TestAllocator *objAllocatorPtr;

                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

                StringT mS(&scratch);  const StringT& S = mS;
                loadString(&mS, LENGTH);

                switch (CONFIG) {
                  case 'a': {
                    objPtr = new (fa) StreamT(S, MODE);
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objPtr = new (fa) StreamT(
                                          S,
                                          MODE,
                                          typename StreamT::allocator_type(0));
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objPtr = new (fa) StreamT(S, MODE, &sa);
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    ASSERTV(CONFIG, !"Bad allocator config.");
                    return;                                           // RETURN
                  } break;
                }

                StreamT& mX = *objPtr;  const StreamT& X = mX;
                                        const BaseT&   B =  X;

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

                // Verify allocations from the object/non-object allocators.

                if ('N' == MEM) {
                    ASSERTV(CONFIG,  oa.numBlocksTotal(),
                            0 ==  oa.numBlocksTotal());
                    ASSERTV(CONFIG, noa.numBlocksTotal(),
                            0 == noa.numBlocksTotal());
                }
                else {
                    ASSERTV(CONFIG,  oa.numBlocksTotal(),
                            0 !=  oa.numBlocksTotal());
                }

                ASSERTV(CONFIG, X.rdbuf());
                ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
                ASSERTV(CONFIG, ti, tj, !IN_OUT || X.str() == S);

                if ((MODE & IosBase::out) || !(MODE & IosBase::in)) {
                    mX << 'X';  // '>>' test is perturbed if 'out' not set
                }

                if (MODE & IosBase::out) {
                    if (MODE & IosBase::ate) {
                        mS.push_back(static_cast<CharT>('X'));
                    }
                    else {
                        if (S.empty()) { mS.resize(1); }
                        mS[0] = static_cast<CharT>('X');
                    }
                }

                ASSERTV(CONFIG, !IN_OUT || X.str() == S);

                CharT c = 'Z';
                mX >> c;

                CharT EXPECTED;
                if (MODE & IosBase::in) {
                    EXPECTED = (0 == LENGTH) && !(MODE & IosBase::out)
                               ? 'Z'
                               : S[0];
                }
                else {
                    EXPECTED = 'Z';
                }

                ASSERTV(CONFIG, EXPECTED == c);
                ASSERTV(CONFIG, !IN_OUT || X.str()  == S);

                // Verify no temporary memory is allocated from the object
                // allocator when supplied.

                if ('c' == CONFIG && !(MODE & IosBase::ate)) {
                    ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                            oa.numBlocksTotal() == oa.numBlocksInUse());
                }

                // Reclaim dynamically allocated object under test.

                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.

                ASSERTV(CONFIG,  fa.numBlocksInUse(),
                        0 ==  fa.numBlocksInUse());
                ASSERTV(CONFIG,  oa.numBlocksInUse(),
                        0 ==  oa.numBlocksInUse());
                ASSERTV(CONFIG, noa.numBlocksInUse(),
                        0 == noa.numBlocksInUse());
            }

            // Once again, only with MovableRef
            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
                typedef BloombergLP::bslmf::MovableRefUtil MoveUtil;

                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                if (veryVerbose) {
                    printf(
                         "\nTesting with various allocator configurations.\n");
                }

                StreamT              *objPtr;
                bslma::TestAllocator *objAllocatorPtr;

                StringT mS(&da);  const StringT& S = mS;
                loadString(&mS, LENGTH);
                StringT mO(&da);  const StringT& O = mO;
                loadString(&mO, LENGTH);

                switch (CONFIG) {
                  case 'a': {
                    objPtr = new (fa) StreamT(MoveUtil::move(mS), MODE);
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objPtr = new (fa) StreamT(
                                          MoveUtil::move(mS),
                                          MODE,
                                          typename StreamT::allocator_type(0));
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objPtr = new (fa) StreamT(MoveUtil::move(mS), MODE, &sa);
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    ASSERTV(CONFIG, !"Bad allocator config.");
                    return;                                           // RETURN
                  } break;
                }

                StreamT& mX = *objPtr;  const StreamT& X = mX;
                                        const BaseT&   B =  X;

                ASSERTV(CONFIG, X.rdbuf());
                ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
                ASSERTV(CONFIG, ti, tj, !IN_OUT || X.str() == O);

                if (stringCouldBeMovedFrom(S, objAllocatorPtr)) {
                    ASSERTV(CONFIG, O.size(), S.size(), stringWasMovedFrom(S));
                }

                if ((MODE & IosBase::out) || !(MODE & IosBase::in)) {
                    mX << 'X';  // '>>' test is perturbed if 'out' not set
                }

                if (MODE & IosBase::out) {
                    if (MODE & IosBase::ate) {
                        mO.push_back(static_cast<CharT>('X'));
                    }
                    else {
                        if (O.empty()) { mO.resize(1); }
                        mO[0] = static_cast<CharT>('X');
                    }
                }

                ASSERTV(CONFIG, !IN_OUT || X.str() == O);

                CharT c = 'Z';
                mX >> c;

                CharT EXPECTED;
                if (MODE & IosBase::in) {
                    EXPECTED = (0 == LENGTH) && !(MODE & IosBase::out)
                               ? 'Z'
                               : O[0];
                }
                else {
                    EXPECTED = 'Z';
                }

                ASSERTV(CONFIG, EXPECTED == c);
                ASSERTV(CONFIG, !IN_OUT || X.str()  == O);

                // Reclaim dynamically allocated object under test.

                fa.deleteObject(objPtr);
            }

            // And with a different string allocator
            for (char cfg = 'a'; cfg <= 'c'; ++cfg) {
                typedef bsl::basic_string<
                    CharT,
                    bsl::char_traits<CharT>,
                    bslma::StdTestAllocator<CharT> > SallocString;

                const char CONFIG = cfg;  // how we specify the allocator

                bslma::TestAllocator da("default",   veryVeryVeryVerbose);
                bslma::TestAllocator fa("footprint", veryVeryVeryVerbose);
                bslma::TestAllocator sa("supplied",  veryVeryVeryVerbose);

                bslma::DefaultAllocatorGuard dag(&da);

                if (veryVerbose) {
                    printf(
                         "\nTesting with various allocator configurations.\n");
                }

                StreamT              *objPtr;
                bslma::TestAllocator *objAllocatorPtr;

                bslma::TestAllocator scratch("scratch", veryVeryVeryVerbose);

                SallocString        mSallocStr(&scratch);
                const SallocString& sallocStr = mSallocStr;
                loadString(&mSallocStr, LENGTH);

                StringT mS(&scratch);  const StringT& S = mS;
                loadString(&mS, LENGTH);

                switch (CONFIG) {
                  case 'a': {
                    objPtr = new (fa) StreamT(sallocStr, MODE);
                    objAllocatorPtr = &da;
                  } break;
                  case 'b': {
                    objPtr = new (fa) StreamT(
                                          sallocStr,
                                          MODE,
                                          typename StreamT::allocator_type(0));
                    objAllocatorPtr = &da;
                  } break;
                  case 'c': {
                    objPtr = new (fa) StreamT(sallocStr, MODE, &sa);
                    objAllocatorPtr = &sa;
                  } break;
                  default: {
                    ASSERTV(CONFIG, !"Bad allocator config.");
                    return;                                           // RETURN
                  } break;
                }

                StreamT& mX = *objPtr;  const StreamT& X = mX;
                                        const BaseT&   B =  X;

                bslma::TestAllocator&  oa = *objAllocatorPtr;
                bslma::TestAllocator& noa = 'c' != CONFIG ? sa : da;

                // Verify allocations from the object/non-object allocators.

                if ('N' == MEM) {
                    ASSERTV(CONFIG,  oa.numBlocksTotal(),
                            0 ==  oa.numBlocksTotal());
                    ASSERTV(CONFIG, noa.numBlocksTotal(),
                            0 == noa.numBlocksTotal());
                }
                else {
                    ASSERTV(CONFIG,  oa.numBlocksTotal(),
                            0 !=  oa.numBlocksTotal());
                }

                ASSERTV(CONFIG, X.rdbuf());
                ASSERTV(CONFIG, X.rdbuf() == B.rdbuf());
                ASSERTV(CONFIG, ti, tj, !IN_OUT || X.str() == S);

                if ((MODE & IosBase::out) || !(MODE & IosBase::in)) {
                    mX << 'X';  // '>>' test is perturbed if 'out' not set
                }

                if (MODE & IosBase::out) {
                    if (MODE & IosBase::ate) {
                        mS.push_back(static_cast<CharT>('X'));
                    }
                    else {
                        if (S.empty()) { mS.resize(1); }
                        mS[0] = static_cast<CharT>('X');
                    }
                }

                ASSERTV(CONFIG, !IN_OUT || X.str() == S);

                CharT c = 'Z';
                mX >> c;

                CharT EXPECTED;
                if (MODE & IosBase::in) {
                    EXPECTED = (0 == LENGTH) && !(MODE & IosBase::out)
                               ? 'Z'
                               : S[0];
                }
                else {
                    EXPECTED = 'Z';
                }

                ASSERTV(CONFIG, EXPECTED == c);
                ASSERTV(CONFIG, !IN_OUT || X.str()  == S);

                // Verify no temporary memory is allocated from the object
                // allocator when supplied.

                if ('c' == CONFIG && !(MODE & IosBase::ate)) {
                    ASSERTV(CONFIG, oa.numBlocksTotal(), oa.numBlocksInUse(),
                            oa.numBlocksTotal() == oa.numBlocksInUse());
                }

                // Reclaim dynamically allocated object under test.

                fa.deleteObject(objPtr);

                // Verify all memory is released on object destruction.

                ASSERTV(CONFIG,  fa.numBlocksInUse(),
                        0 ==  fa.numBlocksInUse());
                ASSERTV(CONFIG,  oa.numBlocksInUse(),
                        0 ==  oa.numBlocksInUse());
                ASSERTV(CONFIG, noa.numBlocksInUse(),
                        0 == noa.numBlocksInUse());
            }
        }
    }
}

template <class CharT>
void testCase8()
{
    // ------------------------------------------------------------------------
    // CONCERN: Standard allocator can be used
    //
    // Concerns:
    //: 1 An object can be created using a standard-compliant allocator.
    //:
    //: 2 The object can successfully obtain memory from that allocator.
    //:
    //: 3 There is no allocation from either the default or global allocator.
    //
    // Plan:
    //: 1 For each of the four constructors, in turn:
    //:   1 Create a 'bslma::TestAllocator' object, and install it as the
    //:     default allocator (note that a ubiquitous test allocator is already
    //:     installed as the global allocator).
    //:
    //:   2 Create an object, using the constructor under test, that uses a
    //:     standard-compliant allocator with minimal features.  (C-1)
    //:
    //:   3 Use the 'str' accessor to verify the initial state of the string
    //:     buffer.  (C-2)
    //:
    //:   4 In the case of the two constructors that do not take a string
    //:     value, use the 'str' manipulator to set the buffer value to 'S', a
    //:     string of sufficient length to guarantee memory allocation; use the
    //:     'str' accessor to verify the new expected value of the string
    //:     buffer.  (C-2)
    //:
    //:   5 Use the test allocator from P-1 to verify that no memory is ever
    //:     allocated from the default allocator.  (C-3)
    //
    // Testing:
    //   CONCERN: Standard allocator can be used.
    // ------------------------------------------------------------------------

    if (verbose) printf("\nCONCERN: Standard allocator can be used"
                        "\n=======================================\n");

    using namespace BloombergLP;

    typedef bsltf::StdTestAllocator<CharT>                            StdAlloc;

    typedef bsl::basic_stringstream<CharT, bsl::char_traits<CharT>, StdAlloc>
                                                                        Stream;

    typedef bsl::basic_string<CharT, bsl::char_traits<CharT>, StdAlloc> String;

    const Mode MODE = IosBase::in | IosBase::out;

    {
        StdAlloc A;

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        String mS;     const String& S = mS;
        loadString(&mS, LENGTH_OF_SUFFICIENTLY_LONG_STRING);

        Stream mX(A);  const Stream& X = mX;           ASSERT(X.str().empty());

        mX.str(S);                                     ASSERT(X.str() == S);

        ASSERT(0 == da.numBlocksTotal());
    }

    {
        StdAlloc A;

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        String mS;     const String& S = mS;
        loadString(&mS, LENGTH_OF_SUFFICIENTLY_LONG_STRING);

        Stream mX(MODE, A);  const Stream& X = mX;     ASSERT(X.str().empty());

        mX.str(S);                                     ASSERT(X.str() == S);

        ASSERT(0 == da.numBlocksTotal());
    }

    {
        StdAlloc A;

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        String mS;        const String& S = mS;
        loadString(&mS, LENGTH_OF_SUFFICIENTLY_LONG_STRING);

        Stream mX(S, A);  const Stream& X = mX;        ASSERT(X.str() == S);

        ASSERT(0 == da.numBlocksTotal());
    }

    {
        StdAlloc A;

        bslma::TestAllocator         da("default", veryVeryVeryVerbose);
        bslma::DefaultAllocatorGuard dag(&da);

        String mS;     const String& S = mS;
        loadString(&mS, LENGTH_OF_SUFFICIENTLY_LONG_STRING);

        Stream mX(S, MODE, A);  const Stream& X = mX;  ASSERT(X.str() == S);

        ASSERT(0 == da.numBlocksTotal());
    }
}

}  // close unnamed namespace

//=============================================================================
//                               USAGE EXAMPLE
//-----------------------------------------------------------------------------

namespace {

///Usage
///-----
// This section illustrates intended use of this component.
//
///Example 1: Basic Input and Output Operations
/// - - - - - - - - - - - - - - - - - - - - - -
// The following example demonstrates the use of 'bsl::stringstream' to read
// and write data of various types to a 'bsl::string' object.
//
// Suppose we want to implement a simplified converter between a pair of
// generic types, 'TYPE1' and 'TYPE2'.  We use 'bsl::stringstream' to implement
// the 'lexicalCast' function.  We write the data of type 'TYPE1' into the
// stream with 'operator<<' and then read it back as the data of 'TYPE2' with
// 'operator>>':
//..
    template <class TYPE2, class TYPE1>
    TYPE2 lexicalCast(const TYPE1& what)
    {
        bsl::stringstream converter;
        converter << what;

        TYPE2 val;
        converter >> val;
        return val;
    }
//..

}  // close unnamed namespace

//=============================================================================
//                                MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test = argc > 1 ? atoi(argv[1]) : 0;
    verbose = argc > 2;
    veryVerbose = argc > 3;
    veryVeryVerbose = argc > 4;
    veryVeryVeryVerbose = argc > 5;

    using namespace BloombergLP;

    // CONCERN: In no case does memory come from the global allocator.

    bslma::TestAllocator globalAllocator("global", veryVeryVeryVerbose);
    bslma::Default::setGlobalAllocator(&globalAllocator);

    setbuf(stdout, NULL);    // Use unbuffered output.

    printf("TEST " __FILE__ " CASE %d\n", test);

    switch (test) { case 0:  // Zero is always the leading case.
      case 10: {
        // --------------------------------------------------------------------
        // TESTING READ/WRITE/SEEK COMBINATIONS
        //
        // Concerns:
        //: 1. Combination of read, write and seek operations doesn't affect
        //:    the consistency of the stream internal state.
        //
        // Plan:
        //: 1. Write to the stream, change the input position, then obtain the
        //:    current stream input position and verify that it's consistent
        //:    with what has been written to the stream.
        // --------------------------------------------------------------------

        if (verbose) printf("\nTESTING READ/WRITE/SEEK COMBINATIONS"
                            "\n====================================\n");

        bsl::stringstream inout;

        // Outputting a string goes through stringbuf::xsputn, but outputting a
        // character doesn't have to involve stringbuf and can be done just by
        // bumping the internal streambuf output pointer, so we do both.
        inout << "abc" << 'd' << 'e';

        // Now verify that we can seek in this stream
        inout.seekg(0, std::ios::beg);
        inout.seekg(0, std::ios::end);
        std::streamoff endPos = inout.tellg();

        ASSERT(inout.good());
        ASSERT(endPos == (std::streamoff) inout.str().size());

        // Verify that we can seek in the empty stream
        bsl::stringstream empty;
        empty.seekg(0, std::ios::beg);
        empty.seekg(0, std::ios::end);
        empty.seekg(0, std::ios::cur);
        empty.seekp(0, std::ios::beg);
        empty.seekp(0, std::ios::end);
        empty.seekp(0, std::ios::cur);

        ASSERT(empty.good());
        ASSERT(empty.tellg() == std::streampos(0));
        ASSERT(empty.tellp() == std::streampos(0));

        // Verify the output position after writing to a stream
        bsl::stringstream out2;
        bsl::string str2 =
            "sufficiently long string longer than the short string buffer";
        out2 << str2;
        std::streamoff endPos2 = out2.tellp();

        ASSERT(endPos2 == (std::streamoff) str2.size());
      } break;
      case 9: {
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

        if (verbose) printf("\nUSAGE EXAMPLE"
                            "\n=============\n");

// Finally, we verify that the 'lexicalCast' function works on some simple test
// cases:
//..
    ASSERT(lexicalCast<int>("1234") == 1234);
//
    ASSERT(lexicalCast<short>("-5") == -5);
//
    ASSERT(lexicalCast<bsl::string>("abc") == "abc");
//
    ASSERT(lexicalCast<bsl::string>(1234) == "1234");
//
    ASSERT(lexicalCast<short>(-5) == -5);
//..

      } break;
      case 8: {
        testCase8<char>();
        testCase8<wchar_t>();
      } break;
      case 7: {
        testCase7<Obj,  std::iostream,  bsl::string,  char>();
        testCase7<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 6: {
        testCase6<Obj,  std::iostream,  bsl::string,  char>();
        testCase6<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 5: {
        testCase5<Obj,  std::iostream,  bsl::string,  char>();
        testCase5<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 4: {
        testCase4<Obj,  std::iostream,  bsl::string,  char>();
        testCase4<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 3: {
        testCase3<Obj,  std::iostream,  bsl::string,  char>();
        testCase3<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 2: {
        testCase2<Obj,  std::iostream,  bsl::string,  char>();
        testCase2<WObj, std::wiostream, bsl::wstring, wchar_t>();
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // BREATHING TEST
        //   Developers' Sandbox.
        //
        // Concerns:
        //   We want to exercise basic functionality.
        //
        // Plan:
        //   This "test" *exercises* basic functionality, but *tests* nothing.
        // --------------------------------------------------------------------

        if (verbose) printf("\nBREATHING TEST"
                            "\n==============\n");

        Obj mX;  const Obj& X = mX;

        const std::iostream& B = X;

        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str().empty());

        const bsl::string S("ab");
        mX.str(S);

        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == S);

        char c = 'x';
        mX >> c;

        ASSERT('a' == c);
        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == S);

        mX << 'c';

        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == "cb");

        mX >> c;

        ASSERT('b' == c);
        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == "cb");

        const bsl::string T("x");
        mX.str(T);

        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == T);

        mX << 'y';

        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == "y");

        mX >> c;

        ASSERT('y' == c);
        ASSERT(!X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == "y");

        mX >> c;

        ASSERT('y' == c);
        ASSERT( X.eof());
        ASSERT( X.rdbuf());
        ASSERT( X.rdbuf() == B.rdbuf());
        ASSERT( X.str()   == "y");

      } break;
      default: {
        fprintf(stderr, "WARNING: CASE `%d' NOT FOUND.\n", test);
        testStatus = -1;
      }
    }

    // CONCERN: In no case does memory come from the global allocator.

    LOOP_ASSERT(globalAllocator.numBlocksTotal(),
                0 == globalAllocator.numBlocksTotal());

    if (testStatus > 0) {
        fprintf(stderr, "Error, non-zero test status = %d.\n", testStatus);
    }

    return testStatus;
}

// ----------------------------------------------------------------------------
// Copyright 2013 Bloomberg Finance L.P.
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
