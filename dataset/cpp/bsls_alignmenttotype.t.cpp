// bsls_alignmenttotype.t.cpp                                         -*-C++-*-

#include <bsls_alignmenttotype.h>
#include <bsls_alignmentimp.h>

#include <bsls_platform.h>

#include <algorithm>
#include <cstddef>     // offsetof() macro
#include <cstdlib>     // atoi()
#include <cstring>
#include <iostream>

using namespace BloombergLP;
using namespace std;

//=============================================================================
//                             TEST PLAN
//-----------------------------------------------------------------------------
//                             Overview
//                             --------
// Most of what this component implements are compile-time computations that
// differ among platforms.  The tests do assume that alignment of 'char' is 1,
// 'short' is 2, 'int' is 4, and 'double' is at least 4.  In addition, it is
// tested that all alignment-to-type calculations result are reversible, so
// that the alignment of the resulting type equals the original input.
//
// For the few run-time functions provided in this component, we establish
// post-conditions and test that the postconditions hold over a reasonable
// range of inputs.
//-----------------------------------------------------------------------------
// [ 1] bsls::AlignmentToType<N>::Type
// [ 2] DRQS 151904020
//-----------------------------------------------------------------------------
// [ 3] USAGE EXAMPLE -- Ensure the usage example compiles and works.
//=============================================================================

//-----------------------------------------------------------------------------
//                  STANDARD BDE ASSERT TEST MACRO
//-----------------------------------------------------------------------------

static int testStatus = 0;

static void aSsErT(int c, const char *s, int i) {
    if (c) {
        cout << "Error " << __FILE__ << "(" << i << "): " << s
             << "    (failed)" << endl;
        if (testStatus >= 0 && testStatus <= 100) ++testStatus;
    }
}
#define ASSERT(X) { aSsErT(!(X), #X, __LINE__); }

//=============================================================================
//                    STANDARD BDE LOOP-ASSERT TEST MACROS
//-----------------------------------------------------------------------------

#define LOOP_ASSERT(I,X) { \
    if (!(X)) { cout << #I << ": " << I << "\n"; aSsErT(1, #X, __LINE__);}}

#define LOOP2_ASSERT(I,J,X) { \
    if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " \
              << J << "\n"; aSsErT(1, #X, __LINE__); } }

#define LOOP3_ASSERT(I,J,K,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" \
              << #K << ": " << K << "\n"; aSsErT(1, #X, __LINE__); } }

#define LOOP4_ASSERT(I,J,K,L,X) { \
   if (!(X)) { cout << #I << ": " << I << "\t" << #J << ": " << J << "\t" << \
       #K << ": " << K << "\t" << #L << ": " << L << "\n"; \
       aSsErT(1, #X, __LINE__); } }

//=============================================================================
//                      SEMI-STANDARD TEST OUTPUT MACROS
//-----------------------------------------------------------------------------

#define P(X) cout << #X " = " << (X) << endl; // Print identifier and value.
#define Q(X) cout << "<| " #X " |>" << endl;  // Quote identifier literally.
#define P_(X) cout << #X " = " << (X) << ", " << flush; // P(X) without '\n'
#define A(X) cout << #X " = " << ((void *) X) << endl;  // Print address
#define A_(X) cout << #X " = " << ((void *) X) << ", " << flush;
#define L_ __LINE__                           // current Line number
#define TAB cout << '\t';

//=============================================================================
//                  GLOBAL DEFINITIONS FOR TESTING
//-----------------------------------------------------------------------------

namespace testCase2 {

template<class TYPE, int CAPACITY>
class StaticVector {
    // This 'class' template provides a container type holding a contiguous
    // sequence of elements of parameterized 'TYPE' with fixed maximum
    // 'CAPACITY'.

  private:
    // DATA
    TYPE d_data[CAPACITY];  // element storage
    int  d_size;            // current size

  public:
    // CREATORS
    StaticVector()
        // Create a 'StaticVector' with size '0'.
    : d_size(0)
    {
    }

    TYPE *begin()
        // Return the pointer to the beginning of the range of elements.
    {
        return d_data;
    }

    TYPE *end()
        // Return the pointer to the end of the range of elements.
    {
        return d_data + d_size;
    }

    int size() const
        // Return the number of elements.
    {
        return d_size;
    }

    void push_back(const TYPE& value)
        // Add the specified 'value' to the end of range of elements.  The
        // behavior is undefined unless 'size() < CAPACITY'.
    {
        ASSERT(d_size < CAPACITY);
        d_data[d_size++] = value;
    }

    void resize(int size)
        // Resize this vector to the specified 'size'.  The behavior is
        // undefined unless '0 <= size < CAPACITY'.
    {
        ASSERT(0 <= size);
        ASSERT(size < CAPACITY);
        d_size = size;
    }
};

class DestructionTestType {
    // This empty 'class' ensures that the destructor is only called on
    // instances that were properly created with one of its constructors.

  private:
    // CLASS DATA
    enum { k_MAX_NUM_INSTANCES = 32 };

    static StaticVector<DestructionTestType *, k_MAX_NUM_INSTANCES>
                                s_instances;
        // set of known instances of this 'class'

  public:
    // CREATORS
    DestructionTestType()
        // Create an instance and add it to the list of known instances.
    {
        s_instances.push_back(this);
    }
    DestructionTestType(const DestructionTestType&)
        // Create a copy instance and add it to the list of known instances.
    {
        s_instances.push_back(this);
    }
    ~DestructionTestType()
        // Assert that this instance was previously created and destroy it.
    {
        int numRemoved = static_cast<int>(
            s_instances.end() -
            std::remove(s_instances.begin(), s_instances.end(), this));

        ASSERT(1 == numRemoved);
        s_instances.resize(s_instances.size() - numRemoved);
    }
};

StaticVector<DestructionTestType *, DestructionTestType::k_MAX_NUM_INSTANCES>
    DestructionTestType::s_instances;

template<int ALIGNMENT>
struct ProblematicStruct {
    typename bsls::AlignmentToType<ALIGNMENT>::Type d_aligmentToType;
    DestructionTestType                             d_testType;
};

template<int ALIGNMENT>
void byValueFunction(ProblematicStruct<ALIGNMENT>)
    // Empty function accepting 'ProblematicStruct ' *by* *value* (required to
    // reproduce DRQS 151904020).
{
}

} // close namespace testCase2

//=============================================================================
//                             USAGE EXAMPLE
//-----------------------------------------------------------------------------

namespace Usage {

///Usage
///-----
// Consider a parameterized type, 'my_AlignedBuffer', that provides aligned
// memory to store a user-defined type.  A 'my_AlignedBuffer' object is useful
// in situations where efficient (e.g., stack-based) storage is required.
//
// The 'my_AlignedBuffer' 'union' (defined below) takes a 'TYPE' and the
// 'ALIGNMENT' requirements for that type as template parameters, and provides
// an appropriately sized and aligned block of memory via the 'buffer'
// functions.  Note that 'my_AlignedBuffer' ensures that the returned memory is
// aligned correctly for the specified size by using
// 'bsls::AlignmentToType<ALIGNMENT>::Type', which provides a primitive type
// having the 'ALIGNMENT' requirement.  The class definition of
// 'my_AlignedBuffer' is as follows:
//..
    template <class TYPE, int ALIGNMENT>
    union my_AlignedBuffer {
      private:
        // DATA
        char                                           d_buffer[sizeof(TYPE)];
        typename bsls::AlignmentToType<ALIGNMENT>::Type d_align;  // force
                                                                  // alignment

      public:
        // MANIPULATORS
        char *buffer();
            // Return the address of the modifiable first byte of memory
            // contained by this object as a 'char *' pointer.

        TYPE& object();
            // Return a reference to the modifiable 'TYPE' object stored in
            // this buffer.  The referenced object has an undefined state
            // unless a valid 'TYPE' object has been constructed in this
            // buffer.

        // ACCESSORS
        const char *buffer() const;
            // Return the address of the non-modifiable first byte of memory
            // contained by this object as a 'const char *' pointer.

        const TYPE& object() const;
            // Return a reference to the non-modifiable 'TYPE' object stored in
            // this buffer.  The referenced object has an undefined state
            // unless a valid 'TYPE' object has been constructed in this
            // buffer.
    };
//..
// The function definitions of 'my_AlignedBuffer' are as follows:
//..
    // MANIPULATORS
    template <class TYPE, int ALIGNMENT>
    inline
    char *my_AlignedBuffer<TYPE, ALIGNMENT>::buffer()
    {
        return d_buffer;
    }

    template <class TYPE, int ALIGNMENT>
    inline
    TYPE& my_AlignedBuffer<TYPE, ALIGNMENT>::object()
    {
        return *reinterpret_cast<TYPE *>(this);
    }

    // ACCESSORS
    template <class TYPE, int ALIGNMENT>
    inline
    const char *my_AlignedBuffer<TYPE, ALIGNMENT>::buffer() const
    {
        return d_buffer;
    }

    template <class TYPE, int ALIGNMENT>
    inline
    const TYPE& my_AlignedBuffer<TYPE, ALIGNMENT>::object() const
    {
        return *reinterpret_cast<const TYPE *>(this);
    }
//..
// 'my_AlignedBuffer' can be used to construct buffers for different types and
// with varied alignment requirements.  Consider that we want to construct an
// object that stores the response of a floating-point operation.  If the
// operation is successful, then the response object stores a 'double' result;
// otherwise, it stores an error string of type 'string', which is based on the
// standard type 'string' (see 'bslstl_string').  For the sake of brevity, the
// implementation of 'string' is not explored here.  Here is the definition for
// the 'Response' class:
//..
class string {

    // DATA
    char            *d_value_p;      // 0 terminated character array
    std::size_t      d_size;         // length of d_value_p

     // PRIVATE CLASS CONSTANTS
    static const char *EMPTY_STRING;

  public:
    // CREATORS
    explicit string()
    : d_value_p(const_cast<char *>(EMPTY_STRING))
    , d_size(0)
    {
    }

    string(const char *value)
    : d_value_p(const_cast<char *>(EMPTY_STRING))
    , d_size(std::strlen(value))
    {
        if (d_size > 0) {
            d_value_p = new char[d_size + 1];
            std::memcpy(d_value_p, value, d_size + 1);
        }
    }

    string(const string& original)
    : d_value_p(const_cast<char *>(EMPTY_STRING))
    , d_size(original.d_size)
    {
        if (d_size > 0) {
            d_value_p = new char[d_size + 1];
            std::memcpy(d_value_p, original.d_value_p, d_size + 1);
        }
    }

    ~string()
    {
        if (d_size > 0) {
            delete[] d_value_p;
        }
    }

    // MANIPULATORS
    string& operator=(const string& rhs) {
        string temp(rhs);
        temp.swap(*this);
        return *this;
    }

    char &operator[](int index)
    {
        return d_value_p[index];
    }

    void swap(string& other)
    {
        std::swap(d_value_p, other.d_value_p);
        std::swap(d_size, other.d_size);
    }

    // ACCESSORS
    std::size_t size() const
    {
        return d_size;
    }

    bool empty() const
    {
        return 0 == d_size;
    }

    const char *c_str() const
    {
        return d_value_p;
    }
};

inline
bool operator==(const string& lhs, const string& rhs)
{
    return 0 == std::strcmp(lhs.c_str(), rhs.c_str());
}

inline
bool operator!=(const string& lhs, const string& rhs)
{
    return !(lhs == rhs);
}

inline
bool operator<(const string& lhs, const string& rhs)
{
    return std::strcmp(lhs.c_str(), rhs.c_str()) < 0;
}

inline
bool operator>(const string& lhs, const string& rhs)
{
    return rhs < lhs;
}

const char *string::EMPTY_STRING = "";

    class Response {
//..
// To create a 'my_AlignedBuffer' object we must specify the alignment value
// for our types.  For simplicity, we use a maximum alignment value for all
// types (assumed to be 8 here):
//..
      enum { MAX_ALIGNMENT = 8 };
//..
// Note that we use 'my_AlignedBuffer' to allocate sufficient, aligned memory
// to store the result of the operation or an error message:
//..
      private:
        union {
            my_AlignedBuffer<double, MAX_ALIGNMENT>      d_result;
            my_AlignedBuffer<string, MAX_ALIGNMENT> d_errorMessage;
        };
//..
// The 'isError' flag indicates whether the response object stores valid data
// or an error message:
//..
        bool d_isError;
//..
// Below we provide a simple public interface suitable for illustration only:
//..
      public:
        // CREATORS
        Response(double result);
            // Create a response object that stores the specified 'result'.

        Response(const string& errorMessage);
            // Create a response object that stores the specified
            // 'errorMessage'.

        ~Response();
            // Destroy this response object.
//..
// The manipulator functions allow clients to update the response object to
// store either a 'double' result or an error message:
//..
        // MANIPULATORS
        void setResult(double result);
            // Update this object to store the specified 'result'.  After this
            // operation 'isError' returns 'false'.

        void setErrorMessage(const string& errorMessage);
            // Update this object to store the specified 'errorMessage'.  After
            // this operation 'isError' returns 'true'.
//..
// The 'isError' function informs clients whether a response object stores a
// result value or an error message:
//..
        // ACCESSORS
        bool isError() const;
            // Return 'true' if this object stores an error message, and
            // 'false' otherwise.

        double result() const;
            // Return the result value stored by this object.  The behavior is
            // undefined unless 'false == isError()'.

        const string& errorMessage() const;
            // Return a reference to the non-modifiable error message stored by
            // this object.  The behavior is undefined unless
            // 'true == isError()'.
    };
//..
// Below we provide the function definitions.  Note that we use the
// 'my_AlignedBuffer::buffer' function to access correctly aligned memory.
// Also note that 'my_AlignedBuffer' just provides the memory for an object;
// therefore, the 'Response' class is responsible for the construction and
// destruction of the specified objects.  Since our 'Response' class is for
// illustration purposes only, we ignore exception-safety concerns; nor do we
// supply an allocator to the string constructor, allowing the default
// allocator to be used instead:
//..
    // creators
    Response::Response(double result)
    {
        new (d_result.buffer()) double(result);
        d_isError = false;
    }

    Response::Response(const string& errorMessage)
    {
        new (d_errorMessage.buffer()) string(errorMessage);
        d_isError = true;
    }

    Response::~Response()
    {
        if (d_isError) {
            typedef string Type;
            d_errorMessage.object().~Type();
        }
    }

    // MANIPULATORS
    void Response::setResult(double result)
    {
        if (!d_isError) {
            d_result.object() = result;
        }
        else {
            typedef string Type;
            d_errorMessage.object().~Type();
            new (d_result.buffer()) double(result);
            d_isError = false;
        }
    }

    void Response::setErrorMessage(const string& errorMessage)
    {
        if (d_isError) {
            d_errorMessage.object() = errorMessage;
        }
        else {
            new (d_errorMessage.buffer()) string(errorMessage);
            d_isError = true;
        }
    }

    // ACCESSORS
    bool Response::isError() const
    {
        return d_isError;
    }

    double Response::result() const
    {
        ASSERT(!d_isError);

        return d_result.object();
    }

    const string& Response::errorMessage() const
    {
        ASSERT(d_isError);

        return d_errorMessage.object();
    }
//..

}  // close unnamed namespace

//=============================================================================
//                              MAIN PROGRAM
//-----------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    int test = argc > 1 ? atoi(argv[1]) : 0;
    int verbose = argc > 2;
    int veryVerbose = argc > 3;

    (void) veryVerbose;

    cout << "TEST " << __FILE__ << " CASE " << test << endl;

    switch (test) { case 0:
      case 3: {
        // --------------------------------------------------------------------
        // USAGE TEST
        //   Make sure main usage examples compile and work as advertized.
        //
        // Test plan:
        //   Copy usage example verbatim into test driver then change 'assert'
        //   to 'ASSERT'.  Since usage example code declares template classes
        //   and functions, it must be placed outside of main().  Within this
        //   test case, therefore, we call the externally-declared functions
        //   defined above.
        //
        // Testing:
        //   USAGE EXAMPLE -- Ensure the usage example compiles and works.
        // --------------------------------------------------------------------

        if (verbose) cout << "\nUSAGE" << endl
                          << "\n=====" << endl;

        using namespace Usage;

// Clients of the 'Response' class can use it as follows:
//..
    double value1 = 111.2, value2 = 92.5;

    if (0 == value2) {
        Response response("Division by 0");

        // Return erroneous response
    }
    else {
        Response response(value1 / value2);

        // Process response object
    }
//..
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING RESOLUTION OF DRQS 151904020
        //
        // CONCERNS
        //   'AlignmentToType' does not trigger a MSVC compiler bug manifesting
        //   in the destructor being called with incorrect 'this' pointer.
        //
        // PLAN
        //   1. Pass by-value a struct containing pointer to member function of
        //      a class with at least two bases and a
        //      'bsls::AlignmentToType<N>::Type' object.
        //
        //   2. Verify that the destructors are called with correct 'this'
        //      pointers by means of 'DestructorTestType' member of
        //      'ProblematicStruct'.
        //
        // TESTING
        //   bsls::AlignmentToType<N>::Type
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTest DRQS 151904020 resolution"
                          << "\n==============================" << endl;


        using namespace testCase2;

        byValueFunction(ProblematicStruct<1>());
        byValueFunction(ProblematicStruct<2>());
        byValueFunction(ProblematicStruct<4>());
        byValueFunction(ProblematicStruct<8>());
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // TESTING META-FUNCTIONS  bsls::AlignmentToType<N>::Type
        //
        // PLAN
        //   1. Compute alignment type using bsls::AlignmentToType<N>::Type and
        //      confirm that the result is as expected.
        //
        // TACTICS
        //   Ad-hoc data selection
        //   Area test using primitive types.
        //
        // TESTING
        //   bsls::AlignmentToType<N>::Type
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTest bsls::AlignmentToType<T>::Type"
                          << "\n===================================" << endl;

        typedef void (*FuncPtr)();

        enum {
            CHAR_ALIGNMENT        = bsls::AlignmentImpCalc<char>::VALUE,
            SHORT_ALIGNMENT       = bsls::AlignmentImpCalc<short>::VALUE,
            INT_ALIGNMENT         = bsls::AlignmentImpCalc<int>::VALUE,
            LONG_ALIGNMENT        = bsls::AlignmentImpCalc<long>::VALUE,
            INT64_ALIGNMENT       = bsls::AlignmentImpCalc<long long>::VALUE,
            BOOL_ALIGNMENT        = bsls::AlignmentImpCalc<bool>::VALUE,
            WCHAR_T_ALIGNMENT     = bsls::AlignmentImpCalc<wchar_t>::VALUE,
            PTR_ALIGNMENT         = bsls::AlignmentImpCalc<void*>::VALUE,
            FUNC_PTR_ALIGNMENT    = bsls::AlignmentImpCalc<FuncPtr>::VALUE,
            FLOAT_ALIGNMENT       = bsls::AlignmentImpCalc<float>::VALUE,
            DOUBLE_ALIGNMENT      = bsls::AlignmentImpCalc<double>::VALUE,
            LONG_DOUBLE_ALIGNMENT = bsls::AlignmentImpCalc<long double>::VALUE
        };

#undef  CHECK_ALIGNMENT
#define CHECK_ALIGNMENT(alignment, type)                                      \
        LOOP_ASSERT(alignment,                                                \
                    0 + bsls::AlignmentImpCalc<type>::VALUE ==                \
                    0 + bsls::AlignmentImpCalc<                               \
                        bsls::AlignmentToType<alignment>::Type>::VALUE)

        CHECK_ALIGNMENT(CHAR_ALIGNMENT, char);
        CHECK_ALIGNMENT(SHORT_ALIGNMENT, short);
        CHECK_ALIGNMENT(INT_ALIGNMENT, int);
        CHECK_ALIGNMENT(BOOL_ALIGNMENT, char);
        CHECK_ALIGNMENT(FLOAT_ALIGNMENT, int);

#if (defined(BSLS_PLATFORM_OS_AIX) && !defined(BSLS_PLATFORM_CPU_64_BIT))   \
 ||  defined(BSLS_PLATFORM_OS_WINDOWS) || defined(BSLS_PLATFORM_OS_CYGWIN)
        CHECK_ALIGNMENT(WCHAR_T_ALIGNMENT, short);
#else
        CHECK_ALIGNMENT(WCHAR_T_ALIGNMENT, int);
#endif

#if defined(BSLS_PLATFORM_OS_LINUX) || defined(BSLS_PLATFORM_OS_DARWIN)
    #if defined(BSLS_PLATFORM_CPU_64_BIT)
        CHECK_ALIGNMENT(INT64_ALIGNMENT, long);
        CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long);
        #if defined(BSLS_PLATFORM_CPU_POWERPC) \
         && defined(BSLS_PLATFORM_OS_LINUX)
            CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long);
        #else
            CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long double);
        #endif
    #else
        #if defined(BSLS_PLATFORM_CPU_ARM)
            CHECK_ALIGNMENT(INT64_ALIGNMENT, long long);
            CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long long);
        #elif defined(BSLS_PLATFORM_CPU_POWERPC) \
           && defined(BSLS_PLATFORM_OS_LINUX)
            CHECK_ALIGNMENT(INT64_ALIGNMENT, long long);
            CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long long);
        #else
            CHECK_ALIGNMENT(INT64_ALIGNMENT, int);
            CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, int);
        #endif

        #if defined(BSLS_PLATFORM_OS_LINUX)
            #if defined(BSLS_PLATFORM_CPU_ARM)
                CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long long);
            #elif defined(BSLS_PLATFORM_CPU_POWERPC)
                CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long long);
            #else
                CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, int);
            #endif
        #else
            CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long double);
        #endif
    #endif
#elif defined(BSLS_PLATFORM_OS_AIX)
    #if defined(BSLS_PLATFORM_CPU_64_BIT)
        CHECK_ALIGNMENT(INT64_ALIGNMENT, long);
        CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, int);
        CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, int);
    #else
        CHECK_ALIGNMENT(INT64_ALIGNMENT, long long);
        CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, int);
        CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, int);
    #endif
#elif defined(BSLS_PLATFORM_OS_CYGWIN)
    #if defined(BSLS_PLATFORM_CPU_64_BIT)
        // TBD
    #else
        CHECK_ALIGNMENT(INT64_ALIGNMENT, bsls::AlignmentImp8ByteAlignedType);
        CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, bsls::AlignmentImp8ByteAlignedType);
        CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, int);
    #endif
#else // NOT AIX, Linux, Darwin, or Cygwin
    #if defined(BSLS_PLATFORM_CPU_64_BIT)
        #if defined(BSLS_PLATFORM_OS_WINDOWS)
            CHECK_ALIGNMENT(INT64_ALIGNMENT, long long);
            CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long long);
            CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long long);
        #else
            CHECK_ALIGNMENT(INT64_ALIGNMENT, long);
            CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long);
            CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long double);
        #endif
    #else
        CHECK_ALIGNMENT(INT64_ALIGNMENT, long long);
        CHECK_ALIGNMENT(DOUBLE_ALIGNMENT, long long);
        CHECK_ALIGNMENT(LONG_DOUBLE_ALIGNMENT, long long);
    #endif
#endif // end defined(BSLS_PLATFORM_OS_AIX)
       //  || defined(BSLS_PLATFORM_OS_LINUX)

#if defined(BSLS_PLATFORM_CPU_64_BIT)
    #if defined(BSLS_PLATFORM_OS_WINDOWS)
        CHECK_ALIGNMENT(LONG_ALIGNMENT, int);
        CHECK_ALIGNMENT(PTR_ALIGNMENT, long long);
        CHECK_ALIGNMENT(FUNC_PTR_ALIGNMENT, long long);
    #else
        CHECK_ALIGNMENT(LONG_ALIGNMENT, long);
        CHECK_ALIGNMENT(PTR_ALIGNMENT, long);
        CHECK_ALIGNMENT(FUNC_PTR_ALIGNMENT, long);
    #endif
#else // !defined(BSLS_PLATFORM_CPU_64_BIT)
    CHECK_ALIGNMENT(LONG_ALIGNMENT, int);
    CHECK_ALIGNMENT(PTR_ALIGNMENT, int);
    CHECK_ALIGNMENT(FUNC_PTR_ALIGNMENT, int);
#endif // end defined(BSLS_PLATFORM_CPU_64_BIT)

#undef  CHECK_ALIGNMENT

#ifdef BSLS_ALIGNMENTTOTYPE_USES_ALIGNAS

#undef  CHECK_ALIGNMENT
#define CHECK_ALIGNMENT(N)                                                    \
    do {                                                                      \
        class alignas(N) X { };                                               \
        LOOP_ASSERT(N, 0 + alignof(X) == 0 + bsls::AlignmentImpCalc<          \
                                     bsls::AlignmentToType<N>::Type>::VALUE); \
    } while (false)

    if (veryVerbose) {
        cout << "Checking over-aligned types\n";
    }

    CHECK_ALIGNMENT(1);
    CHECK_ALIGNMENT(2);
    CHECK_ALIGNMENT(4);
    CHECK_ALIGNMENT(8);
    CHECK_ALIGNMENT(16);
    CHECK_ALIGNMENT(32);
    CHECK_ALIGNMENT(64);
    CHECK_ALIGNMENT(128);
    CHECK_ALIGNMENT(256);
    CHECK_ALIGNMENT(512);
    CHECK_ALIGNMENT(1024);
    CHECK_ALIGNMENT(2048);
    CHECK_ALIGNMENT(4096);
    CHECK_ALIGNMENT(8192);

#undef  CHECK_ALIGNMENT

#else

    if (veryVerbose) {
        cout << "Not checking over-aligned types - alignas not available\n";
    }

#endif
      } break;
      default: {
        cerr << "WARNING: CASE `"<< test << "' NOT FOUND." <<endl;
        testStatus = -1;
      } break;
    }

    if (testStatus > 0) {
        cerr << "Error, non-zero test status = " << testStatus << "." << endl;
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
