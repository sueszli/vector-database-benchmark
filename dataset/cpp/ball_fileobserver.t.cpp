// ball_fileobserver.t.cpp                                            -*-C++-*-
#include <ball_fileobserver.h>

#include <ball_context.h>
#include <ball_log.h>
#include <ball_loggermanager.h>
#include <ball_loggermanagerconfiguration.h>
#include <ball_recordattributes.h>
#include <ball_severity.h>
#include <ball_streamobserver.h>

#include <bdlb_tokenizer.h>

#include <bdls_filesystemutil.h>
#include <bdls_pathutil.h>
#include <bdls_processutil.h>
#include <bdls_tempdirectoryguard.h>

#include <bdlt_currenttime.h>
#include <bdlt_date.h>
#include <bdlt_datetime.h>
#include <bdlt_datetimeutil.h>
#include <bdlt_epochutil.h>
#include <bdlt_localtimeoffset.h>

#include <bsla_maybeunused.h>

#include <bslim_testutil.h>

#include <bslma_default.h>
#include <bslma_defaultallocatorguard.h>
#include <bslma_testallocator.h>
#include <bslma_usesbslmaallocator.h>

#include <bslmf_assert.h>
#include <bslmf_nestedtraitdeclaration.h>

#include <bslmt_threadutil.h>

#include <bsls_assert.h>
#include <bsls_platform.h>
#include <bsls_timeinterval.h>
#include <bsls_types.h>

#include <bsl_climits.h>
#include <bsl_cstddef.h>
#include <bsl_cstdio.h>      // 'remove'
#include <bsl_cstdlib.h>
#include <bsl_cstring.h>
#include <bsl_ctime.h>
#include <bsl_iomanip.h>
#include <bsl_iostream.h>
#include <bsl_memory.h>
#include <bsl_sstream.h>
#include <bsl_string.h>

#include <bsl_c_stdio.h>
#include <bsl_c_stdlib.h>    // 'unsetenv'

#include <sys/types.h>
#include <sys/stat.h>

#ifdef BSLS_PLATFORM_OS_UNIX
#include <glob.h>
#include <bsl_c_signal.h>
#include <sys/resource.h>
#include <bsl_c_time.h>
#include <unistd.h>
#endif

#ifdef BSLS_PLATFORM_OS_WINDOWS
#include <windows.h>
#endif

// Note: on Windows -> WinGDI.h:#define ERROR 0
#if defined(BSLS_PLATFORM_CMP_MSVC) && defined(ERROR)
#undef ERROR
#endif

using namespace BloombergLP;

using bsl::cout;
using bsl::cerr;
using bsl::endl;
using bsl::flush;

// ============================================================================
//                              TEST PLAN
// ----------------------------------------------------------------------------
//                              Overview
//                              --------
// The component under test defines an observer ('ball::FileObserver') that
// writes log records to a file and stdout.
//-----------------------------------------------------------------------------
// CREATORS
// [ 1] FileObserver(Severity::Level, bslma::Allocator * = 0);
// [ 1] FileObserver(Severity::Level, bool, bslma::Allocator * = 0);
// [ 1] ~FileObserver();
// [ 6] FileObserver();
// [ 6] FileObserver(Allocator *);
//
// MANIPULATORS
// [ 1] void disableFileLogging();
// [ 2] void disableLifetimeRotation();
// [  ] void disablePublishInLocalTime();
// [ 2] void disableSizeRotation();
// [ 1] void disableStdoutLoggingPrefix();
// [  ] void disableTimeIntervalRotation();
// [ 1] void disableUserFieldsLogging();
// [ 1] int  enableFileLogging(const char *fileName);
// [ 1] int  enableFileLogging(const char *fileName, bool timestampFlag);
// [  ] void enablePublishInLocalTime();
// [ 1] void enableStdoutLoggingPrefix();
// [ 1] void enableUserFieldsLogging();
// [ 1] void publish(const Record& record, const Context& context);
// [ 1] void publish(const shared_ptr<Record>&, const Context&);
// [ 2] void forceRotation();
// [ 2] void rotateOnLifetime(DatetimeInterval& interval);
// [ 2] void rotateOnSize(int size);
// [ 2] void rotateOnTimeInterval(const DatetimeInterval& interval);
// [ 2] void rotateOnTimeInterval(const DtInterval& i, const Datetime& s);
// [ 1] void setLogFormat(const char*, const char*);
// [ 4] void setOnFileRotationCallback(const OnFileRotationCallback&);
// [ 1] void setStdoutThreshold(ball::Severity::Level stdoutThreshold);
//
// ACCESSORS
// [ 1] void getLogFormat(const char**, const char**) const;
// [ 1] bool isFileLoggingEnabled() const;
// [ 1] bool isFileLoggingEnabled(string& logFilename) const;
// [ 1] bool isStdoutLoggingPrefixEnabled() const;
// [  ] bool isPublishInLocalTimeEnabled() const;
// [ 1] bool isUserFieldsLoggingEnabled() const;
// [  ] bdlt::DatetimeInterval localTimeOffset() const;
// [ 2] bdlt::DatetimeInterval rotationLifetime() const;
// [ 2] int rotationSize() const;
// [ 1] ball::Severity::Level stdoutThreshold() const;
// ----------------------------------------------------------------------------
// [ 6] CONCERN: 'FileObserver' can be created using 'make_shared'.
// [ 6] CONCERN: 'FileObserver' can be created using 'allocate_shared'.
// [ 5] CONCERN: CURRENT LOCAL-TIME OFFSET IN TIMESTAMP
// [ 4] CONCERN: ROTATION CALLBACK INVOCATION
// [ 7] USAGE EXAMPLE

// Note assert and debug macros all output to cerr instead of cout, unlike
// most other test drivers.  This is necessary because test case 1 plays
// tricks with cout and examines what is written there.

// ============================================================================
//                     STANDARD BDE ASSERT TEST FUNCTION
// ----------------------------------------------------------------------------

namespace {

int testStatus = 0;

void aSsErT(bool condition, const char *message, int line)
{
    if (condition) {
        cerr << "Error " __FILE__ "(" << line << "): " << message
             << "    (failed)" << endl;

        if (0 <= testStatus && testStatus <= 100) {
            ++testStatus;
        }
    }
}

}  // close unnamed namespace

namespace {

void aSsErT2(bool condition, const char *message, int line)
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

#define ASSERT2(X) { aSsErT2(!(X), #X, __LINE__); }

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
//                  NEGATIVE-TEST MACRO ABBREVIATIONS
// ----------------------------------------------------------------------------

#define ASSERT_SAFE_PASS(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_PASS(EXPR)
#define ASSERT_SAFE_FAIL(EXPR) BSLS_ASSERTTEST_ASSERT_SAFE_FAIL(EXPR)
#define ASSERT_PASS(EXPR)      BSLS_ASSERTTEST_ASSERT_PASS(EXPR)
#define ASSERT_FAIL(EXPR)      BSLS_ASSERTTEST_ASSERT_FAIL(EXPR)
#define ASSERT_OPT_PASS(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_PASS(EXPR)
#define ASSERT_OPT_FAIL(EXPR)  BSLS_ASSERTTEST_ASSERT_OPT_FAIL(EXPR)

// ============================================================================
//                  GLOBAL TYPEDEFS/CONSTANTS FOR TESTING
// ----------------------------------------------------------------------------
static bool verbose;
static bool veryVerbose;
static bool veryVeryVerbose;
static bool veryVeryVeryVerbose;

typedef ball::FileObserver   Obj;
typedef bdls::FilesystemUtil FsUtil;
typedef bsls::Types::Int64   Int64;

// ============================================================================
//                                TYPE TRAITS
// ----------------------------------------------------------------------------

BSLMF_ASSERT(bslma::UsesBslmaAllocator<Obj>::value);

// ============================================================================
//                  GLOBAL HELPER FUNCTIONS FOR TESTING
// ----------------------------------------------------------------------------

namespace {

bsl::string::size_type replaceSecondSpace(bsl::string *s, char value)
    // Replace the second space character (' ') in the specified 'string' with
    // the specified 'value'.  Return the index position of the character that
    // was replaced on success, and 'bsl::string::npos' otherwise.
{
    bsl::string::size_type index = s->find(' ');
    if (bsl::string::npos != index) {
        index = s->find(' ', index + 1);
        if (bsl::string::npos != index) {
            (*s)[index] = value;
        }
    }
    return index;
}

bdlt::Datetime getCurrentTimestamp()
    // Return current local time as 'bdlt::Datetime' value.
{
    time_t    currentTime = time(0);
    struct tm localtm;
#ifdef BSLS_PLATFORM_OS_WINDOWS
    localtm = *localtime(&currentTime);
#else
    localtime_r(&currentTime, &localtm);
#endif

    bdlt::Datetime stamp;
    bdlt::DatetimeUtil::convertFromTm(&stamp, localtm);
    return stamp;
}

void removeFilesByPrefix(const char *prefix)
    // Remove the files with the specified 'prefix'.
{
#ifdef BSLS_PLATFORM_OS_WINDOWS
    bsl::string filename = prefix;
    filename += "*";
    WIN32_FIND_DATA findFileData;

    bsl::vector<bsl::string> fileNames;
    HANDLE hFind = FindFirstFile(filename.c_str(), &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        fileNames.push_back(findFileData.cFileName);
        while (FindNextFile(hFind, &findFileData)) {
            fileNames.push_back(findFileData.cFileName);
        }
        FindClose(hFind);
    }

    char tmpPathBuf[MAX_PATH];
    GetTempPath(MAX_PATH, tmpPathBuf);
    bsl::string tmpPath(tmpPathBuf);

    bsl::vector<bsl::string>::iterator itr;
    for (itr = fileNames.begin(); itr != fileNames.end(); ++itr) {
        bsl::string fn = tmpPath + (*itr);
        if (!DeleteFile(fn.c_str()))
        {
            LPVOID lpMsgBuf;
            FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER |
                FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL,
                GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                (LPTSTR) &lpMsgBuf,
                0,
                NULL);
            cerr << "Error, " << (char*)lpMsgBuf << endl;
            LocalFree(lpMsgBuf);
        }
    }
#else
    glob_t      globbuf;
    bsl::string filename(prefix);

    filename += "*";
    glob(filename.c_str(), 0, 0, &globbuf);

    for (size_t i = 0; i < globbuf.gl_pathc; ++i) {
        unlink(globbuf.gl_pathv[i]);
    }

    globfree(&globbuf);
#endif
}

bsl::string readPartialFile(bsl::string&   fileName,
                            FsUtil::Offset startOffset)
    // Read the content of a file with the specified 'fileName' starting at the
    // specified 'startOffset' to the end-of-file and return it as a string.
{
    bsl::string result;
    result.reserve(static_cast<bsl::string::size_type>(
                             FsUtil::getFileSize(fileName) + 1 - startOffset));

    FILE *fp = fopen(fileName.c_str(), "r");
    BSLS_ASSERT_OPT(fp);

    BSLA_MAYBE_UNUSED int rc = fseek(fp,
                                     static_cast<long>(startOffset),
                                     SEEK_SET);
    BSLS_ASSERT_OPT(0 == rc);

    int c;
    while (EOF != (c = getc(fp))) {
        result += static_cast<char>(c);
    }

    fclose(fp);

    return result;
}

class LogRotationCallbackTester {
    // This class can be used as a functor matching the signature of
    // 'ball::FileObserver2::OnFileRotationCallback'.  This class records every
    // invocation of the function-call operator, and is intended to test
    // whether 'ball::FileObserver2' calls the log-rotation callback
    // appropriately.

    // PRIVATE TYPES
    struct Rep {
      private:
        // NOT IMPLEMENTED
        Rep(const Rep&);
        Rep& operator=(const Rep&);

      public:
        // DATA
        int         d_invocations;
        int         d_status;
        bsl::string d_rotatedFileName;

        // TRAITS
        BSLMF_NESTED_TRAIT_DECLARATION(Rep, bslma::UsesBslmaAllocator);

        // CREATORS
        explicit Rep(bslma::Allocator *basicAllocator)
            // Create an object with default attribute values.  Use the
            // specified 'basicAllocator' to supply memory.
        : d_invocations(0)
        , d_status(0)
        , d_rotatedFileName(basicAllocator)
        {
        }
    };

    enum {
        k_UNINITIALIZED = INT_MIN
    };

    // DATA
    bsl::shared_ptr<Rep> d_rep;

  public:
    // CREATORS
    explicit LogRotationCallbackTester(bslma::Allocator *basicAllocator)
        // Create a callback tester object with default attribute values.  Use
        // the specified 'basicAllocator' to supply memory.
    {
        d_rep.createInplace(basicAllocator, basicAllocator);
        reset();
    }

    // MANIPULATORS
    void operator()(int status, const bsl::string& rotatedFileName)
        // Set the value at the status address supplied at construction to the
        // specified 'status', and set the value at the log file name address
        // supplied at construction to the specified 'rotatedFileName'.
    {
        ++d_rep->d_invocations;
        d_rep->d_status          = status;
        d_rep->d_rotatedFileName = rotatedFileName;
    }

    void reset()
        // Reset the attributes of this object to their default values.
    {
        d_rep->d_invocations     = 0;
        d_rep->d_status          = k_UNINITIALIZED;
        d_rep->d_rotatedFileName.clear();
    }

    // ACCESSORS
    int numInvocations() const { return d_rep->d_invocations; }
        // Return the number of times that the function-call operator has been
        // invoked since the most recent call to 'reset', or if 'reset' has
        // not been called, since this objects construction.

    int status() const { return d_rep->d_status; }
        // Return the status passed to the most recent invocation of the
        // function-call operation, or 'k_UNINITIALIZED' if 'numInvocations' is
        // 0.

    const bsl::string& rotatedFileName() const
        // Return a 'const' reference to the file name supplied to the most
        // recent invocation of the function-call operator, or the empty string
        // if 'numInvocations' is 0.
    {
        return d_rep->d_rotatedFileName;
    }

};

typedef LogRotationCallbackTester RotCb;

struct TestCurrentTimeCallback {
  private:
    // DATA
    static bsls::TimeInterval s_utcTime;

  public:
    // CLASS METHODS
    static bsls::TimeInterval load();
        // Return the value corresponding to the most recent call to the
        // 'setTimeToReport' method.  The behavior is undefined unless
        // 'setUtcTime' has been called.

    static void setUtcDatetime(const bdlt::Datetime& utcTime);
        // Set the specified 'utcTime' as the value obtained (after conversion
        // to 'bdlt::IntervalConversionUtil') from calls to the 'load' method.
        // The behavior is undefined unless
        // 'bdlt::EpochUtil::epoch() <= utcTime'.
};

bsls::TimeInterval TestCurrentTimeCallback::s_utcTime;

bsls::TimeInterval TestCurrentTimeCallback::load()
{
    return s_utcTime;
}

void TestCurrentTimeCallback::setUtcDatetime(const bdlt::Datetime &utcTime)
{
    ASSERT(bdlt::EpochUtil::epoch() <= utcTime);

    int rc = bdlt::EpochUtil::convertToTimeInterval(&s_utcTime, utcTime);
    ASSERT(0 == rc);
}

struct TestLocalTimeOffsetCallback {
  private:
    // DATA
    static bsls::Types::Int64 s_localTimeOffsetInSeconds;
    static int                s_loadCount;

  public:
    // CLASS METHODS
    static int loadCount();
        // Return the number of times the 'loadLocalTimeOffset' method has been
        // called since the start of process.

    static bsls::TimeInterval loadLocalTimeOffset(const bdlt::Datetime&);
        // Return the local time offset that was set by the previous call to
        // the 'setLocalTimeOffset' method.  If the 'setLocalTimeOffset' method
        // has not been called, load 0.  Note that the specified 'utcDateime'
        // is ignored.

    static void setLocalTimeOffset(
                                  bsls::Types::Int64 localTimeOffsetInSeconds);
        // Set the specified 'localTimeOffsetInSeconds' as the value loaded by
        // calls to the loadLocalTimeOffset' method.

};

bsls::Types::Int64 TestLocalTimeOffsetCallback::s_localTimeOffsetInSeconds = 0;
int                TestLocalTimeOffsetCallback::s_loadCount                = 0;

int TestLocalTimeOffsetCallback::loadCount()
{
    return s_loadCount;
}

bsls::TimeInterval TestLocalTimeOffsetCallback::loadLocalTimeOffset(
                                                         const bdlt::Datetime&)
{
    ++s_loadCount;
    return bsls::TimeInterval(s_localTimeOffsetInSeconds, 0);
}

void TestLocalTimeOffsetCallback::setLocalTimeOffset(
                                   bsls::Types::Int64 localTimeOffsetInSeconds)
{
    s_localTimeOffsetInSeconds = localTimeOffsetInSeconds;
}

int readFileIntoString(int                lineNum,
                       const bsl::string& fileName,
                       bsl::string&       fileContent)
    // Read the content of the specified 'fileName' file into the specified
    // 'fileContent' string.  Return the number of lines read from the file.
{
    bsl::ifstream fs;

#ifdef BSLS_PLATFORM_OS_UNIX
    glob_t globbuf;
    ASSERTV(lineNum, 0 == glob((fileName + "*").c_str(), 0, 0, &globbuf));
    ASSERTV(lineNum, 1 == globbuf.gl_pathc);

    fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
    globfree(&globbuf);
#else
    fs.open(fileName.c_str(), bsl::ifstream::in);
#endif

    ASSERTV(lineNum, fs.is_open());
    fileContent = "";

    bsl::string lineContent;
    int         lines = 0;
    while (getline(fs, lineContent))
    {
        fileContent += lineContent;
        fileContent += '\n';
        lines++;
    }
    fs.close();
    return lines;
}

void splitStringIntoLines(bsl::vector<bsl::string> *result, const char *ascii)
{
    ASSERT(result)
    ASSERT(ascii)

    for (bdlb::Tokenizer itr(bslstl::StringRef(ascii),
                             bslstl::StringRef(""),
                             bslstl::StringRef("\n")); itr.isValid(); ++itr) {
        if (itr.token().length() > 0) {
            result->push_back(itr.token());
        }
    }
}

void getDatetimeField(bsl::string        *result,
                      const bsl::string&  filename,
                      int                 recordNumber)
{
    ASSERT(1 <= recordNumber);

    bsl::string fileContent;

    int lineCount = readFileIntoString(__LINE__, filename, fileContent);
    ASSERT(recordNumber * 2 <= lineCount);

    bsl::vector<bsl::string> lines;
    splitStringIntoLines(&lines, fileContent.c_str());
    int recordIndex = recordNumber  - 1;

    ASSERT(0 <= recordIndex);
    ASSERT(static_cast<int>(lines.size()) >  recordIndex);

    const bsl::string& s = lines[recordIndex];
    *result = s.substr(0, s.find_first_of(' '));
}

}  // close unnamed namespace

// ============================================================================
//                               USAGE EXAMPLE 1
// ----------------------------------------------------------------------------
namespace BALL_FILEOBSERVER_USAGE_EXAMPLE
{
}  // close namespace BALL_FILEOBSERVER_USAGE_EXAMPLE

//=============================================================================
//                                 MAIN PROGRAM
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    int test            = argc > 1 ? bsl::atoi(argv[1]) : 0;

    verbose             = argc > 2;
    veryVerbose         = argc > 3;
    veryVeryVerbose     = argc > 4;
    veryVeryVeryVerbose = argc > 5;

    cout << "TEST " << __FILE__ << " CASE " << test << endl << flush;

    bslma::TestAllocator         defaultAllocator("default",
                                                  veryVeryVeryVerbose);
    bslma::DefaultAllocatorGuard guard(&defaultAllocator);

    switch (test) { case 0:
      case 7: {
        // --------------------------------------------------------------------
        // USAGE EXAMPLE
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

        if (verbose) cout << "\nUSAGE EXAMPLE"
                             "\n=============" << endl;

        // This is standard preamble to create the directory and filename for
        // the test.
        bdls::TempDirectoryGuard tempDirGuard("ball_");
        bsl::string              fileName(tempDirGuard.getTempDirName());
        bdls::PathUtil::appendRaw(&fileName, "test.log");

///Usage
///-----
// This section illustrates intended use of this component.
//
///Example: Basic Usage
/// - - - - - - - - - -
// First, we create a 'ball::LoggerManagerConfiguration' object, 'lmConfig',
// and set the logging "pass-through" level -- the level at which log records
// are published to registered observers -- to 'DEBUG':
//..
        ball::LoggerManagerConfiguration lmConfig;
        lmConfig.setDefaultThresholdLevelsIfValid(ball::Severity::e_DEBUG);
//..
// Next, create a 'ball::LoggerManagerScopedGuard' object whose constructor
// takes the configuration object just created.  The guard will initialize the
// logger manager singleton on creation and destroy the singleton upon
// destruction.  This guarantees that any resources used by the logger manager
// will be properly released when they are not needed:
//..
        ball::LoggerManagerScopedGuard guard(lmConfig);
        ball::LoggerManager& manager = ball::LoggerManager::singleton();
//..
// Next, we create a 'ball::FileObserver' object and register it with the
// 'ball' logging system:
//..
        bsl::shared_ptr<ball::FileObserver> observer =
                                        bsl::make_shared<ball::FileObserver>();
        int rc = manager.registerObserver(observer, "default");
        ASSERT(0 == rc);
//..
// The default format for outputting log records can be changed by calling the
// 'setLogFormat' method.  The statement below outputs record timestamps in ISO
// 8601 format to the log file and in 'bdlt'-style (default) format to
// 'stdout', where timestamps are output with millisecond precision in both
// cases:
//..
        observer->setLogFormat("%I %p:%t %s %f:%l %c %m\n",
                               "%d %p:%t %s %f:%l %c %m\n");
//..
// Note that both of the above format specifications omit user fields ('%u') in
// the output.  Also note that, unlike the default, this format does not emit a
// blank line between consecutive log messages.
//
// Henceforth, all messages that are published by the logging system will be
// transmitted to the 'publish' method of 'observer'.  By default, only the
// messages with a 'e_WARN', 'e_ERROR', or 'e_FATAL' severity will be logged to
// 'stdout':
//..
        BALL_LOG_SET_CATEGORY("main")

        BALL_LOG_INFO << "Will not be published on 'stdout'.";
        BALL_LOG_WARN << "This warning *will* be published on 'stdout'.";
//..
// This default can be changed by specifying an optional argument to the
// 'ball::FileObserver' constructor or by calling the 'setStdoutThreshold'
// method:
//..
        observer->setStdoutThreshold(ball::Severity::e_INFO);

        BALL_LOG_DEBUG << "This debug message is not published on 'stdout'.";
        BALL_LOG_INFO  << "This info message *will* be published on 'stdout'.";
        BALL_LOG_WARN  << "This warning will be published on 'stdout'.";
//..
// Finally, we configure additional loggin to a specified file and specify
// rotation rules based on the size of the log file or its lifetime:
//..
        // Create and log records to a file named "test.log" in a temp folder
        observer->enableFileLogging(fileName.c_str());

        // Disable 'stdout' logging.
        observer->setStdoutThreshold(ball::Severity::e_OFF);

        // Rotate the file when its size becomes greater than or equal to 256
        // megabytes.
        observer->rotateOnSize(1024 * 256);

        // Rotate the file every 24 hours.
        observer->rotateOnTimeInterval(bdlt::DatetimeInterval(1));
//..
// Note that in this configuration the user may end up with multiple log files
// for any given day (because of the rotation-on-size rule).  This feature can
// be disabled dynamically later:
//..
        observer->disableSizeRotation();
//..
      } break;
      case 6: {
        // --------------------------------------------------------------------
        // CONSTRUCTOR, MAKE_SHARED, AND ALLOCATE_SHARED TEST
        //
        // Concerns:
        //: 1 That the severity level, if supplied, determines the severity
        //:   level of the created 'FileObserver'.
        //:
        //: 2 That the boolean 'publishInLocalTime' argument, if passed,
        //:   determines that of the created 'FileObserver'.
        //:
        //: 3 That the allocator passed, if passed, determines that used by the
        //:   non-temporary memory of the created 'FileObserver'.
        //:
        //: 4 That the severity level, if not passed, defaults to
        //:   'ball::Severity::e_WARN'.
        //:
        //: 5 That the 'publishInLocalTime', if not passed, defaults to
        //:   'false'.
        //:
        //: 6 That if the allocator is not passed, allocation is done by the
        //:   the default allocator.
        //:
        //: 7 That the type traits are correct to facilitate creation by
        //:   'bsl::make_shared' and 'bsl::allocate_shared'.
        //
        // Plan:
        //: 1 Iterate through a table providing a variety of values to be
        //:   passed to the 'severity' and 'publishInLocalTime' arguments of
        //:   the c'tors.
        //:
        //: 2 With a switch in a 'for' loop, iterate through creating an 'Obj'
        //:   with all six possible c'tors, then verify through accessors that
        //:   the properties are as expected.
        //:
        //: 3 Call 'make_shared' using 3 different c'tors and verify the
        //:   properties of the created objects.
        //:
        //: 4 Call 'allocate_shared' using 3 different c'tors and verify the
        //:   properties of the created objects.
        //
        // Testing:
        //   FileObserver();
        //   FileObserver(Allocator *);
        //   CONCERN: 'FileObserver' can be created using 'make_shared'.
        //   CONCERN: 'FileObserver' can be created using 'allocate_shared'.
        // --------------------------------------------------------------------

        if (verbose) cout <<
                        "CONSTRUCTOR, MAKE_SHARED, AND ALLOCATE_SHARED TEST\n"
                        "==================================================\n";

        typedef ball::Severity Sev;
        typedef Sev::Level     Level;

        const Sev::Level defaultLevel = Sev::e_WARN;
        const bool       defaultPilt  = false;    // 'PILT' ==
                                                  // Publish In Local Time

        bslma::TestAllocator& da = defaultAllocator;
        bslma::TestAllocator  oa(veryVeryVeryVerbose);

        if (verbose) cout << "Table-driven test of c'tors\n";
        {
            static const struct Data {
                int      d_line;
                Level    d_level;
                bool     d_publishInLocalTime;
            } DATA[] = {
              { L_, Sev::e_ERROR,  false },
              { L_, Sev::e_ERROR,  true  },
              { L_, Sev::e_WARN,   false },
              { L_, Sev::e_WARN,   true  },
              { L_, Sev::e_DEBUG,  false },
              { L_, Sev::e_DEBUG,  true  }
             };
            enum { k_NUM_DEBUG = sizeof DATA / sizeof *DATA };

            for (int ti = 0; ti < k_NUM_DEBUG; ++ti) {
                const Data& ARGS = DATA[ti];
                const int   LINE = ARGS.d_line;

                bool go = true;
                for (int ci = 0; go; ++ci) {
                    Level level = ARGS.d_level;
                    bool  pilt  = ARGS.d_publishInLocalTime;

                    bslma::Allocator *passedAllocator = &oa;

                    Obj *mX_p = 0;

                    // In all of these case, try to call the c'tor with
                    // '(level, pilt, passedAllocator)', but in the case of
                    // those c'tors that don't take all 3 args, assign 'level',
                    // 'pilt', and/or 'passedAllocator' to their default values
                    // if they weren't passed.

                    switch (ci) {
                      case 0: {
                        mX_p = new (da) Obj();

                        level           = defaultLevel;
                        pilt            = defaultPilt;
                        passedAllocator = &defaultAllocator;
                      } break;
                      case 1: {
                        mX_p = new (da) Obj(passedAllocator);

                        level = defaultLevel;
                        pilt  = defaultPilt;
                      } break;
                      case 2: {
                        mX_p = new (da) Obj(level);

                        pilt            = defaultPilt;
                        passedAllocator = &defaultAllocator;
                      } break;
                      case 3: {
                        mX_p = new (da) Obj(level, passedAllocator);

                        pilt = defaultPilt;
                      } break;
                      case 4: {
                        mX_p = new (da) Obj(level, pilt);

                        passedAllocator = &defaultAllocator;
                      } break;
                      case 5: {
                        mX_p = new (da) Obj(level, pilt, passedAllocator);
                      } break;
                      case 6: {
                        go = false;
                        continue;
                      } break;
                      default: {
                        ASSERTV(ci, 0 && "invalid case");
                      } break;
                    }

                    const Obj& X = *mX_p;

                    ASSERTV(ci, LINE, level == X.stdoutThreshold());
                    ASSERTV(ci, LINE, pilt == X.isPublishInLocalTimeEnabled());
                    ASSERTV(ci, LINE, passedAllocator == X.allocator());

                    da.deleteObject(mX_p);
                }
            }
        }

        // Set non-default values to pass to c'tors to test them.

        const Sev::Level otherLevel = Sev::e_FATAL;
        const bool       otherPilt  = true;

        ASSERT(defaultLevel != otherLevel);
        ASSERT(defaultPilt  != otherPilt);
        ASSERT(&da          == &defaultAllocator);
        ASSERT(&da          == bslma::Default::allocator());
        ASSERT(&oa          != &da);

        if (verbose) cout << "Test 'make_shared' with various c'tors.\n";
        {
            {
                bsl::shared_ptr<Obj> mX = bsl::make_shared<Obj>();
                const Obj& X = *mX;

                ASSERT(defaultLevel == X.stdoutThreshold());
                ASSERT(defaultPilt  == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          == X.allocator());
            }

            {
                bsl::shared_ptr<Obj> mX = bsl::make_shared<Obj>(otherLevel);
                const Obj& X = *mX;

                ASSERT(defaultLevel != X.stdoutThreshold());
                ASSERT(otherLevel   == X.stdoutThreshold());
                ASSERT(defaultPilt  == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          == X.allocator());
            }

            {
                bsl::shared_ptr<Obj> mX = bsl::make_shared<Obj>(otherLevel,
                                                                otherPilt);
                const Obj& X = *mX;

                ASSERT(defaultLevel != X.stdoutThreshold());
                ASSERT(otherLevel   == X.stdoutThreshold());
                ASSERT(defaultPilt  != X.isPublishInLocalTimeEnabled());
                ASSERT(otherPilt    == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          == X.allocator());
            }
        }

        if (verbose) cout << "Test 'allocate_shared' with various c'tors.\n";
        {
            {
                bsl::shared_ptr<Obj> mX = bsl::allocate_shared<Obj>(&oa);
                const Obj& X = *mX;

                ASSERT(defaultLevel == X.stdoutThreshold());
                ASSERT(defaultPilt  == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          != X.allocator());
                ASSERT(&oa          == X.allocator());
                ASSERT(0            == da.numBlocksInUse());
            }

            {
                bsl::shared_ptr<Obj> mX = bsl::allocate_shared<Obj>(
                                                                   &oa,
                                                                   otherLevel);
                const Obj& X = *mX;

                ASSERT(defaultLevel != X.stdoutThreshold());
                ASSERT(otherLevel   == X.stdoutThreshold());
                ASSERT(defaultPilt  == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          != X.allocator());
                ASSERT(&oa          == X.allocator());
                ASSERT(0            == da.numBlocksInUse());
            }

            {
                bsl::shared_ptr<Obj> mX = bsl::allocate_shared<Obj>(&oa,
                                                                    otherLevel,
                                                                    otherPilt);
                const Obj& X = *mX;

                ASSERT(defaultLevel != X.stdoutThreshold());
                ASSERT(otherLevel   == X.stdoutThreshold());
                ASSERT(defaultPilt  != X.isPublishInLocalTimeEnabled());
                ASSERT(otherPilt    == X.isPublishInLocalTimeEnabled());
                ASSERT(&da          != X.allocator());
                ASSERT(&oa          == X.allocator());
                ASSERT(0            == da.numBlocksInUse());
            }
        }
      } break;
      case 5: {
        // --------------------------------------------------------------------
        // TESTING CURRENT LOCAL-TIME OFFSET IN TIMESTAMP
        //   Per DRQS 13681097, log records observe DST time transitions when
        //   the default logging functor is used and the 'publishInLocalTime'
        //   attribute is 'true'.
        //
        // Concern:
        //: 1 Log records show the current local time offset (possibly
        //:   different from the local time offset in effect on construction),
        //:   when 'true == isPublishInLocalTimeEnabled()'.
        //:
        //: 2 Log records show UTC when
        //:   'false == isPublishInLocalTimeEnabled()'.
        //:
        //: 3 QoI: The local-time offset is obtained not more than once per log
        //:   record.
        //:
        //: 4 The helper class 'TestSystemTimeCallback' has a method, 'load',
        //:   that loads the user-specified UTC time, and that method can be
        //:   installed as the system-time callback of 'bdlt::CurrentTime'.
        //:
        //: 5 The helper class 'TestLocalTimeOffsetCallback' has a method,
        //:   'loadLocalTimeOffset', that loads the user-specified local-time
        //:   offset value, and that method can be installed as the local-time
        //:   callback of 'bdlt::CurrentTime', and that the value loaded is not
        //:   influenced by the user-specified 'utcDatetime'.
        //:
        //: 6 The helper class method 'TestLocalTimeOffsetCallback::loadCount'
        //:   provides an accurate count of the calls to the
        //:   'TestLocalTimeOffsetCallback::loadLocalTimeOffset' method.
        //
        // Plan:
        //: 1 Test the helper 'TestSystemTimeCallback' class (C-4):
        //:
        //:   1 Using the array-driven technique, confirm that the 'load'
        //:     method obtains the value last set by the 'setUtcDatetime'
        //:     method.  Use UTC values that do not coincide with the actual
        //:     UTC datetime.
        //:
        //:   2 Install the 'TestSystemTimeCallback::load' method as the
        //:     system-time callback of system-time offset callback of
        //:     'bdlt::CurrentTime', and run through the same values as used in
        //:     P-1.1.  Confirm that values returned from 'bdlt::CurrentTime'
        //:     match the user-specified values.
        //:
        //: 2 Test the helper 'TestLocalTimeOffsetCallback' class (C-5):
        //:
        //:   1 Using the array-driven technique, confirm that the
        //:     'loadLocalTimeOffset' method obtains the value last set by the
        //:     'setLocalTimeOffset' method.  At least one value should differ
        //:     from the current, actual local-time offset.
        //:
        //:   2 Install the 'TestLocalTimeOffsetCallback::loadLocalTimeOffset'
        //:     method as the local-time offset callback of
        //:     'bdlt::CurrentTime', and run through the same user-specified
        //:     local time offsets as used in P-2.1.  Confirm that values
        //:     returned from 'bdlt::CurrentTime' match the user-specified
        //:     values.  Repeat the request for (widely) different UTC
        //:     datetime values to confirm that the local time offset value
        //:     remains that defined by the callback.
        //:
        //:   3 Confirm that the value returned by the 'loadCount' method
        //:     increases by exactly 1 each time a local-time offset is
        //:     obtained via 'bdlt::CurrentTime'.  (C-6)
        //:
        //: 3 Using an ad-hoc approach, confirm that the datetime field of a
        //:   published log record is the expected (arbitrary) UTC datetime
        //:   value when publishing in local-time is disabled.  Enable
        //:   publishing in local-time and confirm that the published datetime
        //:   field matches that of the (arbitrary) user-defined local-time
        //:   offsets.  Disable publishing in local time, and confirm that log
        //:   records are again published with the UTC datetime.  (C-1, C-2)
        //:
        //: 4 When publishing in local time is enabled, confirm that there
        //:   exactly 1 request for local time offset for each published
        //:   record.  (C-3)
        //
        // Testing:
        //   CONCERN: CURRENT LOCAL-TIME OFFSET IN TIMESTAMP
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTESTING CURRENT LOCAL-TIME OFFSET IN TIMESTAMP"
                          << "\n=============================================="
                          << endl;

        const bdlt::Datetime UTC_ARRAY[] = { bdlt::EpochUtil::epoch(),
                                             bdlt::Datetime(2001,
                                                            9,
                                                            11,
                                                            8 + 4, // UTC
                                                            46,
                                                            30,
                                                            0),
                                             bdlt::Datetime(9999,
                                                            12,
                                                            31,
                                                            23,
                                                            59,
                                                            59,
                                                            999)
                                      };
        enum { NUM_UTC_ARRAY = sizeof UTC_ARRAY / sizeof *UTC_ARRAY };

        if (verbose) cout << "\nTest TestSystemTimeCallback: Direct" << endl;
        {
            for (int i = 0; i < NUM_UTC_ARRAY; ++i) {
                bdlt::Datetime utcDatetime = UTC_ARRAY[i];

                if (veryVerbose) { T_ P_(i) P(utcDatetime) }

                TestCurrentTimeCallback::setUtcDatetime(utcDatetime);
                bsls::TimeInterval result = TestCurrentTimeCallback::load();

                bdlt::Datetime resultAsDatetime =
                              bdlt::EpochUtil::convertFromTimeInterval(result);
                ASSERTV(i, utcDatetime == resultAsDatetime);
            }
        }

        if (verbose) cout << "\nTest TestSystemTimeCallback: Installed"
                          << endl;
        {
            // Install callback from 'TestSystemTimeCallback'.

            bdlt::CurrentTime::CurrentTimeCallback
                originalCurrentTimeCallback =
                                     bdlt::CurrentTime::setCurrentTimeCallback(
                                               &TestCurrentTimeCallback::load);

            for (int i = 0; i < NUM_UTC_ARRAY; ++i) {
                bdlt::Datetime utcDatetime = UTC_ARRAY[i];

                if (veryVerbose) { T_ P_(i) P(utcDatetime) }

                TestCurrentTimeCallback::setUtcDatetime(utcDatetime);

                bdlt::Datetime result1 = bdlt::CurrentTime::utc();
                bslmt::ThreadUtil::microSleep(0, 2); // two seconds
                bdlt::Datetime result2 = bdlt::CurrentTime::utc();

                ASSERTV(i, utcDatetime == result1);
                ASSERTV(i, result2     == result1);
            }

           // Restore original system-time callback.

            bdlt::CurrentTime::setCurrentTimeCallback(
                                                  originalCurrentTimeCallback);
        }

        const int LTO_ARRAY[] = { -86399, -1, 0, 1, 86399 };

        enum { NUM_LTO_ARRAY = sizeof LTO_ARRAY / sizeof *LTO_ARRAY };

        int loadCount = TestLocalTimeOffsetCallback::loadCount();
        ASSERT(0 ==  loadCount);

        if (verbose) cout << "\nTest TestLocalTimeOffsetCallback: Direct"
                          << endl;
        {
            for (int i = 0; i < NUM_LTO_ARRAY; ++i) {
                int localTimeOffset = LTO_ARRAY[i];

                if (veryVerbose) { T_ P_(i) P(localTimeOffset) }

                TestLocalTimeOffsetCallback::setLocalTimeOffset(
                                                              localTimeOffset);
                for (int j = 0; j < NUM_UTC_ARRAY; ++j) {
                    bdlt::Datetime utcDatetime  = UTC_ARRAY[j];

                    if (veryVerbose) { T_ T_ P_(j) P(utcDatetime) }

                    Int64 result =
                        TestLocalTimeOffsetCallback::loadLocalTimeOffset(
                                                                   utcDatetime)
                                                               .totalSeconds();
                    ++loadCount;

                    ASSERTV(i, j, localTimeOffset == result);
                    ASSERTV(i, j, loadCount       ==
                                     TestLocalTimeOffsetCallback::loadCount());
                }
            }

        }

        if (verbose) cout << "\nTest TestLocalTimeOffsetCallback: Installed"
                          << endl;
        {
            bdlt::LocalTimeOffset::LocalTimeOffsetCallback
                originalLocalTimeOffsetCallback =
                        bdlt::LocalTimeOffset::setLocalTimeOffsetCallback(
                            &TestLocalTimeOffsetCallback::loadLocalTimeOffset);

            for (int i = 0; i < NUM_LTO_ARRAY; ++i) {
                int localTimeOffset = LTO_ARRAY[i];

                if (veryVerbose) { T_ P_(i) P(localTimeOffset) }

                TestLocalTimeOffsetCallback::setLocalTimeOffset(
                                                              localTimeOffset);
                for (int j = 0; j < NUM_UTC_ARRAY; ++j) {
                    bdlt::Datetime utcDatetime  = UTC_ARRAY[j];

                    if (veryVerbose) { T_ T_ P_(j) P(utcDatetime) }

                    Int64 result =
                        bdlt::LocalTimeOffset::localTimeOffset(utcDatetime)
                                                               .totalSeconds();
                    ++loadCount;

                    ASSERTV(i, j, localTimeOffset == result);
                    ASSERTV(i, j, loadCount       ==
                                     TestLocalTimeOffsetCallback::loadCount());
                }
            }

            bdlt::LocalTimeOffset::setLocalTimeOffsetCallback(
                                              originalLocalTimeOffsetCallback);
        }

        if (verbose) cout << "\nTest Logger" << endl;

        if (veryVerbose) cout << "\tConfigure Logger and Callbacks" << endl;

        // This configuration guarantees that the logger manager will publish
        // all messages regardless of their severity and the observer will see
        // each message only once.

        ball::LoggerManagerConfiguration configuration;
        ASSERT(0 == configuration.setDefaultThresholdLevelsIfValid(
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_TRACE,
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_OFF));

        ball::LoggerManagerScopedGuard guard(configuration);

        ball::LoggerManager& manager = ball::LoggerManager::singleton();

        bslma::TestAllocator ta(veryVeryVeryVerbose);

        bdls::TempDirectoryGuard tempDirGuard("ball_");
        bsl::string              fileName(tempDirGuard.getTempDirName());
        bdls::PathUtil::appendRaw(&fileName, "testLog");

        bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                   &ta),
                                      &ta);
        bsl::shared_ptr<const Obj> X = mX;

        ASSERT(0 == manager.registerObserver(mX, "testObserver"));

        P(fileName);

        ASSERT(0 == mX->enableFileLogging(fileName.c_str(), true));

        bsl::string logfilename;

        ASSERT(X->isFileLoggingEnabled(&logfilename));
        P(logfilename);

        BALL_LOG_SET_CATEGORY("TestCategory");

        int                  logRecordCount  = 0;
        int                  testLocalTimeOffsetInSeconds;
        bsl::string          datetimeField;
        bsl::ostringstream   expectedDatetimeField;
        const bdlt::Datetime testUtcDatetime = UTC_ARRAY[1];

        bdlt::CurrentTime::CurrentTimeCallback originalCurrentTimeCallback =
                                     bdlt::CurrentTime::setCurrentTimeCallback(
                                         &TestCurrentTimeCallback::load);
        TestCurrentTimeCallback::setUtcDatetime(testUtcDatetime);

        bdlt::LocalTimeOffset::LocalTimeOffsetCallback
            originalLocalTimeOffsetCallback =
                        bdlt::LocalTimeOffset::setLocalTimeOffsetCallback(
                            &TestLocalTimeOffsetCallback::loadLocalTimeOffset);

        int expectedLoadCount = TestLocalTimeOffsetCallback::loadCount();

        if (veryVerbose) cout << "\tLog with Publish In Local Time Disabled"
                              << endl;

        ASSERT(!X->isPublishInLocalTimeEnabled());

        BALL_LOG_TRACE << "log 1";
        ++logRecordCount;

        getDatetimeField(&datetimeField, logfilename, logRecordCount);

        const int k_SIZE = 32;
        char      buffer[k_SIZE];
        bsl::memset(buffer, 'X', k_SIZE);

        testUtcDatetime.printToBuffer(buffer, k_SIZE, 3);

        bsl::string EXP(buffer);

        if (veryVerbose) { T_ P_(EXP) P(datetimeField) }

        ASSERT(EXP == datetimeField);
        ASSERTV(expectedLoadCount == TestLocalTimeOffsetCallback::loadCount());

        BALL_LOG_TRACE << "log 2";
        ++logRecordCount;

        getDatetimeField(&datetimeField, logfilename, logRecordCount);

        bsl::memset(buffer, 'X', k_SIZE);

        testUtcDatetime.printToBuffer(buffer, k_SIZE, 3);

        EXP.assign(buffer);

        if (veryVerbose) { T_ P_(EXP) P(datetimeField) }

        ASSERT(EXP == datetimeField);
        ASSERTV(expectedLoadCount == TestLocalTimeOffsetCallback::loadCount());

        if (veryVerbose) cout << "\tLog with Publish In Local Time Enabled"
                              << endl;

        mX->enablePublishInLocalTime();
        ASSERT(X->isPublishInLocalTimeEnabled());

        testLocalTimeOffsetInSeconds = -1 * 60 * 60;
        TestLocalTimeOffsetCallback::setLocalTimeOffset(
                                                 testLocalTimeOffsetInSeconds);

        if (veryVerbose) { T_ P(testLocalTimeOffsetInSeconds); }

        BALL_LOG_TRACE << "log 3";
        ++logRecordCount;
        ++expectedLoadCount;

        getDatetimeField(&datetimeField, logfilename, logRecordCount);

        bsl::memset(buffer, 'X', k_SIZE);

        bdlt::Datetime DT = testUtcDatetime +
                           bdlt::DatetimeInterval(0,
                                                  0,
                                                  0,
                                                  testLocalTimeOffsetInSeconds,
                                                  0);
        DT.printToBuffer(buffer, k_SIZE, 3);

        EXP.assign(buffer);

        if (veryVerbose) { T_ P_(EXP) P(datetimeField) }

        ASSERT(EXP == datetimeField);
        ASSERT(expectedLoadCount == TestLocalTimeOffsetCallback::loadCount());

        testLocalTimeOffsetInSeconds = -2 * 60 * 60;
        TestLocalTimeOffsetCallback::setLocalTimeOffset(
                                                 testLocalTimeOffsetInSeconds);

        if (veryVerbose) { T_ P(testLocalTimeOffsetInSeconds); }

        BALL_LOG_TRACE << "log 4";
        ++logRecordCount;
        ++expectedLoadCount;

        getDatetimeField(&datetimeField, logfilename, logRecordCount);

        bsl::memset(buffer, 'X', k_SIZE);

        DT = testUtcDatetime +
                           bdlt::DatetimeInterval(0,
                                                  0,
                                                  0,
                                                  testLocalTimeOffsetInSeconds,
                                                  0);
        DT.printToBuffer(buffer, k_SIZE, 3);

        EXP.assign(buffer);

        if (veryVerbose) { T_ P_(EXP) P(datetimeField) }

        ASSERT(EXP == datetimeField);
        ASSERT(expectedLoadCount == TestLocalTimeOffsetCallback::loadCount());

        mX->disablePublishInLocalTime();
        ASSERT(!X->isPublishInLocalTimeEnabled());

        BALL_LOG_TRACE << "log 5"; ++logRecordCount;
                                                // ++expectedLoadCount;
        getDatetimeField(&datetimeField, logfilename, logRecordCount);
        bsl::memset(buffer, 'X', k_SIZE);

        testUtcDatetime.printToBuffer(buffer, k_SIZE, 3);

        EXP.assign(buffer);

        if (veryVerbose) { T_ P_(EXP) P(datetimeField) }

        ASSERT(EXP == datetimeField);
        ASSERT(expectedLoadCount == TestLocalTimeOffsetCallback::loadCount());

        if (veryVerbose)
            cout << "\tLog with Publish In Local Time Disabled Again" << endl;

        if (veryVerbose) cout << "\tCleanup" << endl;

        bdlt::CurrentTime::setCurrentTimeCallback(originalCurrentTimeCallback);
        bdlt::LocalTimeOffset::setLocalTimeOffsetCallback(
                                              originalLocalTimeOffsetCallback);

        mX->disableFileLogging();

        // Deregister here as we used local allocator for the observer.
        ASSERT(0 == manager.deregisterObserver("testObserver"));
      } break;
      case 4: {
        // --------------------------------------------------------------------
        // TESTING ROTATION CALLBACK INVOCATION
        //
        // Concern:
        //:  1 Rotation callback is invoked on file rotation.
        //
        // Plan:
        //:  1 Setup test infrastructure.
        //:
        //:  2 Install test rotation callback.
        //:
        //:  3 Trigger file rotation on the observer and verify that rotation
        //:    callback is invoked.
        //
        // Testing:
        //   void setOnFileRotationCallback(const OnFileRotationCallback&);
        //   CONCERN: ROTATION CALLBACK INVOCATION
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTESTING ROTATION CALLBACK INVOCATION"
                          << "\n===================================="  << endl;

        // This configuration guarantees that the logger manager will publish
        // all messages regardless of their severity and the observer will see
        // each message only once.

        ball::LoggerManagerConfiguration configuration;
        ASSERT(0 == configuration.setDefaultThresholdLevelsIfValid(
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_TRACE,
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_OFF));

        ball::LoggerManagerScopedGuard guard(configuration);

        ball::LoggerManager& manager = ball::LoggerManager::singleton();

        BALL_LOG_SET_CATEGORY("TestCategory");

        bslma::TestAllocator ta("test", veryVeryVeryVerbose);

        bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                   &ta),
                                      &ta);
        bsl::shared_ptr<const Obj> X = mX;

        ASSERT(0 == manager.registerObserver(mX, "testObserver"));

        // Set callback to monitor rotation.
        RotCb cb(&ta);
        mX->setOnFileRotationCallback(cb);

        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            ASSERT(0    == mX->enableFileLogging(fileName.c_str()));
            ASSERT(true == X->isFileLoggingEnabled());
            ASSERT(0    == cb.numInvocations());

            mX->disableFileLogging();
            cb.reset();
        }

        if (veryVerbose) cout << "\tTesting rotation on time interval" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            mX->rotateOnLifetime(bdlt::DatetimeInterval(0, 0, 0, 1));
            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));

            BALL_LOG_TRACE << "log";
            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            bslmt::ThreadUtil::microSleep(0, 2);
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 1 == cb.numInvocations());
            ASSERT(1 == FsUtil::exists(cb.rotatedFileName().c_str()));

            mX->disableFileLogging();
            cb.reset();
        }


        if (veryVerbose) cout << "\tTesting rotation on time interval" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            mX->rotateOnTimeInterval(bdlt::DatetimeInterval(0, 0, 0, 1));
            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));

            BALL_LOG_TRACE << "log";
            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            bslmt::ThreadUtil::microSleep(0, 2);
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 1 == cb.numInvocations());
            ASSERT(1 == FsUtil::exists(cb.rotatedFileName().c_str()));

            mX->disableFileLogging();
            cb.reset();
        }

        if (veryVerbose) cout << "\tTesting rotation with start time" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            for (int i = 0; i < 2; ++i) {
                bdlt::Datetime startTime;
                if (i == 0) {
                    mX->enablePublishInLocalTime();
                    startTime = bdlt::CurrentTime::local();
                }
                else {
                    mX->disablePublishInLocalTime();
                    startTime = bdlt::CurrentTime::utc();
                }

                startTime += bdlt::DatetimeInterval(-1, 0, 0, 3);
                mX->rotateOnTimeInterval(bdlt::DatetimeInterval(1), startTime);
                ASSERT(0 == mX->enableFileLogging(fileName.c_str()));

                BALL_LOG_TRACE << "log";
                ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

                bslmt::ThreadUtil::microSleep(0, 2);
                BALL_LOG_TRACE << "log";

                ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

                bslmt::ThreadUtil::microSleep(0, 2);
                BALL_LOG_TRACE << "log";

                ASSERTV(cb.numInvocations(), 1 == cb.numInvocations());
                ASSERT(1 == FsUtil::exists(cb.rotatedFileName().c_str()));

                mX->disableFileLogging();
                cb.reset();
            }
        }

        if (veryVerbose) cout << "\tTesting forced rotation" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));

            BALL_LOG_TRACE << "log";
            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            mX->forceRotation();
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 1 == cb.numInvocations());
            ASSERT(1 == FsUtil::exists(cb.rotatedFileName().c_str()));

            mX->disableFileLogging();
            cb.reset();
        }

        if (veryVerbose) cout << "\tTesting suppressing rotated file name "
                                 "uniqueness" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));

            BALL_LOG_TRACE << "log";
            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            mX->suppressUniqueFileNameOnRotation(true);

            mX->forceRotation();
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 1 == cb.numInvocations());
            ASSERTV(fileName,   cb.rotatedFileName(),
                    fileName == cb.rotatedFileName());

            mX->disableFileLogging();

            mX->suppressUniqueFileNameOnRotation(false);

            cb.reset();
        }

        if (veryVerbose) cout << "\tTesting 'disableLifetimeRotation'" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            mX->rotateOnTimeInterval(bdlt::DatetimeInterval(0, 0, 0, 1));
            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));
            mX->disableLifetimeRotation();

            bslmt::ThreadUtil::microSleep(0, 6);
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            mX->disableFileLogging();
            cb.reset();
        }

        if (veryVerbose)
            cout << "\tTesting 'disableTimeIntervalRotation'" << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            mX->rotateOnTimeInterval(bdlt::DatetimeInterval(0, 0, 0, 1));
            ASSERT(0 == mX->enableFileLogging(fileName.c_str()));
            mX->disableTimeIntervalRotation();

            bslmt::ThreadUtil::microSleep(0, 6);
            BALL_LOG_TRACE << "log";

            ASSERTV(cb.numInvocations(), 0 == cb.numInvocations());

            mX->disableFileLogging();
            cb.reset();
        }

        // Deregister here as we used local allocator for the observer.
        ASSERT(0 == manager.deregisterObserver("testObserver"));
      } break;
      case 3: {
        // --------------------------------------------------------------------
        // TESTING LOGGING TO A FAILING STREAM
        //
        // Concerns:
        //:  1 Observer remains operational when the underlying file stream
        //:    fails.
        //
        // Plan:
        //:  1 Perform logging operations to a filestream that fails.  Verify
        //:    that the warning message is issued.
        //
        // Testing:
        //   CONCERN: Logging to a failing stream.
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTESTING LOGGING TO A FAILING STREAM"
                          << "\n===================================" << endl;

#if defined(BSLS_PLATFORM_OS_UNIX) && !defined(BSLS_PLATFORM_OS_CYGWIN)
        // 'setrlimit' is not implemented on Cygwin.

        // This configuration guarantees that the logger manager will publish
        // all messages regardless of their severity and the observer will see
        // each message only once.

        ball::LoggerManagerConfiguration configuration;
        ASSERT(0 == configuration.setDefaultThresholdLevelsIfValid(
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_TRACE,
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_OFF));

        ball::LoggerManagerScopedGuard guard(configuration);

        ball::LoggerManager& manager = ball::LoggerManager::singleton();

        // Don't run this if we're in the debugger because the debugger stops
        // and refuses to continue when we hit the file size limit.

        if (verbose) cerr << "Testing output when the stream fails"
                          << " (UNIX only)."
                          << endl;

        bslma::TestAllocator ta(veryVeryVeryVerbose);

        // Temporary directory for test files.
        bdls::TempDirectoryGuard tempDirGuard("ball_");

        {
            bsl::string fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            struct rlimit rlim;
            ASSERT(0 == getrlimit(RLIMIT_FSIZE, &rlim));
            rlim.rlim_cur = 2048;
            ASSERT(0 == setrlimit(RLIMIT_FSIZE, &rlim));

            struct sigaction act,oact;
            act.sa_handler = SIG_IGN;
            sigemptyset(&act.sa_mask);
            act.sa_flags = 0;
            ASSERT(0 == sigaction(SIGXFSZ, &act, &oact));

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_OFF,
                                                       true,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            bsl::stringstream os;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            // We want to capture the error message that will be written to
            // stderr (not cerr).  Redirect stderr to a file.  We can't
            // redirect it back; we'll have to use 'ASSERT2' (which outputs to
            // cout, not cerr) from now on and report a summary to cout at the
            // end of this case.
            bsl::string stderrFN(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&stderrFN, "stderrLog");

            ASSERT(stderr == freopen(stderrFN.c_str(), "w", stderr));

            ASSERT2(0    == mX->enableFileLogging(fileName.c_str(), true));
            ASSERT2(true == X->isFileLoggingEnabled());
            ASSERT2(1    == mX->enableFileLogging(fileName.c_str(), true));

            for (int i = 0; i < 40 ; ++i) {
                BALL_LOG_TRACE << "log";
            }

            fflush(stderr);
            bsl::fstream stderrFs;
            stderrFs.open(stderrFN.c_str(), bsl::ios_base::in);

            bsl::string line;
            ASSERT2(getline(stderrFs, line)); // we caught an error

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));

            if (testStatus > 0) {
                cout << "Error, non-zero test status = " << testStatus
                     << "." << endl;
            }
        }
#else
        if (verbose) {
            cout << "Skipping case 4 on Windows and Cygwin..." << endl;
        }
#endif
      } break;
      case 2: {
        // --------------------------------------------------------------------
        // TESTING FILE ROTATION
        //
        // Concerns:
        //:  1 'rotateOnSize' triggers a rotation when expected.
        //:
        //:  2 'disableSizeRotation' disables rotation on size.
        //:
        //:  3 'forceRotation' triggers a rotation.
        //:
        //:  4 'rotateOnLifetime' triggers a rotation when expected.
        //:
        //:  5 'disableLifetimeRotation' disables rotation on lifetime.
        //
        // Plan:
        //:  1 We exercise both rotation rules to verify that they work
        //:    properly using glob to count the files and proper timing.  We
        //:    also verify that the size rule is followed by checking the size
        //:    of log files.
        //
        // Testing:
        //   void disableLifetimeRotation();
        //   void disableSizeRotation();
        //   void disableTimeIntervalRotation();
        //   void forceRotation();
        //   void rotateOnLifetime(DatetimeInterval& interval);
        //   void rotateOnSize(int size);
        //   void rotateOnTimeInterval(const DatetimeInterval& interval);
        //   void rotateOnTimeInterval(const DtInterval& i, const Datetime& s);
        //   bdlt::DatetimeInterval rotationLifetime() const;
        //   int rotationSize() const;
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTESTING FILE ROTATION"
                          << "\n=====================" << endl;

        // This configuration guarantees that the logger manager will publish
        // all messages regardless of their severity and the observer will see
        // each message only once.

        ball::LoggerManagerConfiguration configuration;
        ASSERT(0 == configuration.setDefaultThresholdLevelsIfValid(
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_TRACE,
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_OFF));

        ball::LoggerManagerScopedGuard guard(configuration);

        ball::LoggerManager& manager = ball::LoggerManager::singleton();

#ifdef BSLS_PLATFORM_OS_UNIX
        bslma::TestAllocator ta(veryVeryVeryVerbose);

        if (verbose) cout << "Test-case infrastructure setup." << endl;
        {
            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_OFF,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            if (verbose) cout << "Testing setup." << endl;

            {
                ASSERT(0    == mX->enableFileLogging(fileName.c_str(), true));
                ASSERT(true == X->isFileLoggingEnabled());
                ASSERT(1    == mX->enableFileLogging(fileName.c_str(), true));

                BALL_LOG_TRACE << "log 1";

                glob_t globbuf;

                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(1 == globbuf.gl_pathc);

                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                globfree(&globbuf);

                ASSERT(fs.is_open());

                int         linesNum = 0;
                bsl::string line;

                while (getline(fs, line)) {
                    ++linesNum;
                }
                fs.close();
                ASSERT(2    == linesNum);
                ASSERT(true == X->isFileLoggingEnabled());
            }

            if (verbose) cout << "Testing lifetime-constrained rotation."
                              << endl;
            {
                ASSERT(bdlt::DatetimeInterval(0) == X->rotationLifetime());
                mX->rotateOnLifetime(bdlt::DatetimeInterval(0,0,0,3));
                ASSERT(bdlt::DatetimeInterval(0,0,0,3) ==
                                                        X->rotationLifetime());
                bslmt::ThreadUtil::microSleep(0, 4);
                BALL_LOG_TRACE << "log 1";
                BALL_LOG_DEBUG << "log 2";

                // Check that a rotation occurred.

                glob_t globbuf;
                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(2 == globbuf.gl_pathc);

                // Check the number of lines in the file.

                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[1], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);
                ASSERT(fs.is_open());

                int         linesNum = 0;
                bsl::string line(&ta);

                while (getline(fs, line)) {
                    ++linesNum;
                }

                fs.close();
                ASSERT(4 == linesNum);

                mX->disableLifetimeRotation();
                bslmt::ThreadUtil::microSleep(0, 4);

                BALL_LOG_FATAL << "log 3";

                // Check that no rotation occurred.

                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(2 == globbuf.gl_pathc);

                fs.open(globbuf.gl_pathv[1], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);

                ASSERT(fs.is_open());
                linesNum = 0;

                while (getline(fs, line)) {
                    ++linesNum;
                }

                fs.close();
                ASSERT(6 == linesNum);
            }

            if (verbose) cout << "Testing forced rotation." << endl;
            {
                bslmt::ThreadUtil::microSleep(0, 2);
                mX->forceRotation();

                BALL_LOG_TRACE << "log 1";
                BALL_LOG_DEBUG << "log 2";
                BALL_LOG_INFO  << "log 3";
                BALL_LOG_WARN  << "log 4";

                // Check that the rotation occurred.

                glob_t globbuf;
                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(3 == globbuf.gl_pathc);

                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[2], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);

                ASSERT(fs.is_open());

                int         linesNum = 0;
                bsl::string line(&ta);

                while (getline(fs, line)) {
                    ++linesNum;
                }

                fs.close();
                ASSERT(8 == linesNum);
            }

            if (verbose) cout << "Testing size-constrained rotation." << endl;
            {
                bslmt::ThreadUtil::microSleep(0, 2);
                ASSERT(0 == X->rotationSize());
                mX->rotateOnSize(1);
                ASSERT(1 == X->rotationSize());

                for (int i = 0; i < 30; ++i) {
                    BALL_LOG_TRACE << "log";

                    // We sleep because otherwise, the loop is too fast to make
                    // the timestamp change so we cannot observe the rotation.

                    bslmt::ThreadUtil::microSleep(200 * 1000);
                }

                glob_t globbuf;
                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(4 <= globbuf.gl_pathc);

                // We are not checking the last one since we do not have any
                // information on its size.

                bsl::ifstream fs;
                for (size_t i = 0; i < globbuf.gl_pathc - 3; ++i) {
                    fs.open(globbuf.gl_pathv[i + 2], bsl::ifstream::in);
                    fs.clear();

                    ASSERT(fs.is_open());

                    bsl::string::size_type fileSize = 0;
                    bsl::string            line(&ta);

                    while (getline(fs, line)) {
                        fileSize += line.length() + 1;
                    }

                    fs.close();
                    ASSERT(fileSize > 1024);
                }

                int oldNumFiles = static_cast<int>(globbuf.gl_pathc);
                globfree(&globbuf);

                ASSERT(1 == X->rotationSize());
                mX->disableSizeRotation();
                ASSERT(0 == X->rotationSize());

                for (int i = 0; i < 30; ++i) {
                    BALL_LOG_TRACE << "log";
                    bslmt::ThreadUtil::microSleep(50 * 1000);
                }

                // Verify that no rotation occurred.

                ASSERT(0 == glob((fileName + ".2*").c_str(), 0, 0, &globbuf));
                ASSERT(oldNumFiles == (int)globbuf.gl_pathc);
                globfree(&globbuf);
            }

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }
        {
            // Test with no timestamp.

            // Temporary directory for test files.
            bdls::TempDirectoryGuard tempDirGuard("ball_");
            bsl::string              fileName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fileName, "testLog");

            if (verbose) cout << "Test-case infrastructure setup." << endl;

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_OFF,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            if (verbose) cout << "Testing setup." << endl;
            {
                ASSERT(0    == mX->enableFileLogging(fileName.c_str(), false));
                ASSERT(true == X->isFileLoggingEnabled());
                ASSERT(1    == mX->enableFileLogging(fileName.c_str(), false));

                BALL_LOG_TRACE << "log 1";

                glob_t globbuf;
                ASSERT(0 == glob((fileName + "*").c_str(), 0, 0, &globbuf));
                ASSERT(1 == globbuf.gl_pathc);

                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                globfree(&globbuf);

                ASSERT(fs.is_open());

                int         linesNum = 0;
                bsl::string line;

                while (getline(fs, line)) {
                    ++linesNum;
                }

                fs.close();

                ASSERT(2    == linesNum);
                ASSERT(true == X->isFileLoggingEnabled());
            }

            if (verbose) cout << "Testing lifetime-constrained rotation."
                              << endl;
            {
                ASSERT(bdlt::DatetimeInterval(0) == X->rotationLifetime());

                mX->rotateOnLifetime(bdlt::DatetimeInterval(0,0,0,3));

                ASSERT(bdlt::DatetimeInterval(0,0,0,3) ==
                                                        X->rotationLifetime());

                bslmt::ThreadUtil::microSleep(0, 4);

                BALL_LOG_TRACE << "log 1";
                BALL_LOG_DEBUG << "log 2";

                // Check that a rotation occurred.

                glob_t globbuf;
                ASSERT(0 == glob((fileName + "*").c_str(), 0, 0, &globbuf));
                ASSERT(2 == globbuf.gl_pathc);

                // Check the number of lines in the file.

                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);
                ASSERT(fs.is_open());

                int         linesNum = 0;
                bsl::string line(&ta);
                while (getline(fs, line)) {
                    ++linesNum;
                }
                fs.close();
                ASSERT(4 == linesNum);

                mX->disableLifetimeRotation();
                bslmt::ThreadUtil::microSleep(0, 4);
                BALL_LOG_FATAL << "log 3";

                // Check that no rotation occurred.

                ASSERT(0 == glob((fileName + "*").c_str(), 0, 0, &globbuf));
                ASSERT(2 == globbuf.gl_pathc);

                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);

                ASSERT(fs.is_open());

                linesNum = 0;

                while (getline(fs, line)) {
                    ++linesNum;
                }

                fs.close();

                ASSERT(6 == linesNum);
            }

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }
#endif
      } break;
      case 1: {
        // --------------------------------------------------------------------
        // TESTING THRESHOLDS AND OUTPUT FORMAT
        //
        // Concerns:
        //:  1 'publish' logs in the expected format:
        //:     a. using enable/disableUserFieldsLogging
        //:     b. using enable/disableStdoutLogging
        //:
        //:  2 'publish' properly ignores the severity below the one specified
        //:     at construction on 'stdout'.
        //:
        //:  3 'publish' publishes all messages to a file if file logging
        //:    is enabled.
        //:
        //:  4 The name of the log file should be in accordance with what is
        //:    defined by the given pattern if file logging is enabled by a
        //:    pattern.
        //:
        //:  5 'setLogFormat' can change to the desired output format for both
        //:    'stdout' and the log file.
        //
        // Plan:
        //:  1 We will set up the observer and check if logged messages are in
        //:    the expected format and contain the expected data by comparing
        //:    the output of this observer with 'ball::StreamObserver', that we
        //:    slightly modify.  Then, we will configure the observer to ignore
        //:    different severity and test if only the expected messages are
        //:    published.  We will use different manipulators to affect output
        //:    format and verify that it has changed where expected.
        //
        // Testing:
        //   FileObserver(Severity::Level, bslma::Allocator * = 0);
        //   FileObserver(Severity::Level, bool, bslma::Allocator * = 0);
        //   ~FileObserver();
        //   void disableFileLogging();
        //   void disableStdoutLoggingPrefix();
        //   void disableUserFieldsLogging();
        //   int  enableFileLogging(const char *fileName);
        //   int  enableFileLogging(const char *fileName, bool timestampFlag);
        //   void enableStdoutLoggingPrefix();
        //   void enableUserFieldsLogging();
        //   void publish(const Record& record, const Context& context);
        //   void publish(const shared_ptr<Record>&, const Context&);
        //   void setStdoutThreshold(ball::Severity::Level stdoutThreshold);
        //   bool isFileLoggingEnabled() const;
        //   bool isStdoutLoggingPrefixEnabled() const;
        //   bool isUserFieldsLoggingEnabled() const;
        //   ball::Severity::Level stdoutThreshold() const;
        //   bool isPublishInLocalTimeEnabled() const;
        //   void setLogFormat(const char*, const char*);
        //   void getLogFormat(const char**, const char**) const;
        // --------------------------------------------------------------------

        if (verbose) cout << "\nTESTING THRESHOLDS AND OUTPUT FORMAT."
                             "\n=====================================" << endl;

        // This configuration guarantees that the logger manager will publish
        // all messages regardless of their severity and the observer will see
        // each message only once.

        ball::LoggerManagerConfiguration configuration;
        ASSERT(0 == configuration.setDefaultThresholdLevelsIfValid(
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_TRACE,
                                                       ball::Severity::e_OFF,
                                                       ball::Severity::e_OFF));

        ball::LoggerManagerScopedGuard guard(configuration);

        ball::LoggerManager& manager = ball::LoggerManager::singleton();

        bslma::TestAllocator ta(veryVeryVeryVerbose);
        bslma::TestAllocator oa(veryVeryVeryVerbose);

        // Temporary directory for test files.
        bdls::TempDirectoryGuard tempDirGuard("ball_");
        bsl::string              fileName(tempDirGuard.getTempDirName());
        bdls::PathUtil::appendRaw(&fileName, "testLog");
        {
            const FILE *out = stdout;
            ASSERT(out == freopen(fileName.c_str(), "w", stdout));
            fflush(stdout);
        }

        ASSERT(FsUtil::exists(fileName));
        ASSERT(0 == FsUtil::getFileSize(fileName));

#if defined(BSLS_PLATFORM_OS_UNIX) && \
   (!defined(BSLS_PLATFORM_OS_SOLARIS) || BSLS_PLATFORM_OS_VER_MAJOR >= 10)
        // For the localtime to be picked to avoid the all.pl environment to
        // pollute us.
        unsetenv("TZ");
#endif

        if (verbose) cerr << "Testing threshold and output format." << endl;
        {
            Obj *mX_p = new (ta) Obj(ball::Severity::e_ERROR, true, &oa);
            bsl::shared_ptr<Obj>       mX(mX_p, &ta);
            bsl::shared_ptr<const Obj>  X = mX;

            ASSERT(ball::Severity::e_ERROR == X->stdoutThreshold());
            ASSERTV(&oa == X->allocator());
            ASSERTV(X->isPublishInLocalTimeEnabled());

            mX->disablePublishInLocalTime();
            ASSERTV(!X->isPublishInLocalTimeEnabled());

            bsl::ostringstream os, dos;

            bsl::shared_ptr<ball::StreamObserver>
                                refX(new (ta) ball::StreamObserver(&dos), &ta);

            ASSERT(0 == manager.registerObserver(mX,   "testObserver"));
            ASSERT(0 == manager.registerObserver(refX, "refObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            bsl::streambuf *coutSbuf = bsl::cout.rdbuf();

            bsl::cout.rdbuf(os.rdbuf());

            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            // these two lines are a desperate kludge to make windows work --
            // this test driver works everywhere else without them.

            (void) readPartialFile(fileName, 0);
            fileOffset = FsUtil::getFileSize(fileName);

            BALL_LOG_TRACE << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_DEBUG << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_INFO << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_WARN << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_WARN << "not logged";
            ASSERT("" == readPartialFile(fileName, fileOffset));
            dos.str("");

            BALL_LOG_ERROR << "log ERROR";
            // Replace the spaces after pid, __FILE__ to make dos match the
            // file
            {
                bsl::string temp = dos.str();
                temp[temp.find(__FILE__) + sizeof(__FILE__) - 1] = ':';
                replaceSecondSpace(&temp, ':');
                dos.str(temp);
            }

            if (veryVeryVerbose) { P_(dos.str()); P(os.str()); }

            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(dos.str(), coutS, dos.str() == coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);
            dos.str("");

            BALL_LOG_FATAL << "log FATAL";
            // Replace the spaces after pid, __FILE__ to make dos match the
            // file
            {
                bsl::string temp = dos.str();
                temp[temp.find(__FILE__) + sizeof(__FILE__) - 1] = ':';
                replaceSecondSpace(&temp, ':');
                dos.str(temp);
            }

            if (veryVeryVerbose) { P_(dos.str()); P(os.str()); }

            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(dos.str(), coutS, dos.str() == coutS);
            }

            fileOffset = FsUtil::getFileSize(fileName);
            dos.str("");

            bsl::cout.rdbuf(coutSbuf);

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
            ASSERT(0 == manager.deregisterObserver("refObserver"));
        }

        if (verbose) cerr << "Testing constructor threshold." << endl;
        {
            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_FATAL,
                                                       &ta),
                                          &ta);

            bsl::ostringstream os, dos;

            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            bsl::shared_ptr<ball::StreamObserver>
                                refX(new (ta) ball::StreamObserver(&dos), &ta);

            ASSERT(0 == manager.registerObserver(mX,   "testObserver"));
            ASSERT(0 == manager.registerObserver(refX, "refObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            bsl::streambuf *coutSbuf = bsl::cout.rdbuf();

            bsl::cout.rdbuf(os.rdbuf());
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_TRACE << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_DEBUG << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_INFO << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_WARN << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_ERROR << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);
            dos.str("");

            BALL_LOG_FATAL << "log";
            // Replace the spaces after pid, __FILE__ to make dos match the
            // file.
            {
                bsl::string temp = dos.str();
                temp[temp.find(__FILE__) + sizeof(__FILE__) - 1] = ':';
                replaceSecondSpace(&temp, ':');
                dos.str(temp);
            }

            if (veryVeryVerbose) { P_(dos.str()); P(os.str()); }

            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(dos.str(), coutS, dos.str() == coutS);
            }

            ASSERT(dos.str() == readPartialFile(fileName, fileOffset));
            fileOffset = FsUtil::getFileSize(fileName);
            dos.str("");

            ASSERT("" == os.str());

            bsl::cout.rdbuf(coutSbuf);

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
            ASSERT(0 == manager.deregisterObserver("refObserver"));
        }

        if (verbose) cerr << "Testing short format." << endl;
        {
            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(false == X->isPublishInLocalTimeEnabled());
            ASSERT(true  == X->isStdoutLoggingPrefixEnabled());

            mX->disableStdoutLoggingPrefix();

            ASSERT(false == X->isStdoutLoggingPrefixEnabled());

            bsl::ostringstream os, testOs, dos;

            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            bsl::shared_ptr<ball::StreamObserver>
                                refX(new (ta) ball::StreamObserver(&dos), &ta);

            ASSERT(0 == manager.registerObserver(mX,   "testObserver"));
            ASSERT(0 == manager.registerObserver(refX, "refObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            bsl::streambuf *coutSbuf = bsl::cout.rdbuf();

            bsl::cout.rdbuf(os.rdbuf());

            BALL_LOG_TRACE << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_DEBUG << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_INFO << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_WARN << "log WARN";
            testOs << "\nWARN " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log WARN " << "\n";
            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(testOs.str(), coutS, testOs.str() == coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);
            testOs.str("");

            BALL_LOG_ERROR << "log ERROR";
            testOs << "\nERROR " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log ERROR " << "\n";
            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(testOs.str(), coutS, testOs.str() == coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);
            testOs.str("");

            ASSERT(false == X->isStdoutLoggingPrefixEnabled());

            mX->enableStdoutLoggingPrefix();

            ASSERT(true  == X->isStdoutLoggingPrefixEnabled());

            dos.str("");

            BALL_LOG_FATAL << "log FATAL";
            testOs << "\nFATAL " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log FATAL " << "\n";
            {
                // Replace the spaces after pid, __FILE__ to make dos match the
                // file.
                bsl::string temp = dos.str();
                temp[temp.find(__FILE__) + sizeof(__FILE__) - 1] = ':';
                replaceSecondSpace(&temp, ':');
                dos.str(temp);

                bsl::string coutS = readPartialFile(fileName, fileOffset);
                if (veryVeryVerbose) { P_(dos.str()); P(coutS); }
                ASSERTV(dos.str(), coutS, dos.str() == coutS);
                ASSERT(testOs.str() != coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);

            ASSERT("" == os.str());

            bsl::cout.rdbuf(coutSbuf);

            // 'setLogFormat' implicitly disables short format.
            {
                mX->disableStdoutLoggingPrefix();
                ASSERT(false == X->isStdoutLoggingPrefixEnabled());

                mX->setLogFormat("%d %p %t %s %l %c %m %u",
                                 "%i %p %t %s %l %c %m %u");
                ASSERT(true == X->isStdoutLoggingPrefixEnabled());
            }

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
            ASSERT(0 == manager.deregisterObserver("refObserver"));
        }

        if (verbose) cerr << "Testing short format with local time "
                          << "offset."
                          << endl;
        {
            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       true,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(true  == X->isPublishInLocalTimeEnabled());
            ASSERT(true  == X->isStdoutLoggingPrefixEnabled());

            mX->disableStdoutLoggingPrefix();

            ASSERT(false == X->isStdoutLoggingPrefixEnabled());

            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            bsl::ostringstream os, testOs, dos;

            bsl::shared_ptr<ball::StreamObserver>
                                refX(new (ta) ball::StreamObserver(&dos), &ta);

            ASSERT(0 == manager.registerObserver(mX,   "testObserver"));
            ASSERT(0 == manager.registerObserver(refX, "refObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            bsl::streambuf *coutSbuf = bsl::cout.rdbuf();

            bsl::cout.rdbuf(os.rdbuf());

            BALL_LOG_TRACE << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_DEBUG << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_INFO << "not logged";
            ASSERT(FsUtil::getFileSize(fileName) == fileOffset);

            BALL_LOG_WARN << "log WARN";
            testOs << "\nWARN " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log WARN " << "\n";
            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(testOs.str(), coutS, testOs.str() == coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);
            testOs.str("");

            BALL_LOG_ERROR << "log ERROR";
            testOs << "\nERROR " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log ERROR " << "\n";
            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                ASSERTV(testOs.str(), coutS, testOs.str() == coutS);
            }
            fileOffset = FsUtil::getFileSize(fileName);
            testOs.str("");

            ASSERT(false == X->isStdoutLoggingPrefixEnabled());

            mX->enableStdoutLoggingPrefix();

            ASSERT(true  == X->isStdoutLoggingPrefixEnabled());

            dos.str("");

            BALL_LOG_FATAL << "log FATAL";
            testOs << "FATAL " << __FILE__ << ":" << __LINE__ - 1 <<
                      " TestCategory log FATAL " << "\n";
            // Replace the spaces after pid, __FILE__.
            {
                bsl::string temp = dos.str();
                temp[temp.find(__FILE__) + sizeof(__FILE__) - 1] = ':';
                replaceSecondSpace(&temp, ':');
                dos.str(temp);
            }

            {
                bsl::string coutS = readPartialFile(fileName, fileOffset);
                if (0 == bdlt::LocalTimeOffset::localTimeOffset(
                                    bdlt::CurrentTime::utc()).totalSeconds()) {
                    ASSERTV(dos.str(), os.str(), dos.str() == coutS);
                }
                else {
                    ASSERTV(dos.str(), os.str(), dos.str() != coutS);
                }
                ASSERT(testOs.str() != coutS);
                ASSERTV(coutS, testOs.str(),
                            bsl::string::npos != coutS.find(testOs.str()));

                // Now let's verify the actual difference.
                int defaultObsHour = 0;
                if (dos.str().length() >= 11) {
                    bsl::istringstream is(dos.str().substr(11, 2));
                    ASSERT(is >> defaultObsHour);
                } else {
                    ASSERT(0 && "can't substr(11,2), string too short");
                }
                int fileObsHour = 0;
                if (coutS.length() >= 11) {
                    bsl::istringstream is(coutS.substr(11, 2));
                    ASSERT(is >> fileObsHour);
                } else {
                    ASSERT(0 && "can't substr(11,2), string too short");
                }
                int difference = bdlt::CurrentTime::utc().hour() -
                                 bdlt::CurrentTime::local().hour();
                ASSERTV(fileObsHour, defaultObsHour, difference,
                       (fileObsHour + difference + 24) % 24 == defaultObsHour);
                {
                    bsl::string temp = dos.str();
                    temp[11] = coutS[11];
                    temp[12] = coutS[12];
                    dos.str(temp);
                }

                if (defaultObsHour - difference >= 0 &&
                    defaultObsHour - difference < 24) {
                    // UTC and local time are on the same day

                    if (veryVeryVerbose) { P_(dos.str()); P(coutS); }
                }
                else if (coutS.length() >= 11) {
                    // UTC and local time are on different days.  Ignore date.

                    ASSERT(dos.str().substr(10) == os.str().substr(10));
                } else {
                    ASSERT(0 && "can't substr(11,2), string too short");
                }
                fileOffset = FsUtil::getFileSize(fileName);
                ASSERT(0 == os.str().length());

                bsl::cout.rdbuf(coutSbuf);

                // Deregister here as we used local allocator for the observer.
                ASSERT(0 == manager.deregisterObserver("testObserver"));
                ASSERT(0 == manager.deregisterObserver("refObserver"));
            }
        }

        if (verbose) cerr << "Testing file logging." << endl;
        {
            bsl::string fn(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fn, "test2Log");

            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            if (veryVerbose) {
                cerr << fn << endl;
                cerr << fileName << endl;
            }
            bsl::stringstream  ss;
            bsl::streambuf    *coutSbuf = bsl::cout.rdbuf();
            bsl::cout.rdbuf(ss.rdbuf());

            ASSERT(0    == mX->enableFileLogging(fn.c_str()));
            ASSERT(true == X->isFileLoggingEnabled());
            ASSERT(1    == mX->enableFileLogging(fn.c_str()));

            BALL_LOG_TRACE << "log 1";
            BALL_LOG_DEBUG << "log 2";
            BALL_LOG_INFO <<  "log 3";
            BALL_LOG_WARN <<  "log 4";
            BALL_LOG_ERROR << "log 5";
            BALL_LOG_FATAL << "log 6";

            bsl::ifstream fs, coutFs;

            fs.open(fn.c_str(),           bsl::ifstream::in);
            coutFs.open(fileName.c_str(), bsl::ifstream::in);

            ASSERT(fs.is_open());
            ASSERT(coutFs.is_open());

            coutFs.seekg(fileOffset);

            int         linesNum = 0;
            bsl::string line;

            while (getline(fs, line)) {
                bsl::cerr << "Line:     '" << line << "'" << endl;
                if (linesNum >= 6) {
                    // check format
                    bsl::string coutLine;
                    getline(coutFs, coutLine);

                    ASSERTV(coutLine, line, coutLine == line);
                    bsl::cerr << "coutLine: '" << coutLine << "'" << endl
                              << "line:     '" << line << "'" <<endl;
                }
                ++linesNum;
            }
            fs.close();

            ASSERT(!getline(coutFs, line));

            coutFs.close();

            ASSERTV(linesNum, 12 == linesNum);

            ASSERT(true  == X->isFileLoggingEnabled());
            mX->disableFileLogging();
            ASSERT(false == X->isFileLoggingEnabled());

            BALL_LOG_TRACE << "log 1";
            BALL_LOG_DEBUG << "log 2";
            BALL_LOG_INFO <<  "log 3";
            BALL_LOG_WARN <<  "log 4";
            BALL_LOG_ERROR << "log 5";
            BALL_LOG_FATAL << "log 6";

            fs.open(fn.c_str(), bsl::ifstream::in);

            ASSERT(fs.is_open());

            fs.clear();

            linesNum = 0;
            while (getline(fs, line)) {
                ++linesNum;
            }
            fs.close();

            ASSERT(12 == linesNum);

            ASSERT(0    == mX->enableFileLogging(fn.c_str()));
            ASSERT(true == X->isFileLoggingEnabled());
            ASSERT(1    == mX->enableFileLogging(fn.c_str()));

            BALL_LOG_TRACE << "log 7";
            BALL_LOG_DEBUG << "log 8";
            BALL_LOG_INFO  << "log 9";
            BALL_LOG_WARN  << "log 1";
            BALL_LOG_ERROR << "log 2";
            BALL_LOG_FATAL << "log 3";

            fs.open(fn.c_str(), bsl::ifstream::in);
            ASSERT(fs.is_open());
            fs.clear();

            linesNum = 0;
            while (getline(fs, line)) {
                ++linesNum;
            }

            fs.close();

            ASSERT(24 == linesNum);

            bsl::cout.rdbuf(coutSbuf);

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }

#ifdef BSLS_PLATFORM_OS_UNIX
        if (verbose) cerr << "Testing file logging with timestamp."
                          << endl;
        {
            bsl::string fn(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&fn, "test3Log");

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            bsl::ostringstream os;

            BALL_LOG_SET_CATEGORY("TestCategory");

            bsl::streambuf *coutSbuf = bsl::cout.rdbuf();
            bsl::cout.rdbuf(os.rdbuf());
            ASSERT(0    == mX->enableFileLogging(fn.c_str(), true));
            ASSERT(true == X->isFileLoggingEnabled());
            ASSERT(1    == mX->enableFileLogging(fn.c_str(), true));

            BALL_LOG_TRACE << "log 1";
            BALL_LOG_DEBUG << "log 2";
            BALL_LOG_INFO <<  "log 3";
            BALL_LOG_WARN <<  "log 4";
            BALL_LOG_ERROR << "log 5";
            BALL_LOG_FATAL << "log 6";

            glob_t globbuf;
            ASSERT(0 == glob((fn + ".2*").c_str(), 0, 0, &globbuf));
            ASSERT(1 == globbuf.gl_pathc);
            bsl::ifstream fs;
            fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
            ASSERT(fs.is_open());

            int         linesNum = 0;
            bsl::string line;

            while (getline(fs, line)) {
                ++linesNum;
            }

            fs.close();

            ASSERT(12   == linesNum);
            ASSERT(true == X->isFileLoggingEnabled());

            bsl::cout.rdbuf(coutSbuf);

            ASSERT("" == os.str());

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }

        if (verbose) cerr << "Testing log file name pattern." << endl;
        {
            bsl::string baseName(tempDirGuard.getTempDirName());
            bdls::PathUtil::appendRaw(&baseName, "test4Log");

            bsl::string pattern  = baseName + "%Y%M%D%h%m%s-%p";

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            bdlt::Datetime startDatetime, endDatetime;

            mX->disableLifetimeRotation();
            mX->disableSizeRotation();
            mX->disableFileLogging();
            mX->enablePublishInLocalTime();

            // loop until startDatetime is equal to endDatetime
            do {
                startDatetime = getCurrentTimestamp();

                ASSERT(0    == mX->enableFileLogging(pattern.c_str(), false));
                ASSERT(true == X->isFileLoggingEnabled());
                ASSERT(1    == mX->enableFileLogging(pattern.c_str(), false));

                endDatetime = getCurrentTimestamp();

                if (startDatetime.date()   != endDatetime.date()
                 || startDatetime.hour()   != endDatetime.hour()
                 || startDatetime.minute() != endDatetime.minute()
                 || startDatetime.second() != endDatetime.second()) {
                    // not sure the exact time when the log file was opened
                    // because startDatetime and endDatetime are different;
                    // will try it again
                    bsl::string fn;
                    ASSERT(true  == mX->isFileLoggingEnabled(&fn));
                    mX->disableFileLogging();
                    ASSERT(false == bsl::remove(fn.c_str()));
                }
            } while (!X->isFileLoggingEnabled());

            ASSERT(startDatetime.year()   == endDatetime.year());
            ASSERT(startDatetime.month()  == endDatetime.month());
            ASSERT(startDatetime.day()    == endDatetime.day());
            ASSERT(startDatetime.hour()   == endDatetime.hour());
            ASSERT(startDatetime.minute() == endDatetime.minute());
            ASSERT(startDatetime.second() == endDatetime.second());

            BALL_LOG_INFO<< "log";

            mX->disableFileLogging();

            // now construct the name of the log file from startDatetime
            bsl::ostringstream fnOs;
            fnOs << baseName;
            fnOs << bsl::setw(4) << bsl::setfill('0')
                     << startDatetime.year();
            fnOs << bsl::setw(2) << bsl::setfill('0')
                     << startDatetime.month();
            fnOs << bsl::setw(2) << bsl::setfill('0')
                     << startDatetime.day();
            fnOs << bsl::setw(2) << bsl::setfill('0')
                     << startDatetime.hour();
            fnOs << bsl::setw(2) << bsl::setfill('0')
                     << startDatetime.minute();
            fnOs << bsl::setw(2) << bsl::setfill('0')
                     << startDatetime.second();
            fnOs << "-";
            fnOs << bdls::ProcessUtil::getProcessId();

            // look for the file with the constructed name
            glob_t globbuf;
            ASSERTV(fnOs.str(), 0 == glob(fnOs.str().c_str(), 0, 0, &globbuf));
            ASSERTV(globbuf.gl_pathc, 1 == globbuf.gl_pathc);

            // read the file to get the number of lines
            bsl::ifstream fs;
            fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
            fs.clear();
            globfree(&globbuf);

            ASSERT(fs.is_open());

            int         linesNum = 0;
            bsl::string line;

            while (getline(fs, line)) {
                ++linesNum;
            }
            fs.close();

            ASSERT(2 == linesNum);

            mX->disableFileLogging();

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }

        if (verbose) cerr << "Testing '%%' in file name pattern." << endl;
        {
            static const struct {
                int         d_lineNum;           // source line number
                const char *d_patternSuffix_p;   // pattern suffix
                const char *d_filenameSuffix_p;  // filename suffix
            } DATA[] = {
                //line  pattern suffix  filename suffix
                //----  --------------  ---------------
                { L_,   "foo",          "foo"                           },
                { L_,   "foo%",         "foo%"                          },
                { L_,   "foo%bar",      "foo%bar"                       },
                { L_,   "foo%%",        "foo"                           },
                { L_,   "foo%%bar",     "foobar"                        },
                { L_,   "foo%%%",       "foo%"                          },
                { L_,   "foo%%%bar",    "foo%bar"                       },
            };
            enum { NUM_DATA = sizeof DATA / sizeof *DATA };

            for (int ti = 0; ti < NUM_DATA; ++ti) {
                const int   LINE     = DATA[ti].d_lineNum;
                const char *PATTERN  = DATA[ti].d_patternSuffix_p;
                const char *FILENAME = DATA[ti].d_filenameSuffix_p;

                bsl::string baseName(tempDirGuard.getTempDirName());
                bdls::PathUtil::appendRaw(&baseName, "test5Log");

                bsl::string pattern(baseName);   pattern  += PATTERN;
                bsl::string expected(baseName);  expected += FILENAME;
                bsl::string actual;

                bsl::shared_ptr<Obj>       mX(new (ta) Obj(
                                                        ball::Severity::e_WARN,
                                                        &ta),
                                              &ta);
                bsl::shared_ptr<const Obj> X = mX;

                ASSERTV(LINE, 0    == mX->enableFileLogging(
                                                      pattern.c_str(), false));
                ASSERTV(LINE, true == X->isFileLoggingEnabled(&actual));

                if (veryVeryVerbose) {
                    P_(PATTERN);  P_(expected);  P(actual);
                }

                ASSERTV(LINE, expected == actual);

                mX->disableFileLogging();

                // look for the file with the expected name
                glob_t globbuf;
                ASSERTV(LINE, 0 == glob(expected.c_str(), 0, 0, &globbuf));
                ASSERTV(LINE, 1 == globbuf.gl_pathc);

                removeFilesByPrefix(expected.c_str());
            }
        }

        if (verbose) cerr << "Testing customized format." << endl;
        {
            FsUtil::Offset fileOffset = FsUtil::getFileSize(fileName);

            bsl::shared_ptr<Obj>       mX(new (ta) Obj(ball::Severity::e_WARN,
                                                       &ta),
                                          &ta);
            bsl::shared_ptr<const Obj> X = mX;

            ASSERT(ball::Severity::e_WARN == X->stdoutThreshold());

            ASSERT(0 == manager.registerObserver(mX, "testObserver"));

            BALL_LOG_SET_CATEGORY("TestCategory");

            // redirect 'stdout' to a string stream
            {
                bsl::string baseName(tempDirGuard.getTempDirName());
                bdls::PathUtil::appendRaw(&baseName, "test6Log");

                ASSERT(0    == mX->enableFileLogging(baseName.c_str(), false));
                ASSERT(true == X->isFileLoggingEnabled());
                ASSERT(1    == mX->enableFileLogging(baseName.c_str(), false));

                bsl::stringstream  os;
                bsl::streambuf    *coutSbuf = bsl::cout.rdbuf();
                bsl::cout.rdbuf(os.rdbuf());

                // for log file, use bdlt::Datetime format for stdout, use ISO
                // format

                mX->setLogFormat("%d %p %t %s %l %c %m %u",
                                 "%i %p %t %s %l %c %m %u");
                ASSERT(true == X->isStdoutLoggingPrefixEnabled());

                fileOffset = FsUtil::getFileSize(fileName);

                BALL_LOG_WARN << "log";

                // look for the file with the constructed name
                glob_t globbuf;
                ASSERT(0 == glob(baseName.c_str(), 0, 0, &globbuf));
                ASSERT(1 == globbuf.gl_pathc);

                // read the log file to get the record
                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);

                ASSERT(fs.is_open());
                bsl::string line;
                ASSERT(getline(fs, line));
                fs.close();

                bsl::string            datetime1, datetime2;
                bsl::string            log1, log2;
                bsl::string::size_type pos;

                // divide line into datetime and the rest
                pos = line.find(' ');
                datetime1 = line.substr(0, pos);
                log1 = line.substr(pos, line.length());

                fflush(stdout);
                bsl::string fStr = readPartialFile(fileName, fileOffset);

                ASSERT("" == os.str());

                // divide os.str() into datetime and the rest
                pos = fStr.find(' ');
                ASSERT(bsl::string::npos != pos);
                datetime2 = fStr.substr(0, pos);

                log2 = fStr.substr(pos, fStr.length()-pos);

                ASSERTV(log1, log2, log1 == log2);

                // Now we try to convert datetime2 from ISO to bdlt::Datetime

                bsl::istringstream iss(datetime2);

                int  year, month, day, hour, minute, second;
                char c;

                iss >> year >> c >> month >> c >> day >> c
                    >> hour >> c >> minute >> c >> second;

                bdlt::Datetime datetime3(year, month, day,
                                         hour, minute, second);

                bsl::ostringstream oss;
                oss << datetime3;

                // Ignore the millisecond field so don't compare the entire
                // strings.

                ASSERT(0 == oss.str().compare(0, 18, datetime1, 0, 18));

                mX->disableFileLogging();

                ASSERT("" == os.str());
                fileOffset = FsUtil::getFileSize(fileName);
                bsl::cout.rdbuf(coutSbuf);

                mX->disableFileLogging();
            }
            // now swap the two string formats

            if (verbose) cerr << "   .. customized format swapped.\n";
            {
                bsl::string baseName(tempDirGuard.getTempDirName());
                bdls::PathUtil::appendRaw(&baseName, "test7Log");

                ASSERT(0    == mX->enableFileLogging(baseName.c_str(), false));
                ASSERT(true == X->isFileLoggingEnabled());
                ASSERT(1    == mX->enableFileLogging(baseName.c_str(), false));
                ASSERT(true == X->isFileLoggingEnabled());

                fileOffset = FsUtil::getFileSize(fileName);

                bsl::stringstream  os;
                bsl::streambuf    *coutSbuf = bsl::cout.rdbuf();
                bsl::cout.rdbuf(os.rdbuf());

                mX->setLogFormat("%i %p %t %s %f %l %c %m %u",
                                 "%d %p %t %s %f %l %c %m %u");
                ASSERT(true == X->isStdoutLoggingPrefixEnabled());

                BALL_LOG_WARN << "log";

                // look for the file with the constructed name
                glob_t globbuf;
                ASSERT(0 == glob(baseName.c_str(), 0, 0, &globbuf));
                ASSERT(1 == globbuf.gl_pathc);

                // Read the log file to get the record.
                bsl::ifstream fs;
                fs.open(globbuf.gl_pathv[0], bsl::ifstream::in);
                fs.clear();
                globfree(&globbuf);

                ASSERT(fs.is_open());
                bsl::string line;
                ASSERT(getline(fs, line));
                fs.close();

                bsl::string            datetime1, datetime2;
                bsl::string            log1, log2;
                bsl::string::size_type pos;

                // Get datetime and the rest from stdout.
                bsl::string soStr = readPartialFile(fileName, fileOffset);
                pos = soStr.find(' ');
                datetime1 = soStr.substr(0, pos);
                log1 = soStr.substr(pos, soStr.length());

                // Divide line into datetime and the rest.
                pos = line.find(' ');
                datetime2 = line.substr(0, pos);
                log2 = line.substr(pos, line.length()-pos);

                ASSERTV(log1, log2, log1 == log2);

                // Convert datetime2 from ISO to bdlt::Datetime.
                bsl::istringstream iss(datetime2);

                int  year, month, day, hour, minute, second;
                char c;

                iss >> year >> c >> month >> c >> day >> c >> hour
                    >> c >> minute >> c >> second;

                bdlt::Datetime     datetime3(year, month, day, hour,
                                             minute, second);
                bsl::ostringstream oss;

                oss << datetime3;

                // Ignore the millisecond field so don't compare the entire
                // strings.
                ASSERT(0 == oss.str().compare(0, 18, datetime1, 0, 18));

                if (veryVerbose) {
                    bsl::cerr << "datetime3: " << datetime3 << bsl::endl;
                    bsl::cerr << "datetime2: " << datetime2 << bsl::endl;
                    bsl::cerr << "datetime1: " << datetime1 << bsl::endl;
                }

                fileOffset = FsUtil::getFileSize(fileName);
                bsl::cout.rdbuf(coutSbuf);
                mX->disableFileLogging();
            }

            // Deregister here as we used local allocator for the observer.
            ASSERT(0 == manager.deregisterObserver("testObserver"));
        }
#endif

        if (verbose) cerr << "Testing User-Defined Fields Toggling\n";
        {
            Obj mX(ball::Severity::e_WARN, &ta);  const Obj& X = mX;

            const char *logFileFormat;
            const char *stdoutFormat;

            ASSERT(X.isUserFieldsLoggingEnabled());
            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));

            mX.disableUserFieldsLogging();

            ASSERT(false == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);

            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%d %p:%t %s %f:%l %c %m\n"));

            mX.enableUserFieldsLogging();

            ASSERT(X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);

            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));

            // Now change to short format for stdout.
            ASSERT(true == X.isStdoutLoggingPrefixEnabled());

            mX.disableStdoutLoggingPrefix();
            ASSERT(false == X.isStdoutLoggingPrefixEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%s %f:%l %c %m %u\n"));

            mX.disableUserFieldsLogging();
            ASSERT(false == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%s %f:%l %c %m\n"));

            mX.enableUserFieldsLogging();
            ASSERT(true == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%s %f:%l %c %m %u\n"));

            // Change back to long format for stdout.
            ASSERT(false == X.isStdoutLoggingPrefixEnabled());

            mX.enableStdoutLoggingPrefix();
            ASSERT(true  == X.isStdoutLoggingPrefixEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));
            ASSERT(0 == bsl::strcmp(stdoutFormat,
                                    "\n%d %p:%t %s %f:%l %c %m %u\n"));

            // Now see what happens with customized format.  Notice that we
            // intentionally use the default short format.

            const char *newLogFileFormat = "\n%s %f:%l %c %m %u\n";
            const char *newStdoutFormat  = "\n%s %f:%l %c %m %u\n";

            mX.setLogFormat(newLogFileFormat, newStdoutFormat);
            ASSERT(true == X.isStdoutLoggingPrefixEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, newStdoutFormat));

            // Toggling user fields logging should not change the formats.
            mX.disableUserFieldsLogging();
            ASSERT(false == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, newStdoutFormat));

            mX.enableUserFieldsLogging();
            ASSERT(true == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, newStdoutFormat));

            // Now set short format for stdout.
            ASSERT(true  == X.isStdoutLoggingPrefixEnabled());

            mX.disableStdoutLoggingPrefix();
            ASSERT(false == X.isStdoutLoggingPrefixEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, newStdoutFormat));

            // stdoutFormat should change, since even if we are now using
            // customized formats, the format happens to be the same as the
            // default short format.

            mX.disableUserFieldsLogging();
            ASSERT(false == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, "\n%s %f:%l %c %m\n"));

            mX.enableUserFieldsLogging();
            ASSERT(true == X.isUserFieldsLoggingEnabled());

            X.getLogFormat(&logFileFormat, &stdoutFormat);
            ASSERT(0 == bsl::strcmp(logFileFormat, newLogFileFormat));
            ASSERT(0 == bsl::strcmp(stdoutFormat, newStdoutFormat));
        }
        fclose(stdout);
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
// Copyright 2017 Bloomberg Finance L.P.
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
