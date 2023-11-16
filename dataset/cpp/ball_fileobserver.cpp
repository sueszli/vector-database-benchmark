// ball_fileobserver.cpp                                              -*-C++-*-
#include <ball_fileobserver.h>

#include <bsls_ident.h>
BSLS_IDENT_RCSID(ball_fileobserver_cpp,"$Id$ $CSID$")

#include <ball_context.h>
#include <ball_log.h>                         // for testing only
#include <ball_loggermanager.h>               // for testing only
#include <ball_loggermanagerconfiguration.h>  // for testing only
#include <ball_record.h>
#include <ball_streamobserver.h>              // for testing only

#include <bslmt_lockguard.h>

#include <bsl_cstdio.h>
#include <bsl_cstring.h>                      // for 'bsl::strcmp'
#include <bsl_sstream.h>

namespace BloombergLP {
namespace ball {

namespace {

static const char *const k_DEFAULT_LONG_FORMAT =
                                         "\n%d %p:%t %s %f:%l %c %m %u\n";

static const char *const k_DEFAULT_LONG_FORMAT_WITHOUT_USERFIELDS =
                                         "\n%d %p:%t %s %f:%l %c %m\n";

static const char *const k_DEFAULT_SHORT_FORMAT =
                                         "\n%s %f:%l %c %m %u\n";

static const char *const k_DEFAULT_SHORT_FORMAT_WITHOUT_USERFIELDS =
                                         "\n%s %f:%l %c %m\n";

}  // close unnamed namespace

                          // ------------------
                          // class FileObserver
                          // ------------------

// CREATORS
FileObserver::FileObserver()
: d_logFileFormatter(k_DEFAULT_LONG_FORMAT, bdlt::DatetimeInterval(0))
, d_stdoutFormatter(k_DEFAULT_LONG_FORMAT, bdlt::DatetimeInterval(0))
, d_stdoutThreshold(Severity::e_WARN)
, d_useRegularFormatOnStdoutFlag(true)
, d_publishInLocalTime(false)
, d_userFieldsLoggingFlag(true)
, d_stdoutLongFormat(k_DEFAULT_LONG_FORMAT)
, d_stdoutShortFormat(k_DEFAULT_SHORT_FORMAT)
, d_fileObserver2()
{
}

FileObserver::FileObserver(bslma::Allocator *basicAllocator)
: d_logFileFormatter(k_DEFAULT_LONG_FORMAT,
                     bdlt::DatetimeInterval(0),
                     basicAllocator)
, d_stdoutFormatter(k_DEFAULT_LONG_FORMAT,
                    bdlt::DatetimeInterval(0),
                    basicAllocator)
, d_stdoutThreshold(Severity::e_WARN)
, d_useRegularFormatOnStdoutFlag(true)
, d_publishInLocalTime(false)
, d_userFieldsLoggingFlag(true)
, d_stdoutLongFormat(k_DEFAULT_LONG_FORMAT, basicAllocator)
, d_stdoutShortFormat(k_DEFAULT_SHORT_FORMAT, basicAllocator)
, d_fileObserver2(basicAllocator)
{
}

FileObserver::FileObserver(Severity::Level   stdoutThreshold,
                           bslma::Allocator *basicAllocator)
: d_logFileFormatter(k_DEFAULT_LONG_FORMAT,
                     bdlt::DatetimeInterval(0),
                     basicAllocator)
, d_stdoutFormatter(k_DEFAULT_LONG_FORMAT,
                    bdlt::DatetimeInterval(0),
                    basicAllocator)
, d_stdoutThreshold(stdoutThreshold)
, d_useRegularFormatOnStdoutFlag(true)
, d_publishInLocalTime(false)
, d_userFieldsLoggingFlag(true)
, d_stdoutLongFormat(k_DEFAULT_LONG_FORMAT, basicAllocator)
, d_stdoutShortFormat(k_DEFAULT_SHORT_FORMAT, basicAllocator)
, d_fileObserver2(basicAllocator)
{
}

FileObserver::FileObserver(Severity::Level   stdoutThreshold,
                           bool              publishInLocalTime,
                           bslma::Allocator *basicAllocator)
: d_logFileFormatter(k_DEFAULT_LONG_FORMAT, publishInLocalTime, basicAllocator)
, d_stdoutFormatter(k_DEFAULT_LONG_FORMAT, publishInLocalTime, basicAllocator)
, d_stdoutThreshold(stdoutThreshold)
, d_useRegularFormatOnStdoutFlag(true)
, d_publishInLocalTime(publishInLocalTime)
, d_userFieldsLoggingFlag(true)
, d_stdoutLongFormat(k_DEFAULT_LONG_FORMAT, basicAllocator)
, d_stdoutShortFormat(k_DEFAULT_SHORT_FORMAT, basicAllocator)
, d_fileObserver2(basicAllocator)
{
    if (d_publishInLocalTime) {
        d_fileObserver2.enablePublishInLocalTime();
    }
}

FileObserver::~FileObserver()
{
}

// MANIPULATORS
void FileObserver::disableStdoutLoggingPrefix()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    if (false == d_useRegularFormatOnStdoutFlag) {
        return;                                                       // RETURN
    }

    d_useRegularFormatOnStdoutFlag = false;
    d_stdoutFormatter.setFormat(d_stdoutShortFormat.c_str());
}

void FileObserver::disableUserFieldsLogging()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    if (false == d_userFieldsLoggingFlag) {
        return;                                                       // RETURN
    }

    d_userFieldsLoggingFlag = false;

    if (0 == bsl::strcmp(
                    d_stdoutFormatter.format(),
                    d_useRegularFormatOnStdoutFlag ? k_DEFAULT_LONG_FORMAT
                                                   : k_DEFAULT_SHORT_FORMAT)) {
        d_stdoutFormatter.setFormat(
                              d_useRegularFormatOnStdoutFlag
                                  ? k_DEFAULT_LONG_FORMAT_WITHOUT_USERFIELDS
                                  : k_DEFAULT_SHORT_FORMAT_WITHOUT_USERFIELDS);
    }

    if (0 == bsl::strcmp(d_logFileFormatter.format(), k_DEFAULT_LONG_FORMAT)) {
        d_logFileFormatter.setFormat(k_DEFAULT_LONG_FORMAT_WITHOUT_USERFIELDS);
        d_fileObserver2.setLogFileFunctor(d_logFileFormatter);
    }
}

void FileObserver::disablePublishInLocalTime()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    d_publishInLocalTime = false;
    d_stdoutFormatter.disablePublishInLocalTime();
    d_logFileFormatter.disablePublishInLocalTime();
    d_fileObserver2.disablePublishInLocalTime();

    // Unfortunately, this is necessary because 'd_fileObserver2' has a *copy*
    // of the log file formatter.

    d_fileObserver2.setLogFileFunctor(d_logFileFormatter);
}

void FileObserver::enableStdoutLoggingPrefix()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    if (true == d_useRegularFormatOnStdoutFlag) {
        return;                                                       // RETURN
    }

    d_useRegularFormatOnStdoutFlag = true;
    d_stdoutFormatter.setFormat(d_stdoutLongFormat.c_str());
}

void FileObserver::enableUserFieldsLogging()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    if (true == d_userFieldsLoggingFlag) {
        return;                                                       // RETURN
    }

    d_userFieldsLoggingFlag = true;

    if (0 == bsl::strcmp(d_stdoutFormatter.format(),
                         d_useRegularFormatOnStdoutFlag
                             ? k_DEFAULT_LONG_FORMAT_WITHOUT_USERFIELDS
                             : k_DEFAULT_SHORT_FORMAT_WITHOUT_USERFIELDS)) {
        d_stdoutFormatter.setFormat(
                              d_useRegularFormatOnStdoutFlag
                                  ? k_DEFAULT_LONG_FORMAT
                                  : k_DEFAULT_SHORT_FORMAT);
    }

    if (0 == bsl::strcmp(d_logFileFormatter.format(),
                         k_DEFAULT_LONG_FORMAT_WITHOUT_USERFIELDS)) {
        d_logFileFormatter.setFormat(k_DEFAULT_LONG_FORMAT);
        d_fileObserver2.setLogFileFunctor(d_logFileFormatter);
    }
}

void FileObserver::enablePublishInLocalTime()
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    d_publishInLocalTime = true;
    d_stdoutFormatter.enablePublishInLocalTime();
    d_logFileFormatter.enablePublishInLocalTime();
    d_fileObserver2.enablePublishInLocalTime();

    // Unfortunately, this is necessary because 'd_fileObserver2' has a *copy*
    // of the log file formatter.

    d_fileObserver2.setLogFileFunctor(d_logFileFormatter);
}

void FileObserver::publish(const Record& record, const Context& context)
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    if (record.fixedFields().severity() <= d_stdoutThreshold) {
        bsl::ostringstream oss;
        d_stdoutFormatter(oss, record);

        // Use 'fwrite' to specify the length to write.

        bsl::fwrite(oss.str().c_str(), 1, oss.str().length(), stdout);
        bsl::fflush(stdout);
    }

    d_fileObserver2.publish(record, context);
}

void FileObserver::setLogFormat(const char *logFileFormat,
                                const char *stdoutFormat)
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    d_logFileFormatter.setFormat(logFileFormat);
    d_fileObserver2.setLogFileFunctor(d_logFileFormatter);

    d_stdoutLongFormat = stdoutFormat;
    d_stdoutFormatter.setFormat(stdoutFormat);
    d_useRegularFormatOnStdoutFlag = true;  // enable prefix
}

void FileObserver::setStdoutThreshold(Severity::Level stdoutThreshold)
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    d_stdoutThreshold = stdoutThreshold;
}

// ACCESSORS
bslma::Allocator *FileObserver::allocator() const
{
    return d_stdoutLongFormat.allocator().mechanism();
}

void FileObserver::getLogFormat(const char **logFileFormat,
                                const char **stdoutFormat) const
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    *logFileFormat = d_logFileFormatter.format();
    *stdoutFormat  = d_stdoutFormatter.format();
}

bool FileObserver::isPublishInLocalTimeEnabled() const
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    return d_publishInLocalTime;
}

bool FileObserver::isStdoutLoggingPrefixEnabled() const
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    return d_useRegularFormatOnStdoutFlag;
}

bool FileObserver::isUserFieldsLoggingEnabled() const
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    return d_userFieldsLoggingFlag;
}

Severity::Level FileObserver::stdoutThreshold() const
{
    bslmt::LockGuard<bslmt::Mutex> guard(&d_mutex);

    return d_stdoutThreshold;
}

}  // close package namespace
}  // close enterprise namespace

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
