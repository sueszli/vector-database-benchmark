//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/win/event_enumerator.h"

#include "base/logging.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/unicode.h"
#include "base/win/registry.h"

#include <strsafe.h>

namespace base::win {

namespace {

//--------------------------------------------------------------------------------------------------
void resizeBuffer(ByteArray* buffer, size_t size)
{
    if (buffer->capacity() < size)
        buffer->reserve(size);

    buffer->resize(size);
}

//--------------------------------------------------------------------------------------------------
HANDLE openEventLogHandle(const wchar_t* source, DWORD* records_count, DWORD* first_record)
{
    ScopedEventLog event_log(OpenEventLogW(nullptr, source));
    if (!event_log.isValid())
    {
        PLOG(LS_ERROR) << "OpenEventLogW failed";
        return nullptr;
    }

    if (!GetNumberOfEventLogRecords(event_log.get(), records_count))
    {
        PLOG(LS_ERROR) << "GetNumberOfEventLogRecords failed";
        return nullptr;
    }

    if (!GetOldestEventLogRecord(event_log.get(), first_record))
    {
        PLOG(LS_ERROR) << "GetOldestEventLogRecord failed";
        return nullptr;
    }

    return event_log.release();
}

//--------------------------------------------------------------------------------------------------
bool eventLogRecord(HANDLE event_log, DWORD record_offset, ByteArray* record_buffer)
{
    resizeBuffer(record_buffer, sizeof(EVENTLOGRECORD));

    EVENTLOGRECORD* record = reinterpret_cast<EVENTLOGRECORD*>(record_buffer->data());

    DWORD bytes_read = 0;
    DWORD bytes_needed = 0;

    if (!ReadEventLogW(event_log,
                       EVENTLOG_SEEK_READ | EVENTLOG_BACKWARDS_READ,
                       record_offset,
                       record,
                       sizeof(EVENTLOGRECORD),
                       &bytes_read,
                       &bytes_needed))
    {
        DWORD error_code = GetLastError();

        if (error_code != ERROR_INSUFFICIENT_BUFFER)
        {
            LOG(LS_ERROR) << "ReadEventLogW failed: " << SystemError(error_code).toString();
            return false;
        }

        resizeBuffer(record_buffer, bytes_needed);
        record = reinterpret_cast<EVENTLOGRECORD*>(record_buffer->data());

        if (!ReadEventLogW(event_log,
                           EVENTLOG_SEEK_READ | EVENTLOG_BACKWARDS_READ,
                           record_offset,
                           record,
                           bytes_needed,
                           &bytes_read,
                           &bytes_needed))
        {
            PLOG(LS_ERROR) << "ReadEventLogW failed";
            return false;
        }
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
bool eventLogMessageFileDLL(
    const wchar_t* log_name, const wchar_t* source, std::wstring* message_file)
{
    wchar_t key_path[MAX_PATH];

    HRESULT hr = StringCbPrintfW(key_path, sizeof(key_path),
                                 L"SYSTEM\\CurrentControlSet\\Services\\EventLog\\%s\\%s",
                                 log_name, source);
    if (FAILED(hr))
    {
        LOG(LS_ERROR) << "StringCbPrintfW failed: "
                      << SystemError(static_cast<DWORD>(hr)).toString();
        return false;
    }

    RegistryKey key;

    LONG status = key.open(HKEY_LOCAL_MACHINE, key_path, KEY_READ);
    if (status != ERROR_SUCCESS)
    {
        LOG(LS_ERROR) << "key.open failed: "
                      << SystemError(static_cast<DWORD>(status)).toString();
        return false;
    }

    status = key.readValue(L"EventMessageFile", message_file);
    if (status != ERROR_SUCCESS)
    {
        LOG(LS_INFO) << "key.readValue failed: "
                     << SystemError(static_cast<DWORD>(status)).toString();
        return false;
    }

    return true;
}

//--------------------------------------------------------------------------------------------------
wchar_t* loadMessageFromDLL(const wchar_t* module_name, DWORD event_id, wchar_t** arguments)
{
    HINSTANCE module = LoadLibraryExW(module_name,
                                      nullptr,
                                      DONT_RESOLVE_DLL_REFERENCES | LOAD_LIBRARY_AS_DATAFILE);
    if (!module)
        return nullptr;

    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE |
                  FORMAT_MESSAGE_ARGUMENT_ARRAY | FORMAT_MESSAGE_MAX_WIDTH_MASK;

    wchar_t* message_buffer = nullptr;
    DWORD length = 0;

    __try
    {
        // SEH to protect from invalid string parameters.
        __try
        {
            length = FormatMessageW(flags,
                                    module,
                                    event_id,
                                    MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL),
                                    reinterpret_cast<LPWSTR>(&message_buffer),
                                    0,
                                    reinterpret_cast<va_list*>(arguments));
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            flags &= ~FORMAT_MESSAGE_ARGUMENT_ARRAY;
            flags |= FORMAT_MESSAGE_IGNORE_INSERTS;

            length = FormatMessageW(flags,
                                    module,
                                    event_id,
                                    MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL),
                                    reinterpret_cast<LPWSTR>(&message_buffer),
                                    0,
                                    nullptr);
        }
    }
    __finally
    {
        FreeLibrary(module);
    }

    if (!length)
        return nullptr;

    return message_buffer;
}

//--------------------------------------------------------------------------------------------------
bool eventLogMessage(const wchar_t* log_name, EVENTLOGRECORD* record, std::wstring* message)
{
    wchar_t* source = reinterpret_cast<wchar_t*>(record + 1);

    std::wstring message_file;

    if (!eventLogMessageFileDLL(log_name, source, &message_file))
        return false;

    wchar_t* argument = reinterpret_cast<wchar_t*>(
        reinterpret_cast<LPBYTE>(record) + record->StringOffset);

    static const WORD kMaxInsertStrings = 100;

    wchar_t* arguments[kMaxInsertStrings];
    memset(arguments, 0, sizeof(arguments));

    WORD num_strings = std::min(record->NumStrings, kMaxInsertStrings);
    for (WORD i = 0; i < num_strings; ++i)
    {
        arguments[i] = argument;
        argument += lstrlenW(argument) + 1;
    }

    wchar_t* file = &message_file[0];

    while (file)
    {
        wchar_t* next_file = wcschr(file, L';');
        if (next_file != nullptr)
        {
            *next_file = 0;
            ++next_file;
        }

        wchar_t module_name[MAX_PATH];
        if (ExpandEnvironmentStringsW(file, module_name, _countof(module_name)) != 0)
        {
            wchar_t* message_buffer =
                loadMessageFromDLL(module_name, record->EventID, arguments);

            if (message_buffer)
            {
                message->assign(message_buffer);
                LocalFree(message_buffer);
                return true;
            }
        }

        file = next_file;
    }

    if (num_strings > 0)
    {
        argument = reinterpret_cast<wchar_t*>(
           reinterpret_cast<LPBYTE>(record) + record->StringOffset);

        for (int i = 0; i < num_strings; ++i)
        {
            message->append(argument);

            if (i != num_strings - 1)
                message->append(L"; ");

            argument += lstrlenW(argument) + 1;
        }

        return true;
    }

    return false;
}

} // namespace

//--------------------------------------------------------------------------------------------------
EventEnumerator::EventEnumerator(std::wstring_view log_name, uint32_t start, uint32_t count)
    : log_name_(log_name)
{
    if (!count)
        return;

    DWORD first_record = 0;
    DWORD records_count = 0;

    event_log_.reset(openEventLogHandle(log_name_.c_str(), &records_count, &first_record));
    if (!records_count)
        return;

    records_count_ = records_count;

    int end = static_cast<int>(first_record + records_count);

    current_pos_ = end - static_cast<int>(start) - 1;
    if (current_pos_ < static_cast<int>(first_record))
        current_pos_ = static_cast<int>(first_record);
    else if (current_pos_ > end - 1)
        current_pos_ = end - 1;

    end_record_ = current_pos_ - static_cast<int>(count) + 1;
    if (end_record_ < static_cast<int>(first_record))
        end_record_ = static_cast<int>(first_record);

    LOG(LS_INFO) << "Log name: " << log_name_;
    LOG(LS_INFO) << "First: " << first_record << " count: " << records_count
                 << " pos: " << current_pos_ << " end: " << end_record_;
}

//--------------------------------------------------------------------------------------------------
EventEnumerator::~EventEnumerator() = default;

//--------------------------------------------------------------------------------------------------
uint32_t EventEnumerator::count() const
{
    return records_count_;
}

//--------------------------------------------------------------------------------------------------
bool EventEnumerator::isAtEnd() const
{
    if (!event_log_.isValid())
        return true;

    for (;;)
    {
        if (current_pos_ < end_record_)
            return true;

        if (eventLogRecord(event_log_.get(), static_cast<uint32_t>(current_pos_), &record_buffer_))
            return false;

        record_buffer_.clear();
        --current_pos_;
    }
}

//--------------------------------------------------------------------------------------------------
void EventEnumerator::advance()
{
    record_buffer_.clear();
    --current_pos_;
}

//--------------------------------------------------------------------------------------------------
EventEnumerator::Type EventEnumerator::type() const
{
    switch (record()->EventType)
    {
        case EVENTLOG_ERROR_TYPE:
            return Type::ERR;

        case EVENTLOG_WARNING_TYPE:
            return Type::WARN;

        case EVENTLOG_INFORMATION_TYPE:
            return Type::INFO;

        case EVENTLOG_AUDIT_SUCCESS:
            return Type::AUDIT_SUCCESS;

        case EVENTLOG_AUDIT_FAILURE:
            return Type::AUDIT_FAILURE;

        case EVENTLOG_SUCCESS:
            return Type::SUCCESS;

        default:
            return Type::UNKNOWN;
    }
}

//--------------------------------------------------------------------------------------------------
int64_t EventEnumerator::time() const
{
    return record()->TimeGenerated;
}

//--------------------------------------------------------------------------------------------------
std::string EventEnumerator::category() const
{
    return numberToString(record()->EventCategory);
}

//--------------------------------------------------------------------------------------------------
uint32_t EventEnumerator::eventId() const
{
    return record()->EventID & 0xFFFF;
}

//--------------------------------------------------------------------------------------------------
std::string EventEnumerator::source() const
{
    return utf8FromWide(reinterpret_cast<wchar_t*>(record() + 1));
}

//--------------------------------------------------------------------------------------------------
std::string EventEnumerator::description() const
{
    std::wstring desc_wide;
    if (!eventLogMessage(log_name_.c_str(), record(), &desc_wide))
        return std::string();

    return utf8FromWide(desc_wide);
}

//--------------------------------------------------------------------------------------------------
EVENTLOGRECORD* EventEnumerator::record() const
{
    return reinterpret_cast<EVENTLOGRECORD*>(record_buffer_.data());
}

} // namespace base::win
