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

#include "base/net/open_files_enumerator.h"

#include "base/logging.h"
#include "base/strings/unicode.h"

#include <Windows.h>
#include <LM.h>

namespace base {

//--------------------------------------------------------------------------------------------------
OpenFilesEnumerator::OpenFilesEnumerator()
{
    DWORD total_entries = 0;
    DWORD entries_read = 0;

    DWORD error_code = NetFileEnum(nullptr, nullptr, nullptr, 3,
                                   reinterpret_cast<LPBYTE*>(&file_info_),
                                   MAX_PREFERRED_LENGTH,
                                   &entries_read,
                                   &total_entries,
                                   nullptr);
    if (error_code != NERR_Success)
    {
        LOG(LS_ERROR) << "NetShareEnum failed: " << SystemError(error_code).toString();
        return;
    }

    if (!file_info_)
    {
        LOG(LS_ERROR) << "Invalid file info";
        return;
    }

    total_entries_ = total_entries;
}

//--------------------------------------------------------------------------------------------------
OpenFilesEnumerator::~OpenFilesEnumerator()
{
    if (file_info_)
        NetApiBufferFree(file_info_);
}

//--------------------------------------------------------------------------------------------------
bool OpenFilesEnumerator::isAtEnd() const
{
    return current_pos_ >= total_entries_;
}

//--------------------------------------------------------------------------------------------------
void OpenFilesEnumerator::advance()
{
    ++current_pos_;
}

//--------------------------------------------------------------------------------------------------
uint32_t OpenFilesEnumerator::id() const
{
    return file_info_[current_pos_].fi3_id;
}

//--------------------------------------------------------------------------------------------------
std::string OpenFilesEnumerator::userName() const
{
    wchar_t* user_name = file_info_[current_pos_].fi3_username;
    if (!user_name)
        return std::string();

    return utf8FromWide(user_name);
}

//--------------------------------------------------------------------------------------------------
uint32_t OpenFilesEnumerator::lockCount() const
{
    return file_info_[current_pos_].fi3_num_locks;
}

//--------------------------------------------------------------------------------------------------
std::string OpenFilesEnumerator::filePath() const
{
    wchar_t* file_path = file_info_[current_pos_].fi3_pathname;
    if (!file_path)
        return std::string();

    return utf8FromWide(file_path);
}

} // namespace base
