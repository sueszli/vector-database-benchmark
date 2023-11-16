// Copyright (c) 2023 by Apex.AI Inc. All rights reserved.
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
//
// SPDX-License-Identifier: Apache-2.0

#ifndef IOX_HOOFS_POSIX_VOCABULARY_FILE_NAME_HPP
#define IOX_HOOFS_POSIX_VOCABULARY_FILE_NAME_HPP

#include "iox/semantic_string.hpp"

namespace iox
{
namespace details
{
bool file_name_does_contain_invalid_characters(const string<platform::IOX_MAX_FILENAME_LENGTH>& value) noexcept;
bool file_name_does_contain_invalid_content(const string<platform::IOX_MAX_FILENAME_LENGTH>& value) noexcept;
} // namespace details

/// @brief Represents a single file name. It is not allowed to contain any path elements
///        like "./some_file" or "path/to/file". Just a plain old simple "my_file.bla".
class FileName : public SemanticString<FileName,
                                       platform::IOX_MAX_FILENAME_LENGTH,
                                       details::file_name_does_contain_invalid_content,
                                       details::file_name_does_contain_invalid_characters>
{
    using Parent = SemanticString<FileName,
                                  platform::IOX_MAX_FILENAME_LENGTH,
                                  details::file_name_does_contain_invalid_content,
                                  details::file_name_does_contain_invalid_characters>;
    using Parent::Parent;
};
} // namespace iox

#endif
