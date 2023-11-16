/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <core23/logger.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <io/io_utils.hpp>
#include <io/local_filesystem.hpp>
#include <iostream>

namespace HugeCTR {

LocalFileSystem::LocalFileSystem() {}

LocalFileSystem::~LocalFileSystem() {}

size_t LocalFileSystem::get_file_size(const std::string& path) const {
  std::ifstream file_stream(path);
  HCTR_CHECK_HINT(file_stream.is_open(), "File not open: ", path);
  file_stream.close();
  return std::filesystem::file_size(path);
}

void LocalFileSystem::create_dir(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    bool success = std::filesystem::create_directories(path);
    HCTR_CHECK_HINT(success, "Failed to create the directory: ", path);
  }
}

void LocalFileSystem::delete_file(const std::string& path) { std::filesystem::remove_all(path); }

void LocalFileSystem::fetch(const std::string& source_path, const std::string& target_path) {
  std::filesystem::copy(source_path, target_path);
}

void LocalFileSystem::upload(const std::string& source_path, const std::string& target_path) {
  std::filesystem::copy(source_path, target_path);
}

int LocalFileSystem::write(const std::string& path, const void* const data, const size_t data_size,
                           const bool overwrite) {
  std::string parent_dir = IOUtils::get_parent_dir(path);
  if (parent_dir != "" && parent_dir != ".") {
    std::filesystem::create_directories(parent_dir);
  }
  if (overwrite) {
    std::ofstream file_stream(path, std::ofstream::binary);
    HCTR_CHECK_HINT(file_stream.is_open(), "File not open for writing: ", path);
    file_stream.write(reinterpret_cast<const char*>(data), data_size);
  } else {
    std::ofstream file_stream(path, std::ofstream::binary | std::ofstream::app);
    HCTR_CHECK_HINT(file_stream.is_open(), "File not open for appending: ", path);
    file_stream.write(reinterpret_cast<const char*>(data), data_size);
  }
  return data_size;
}

int LocalFileSystem::read(const std::string& path, void* const buffer, const size_t buffer_size,
                          const size_t offset) {
  std::ifstream file_stream(path);
  HCTR_CHECK_HINT(file_stream.is_open(), "File not open for reading: ", path);
  file_stream.seekg(offset);
  file_stream.read(reinterpret_cast<char*>(buffer), buffer_size);
  int num_bytes_read;
  if (file_stream) {
    num_bytes_read = buffer_size;
  } else {
    num_bytes_read = file_stream.gcount();
  }
  return num_bytes_read;
}

void LocalFileSystem::copy(const std::string& source_path, const std::string& target_path) {
  std::filesystem::copy(source_path, target_path);
}

void LocalFileSystem::batch_fetch(const std::string& source_path, const std::string& target_path) {
  std::filesystem::copy(source_path, target_path);
}

void LocalFileSystem::batch_upload(const std::string& source_path, const std::string& target_path) {
  std::filesystem::copy(source_path, target_path);
}

}  // namespace HugeCTR
