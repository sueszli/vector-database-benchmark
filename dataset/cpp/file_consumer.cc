#include "file_consumer.h"

#include "clang_utils.h"
#include "indexer.h"
#include "platform.h"
#include "utils.h"

#include <loguru.hpp>

bool operator==(const CXFileUniqueID& a, const CXFileUniqueID& b) {
  return a.data[0] == b.data[0] && a.data[1] == b.data[1] &&
         a.data[2] == b.data[2];
}

bool FileConsumerSharedState::Mark(const std::string& file) {
  std::lock_guard<std::mutex> lock(mutex);
  return used_files.insert(file).second;
}

void FileConsumerSharedState::Reset(const std::string& file) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = used_files.find(file);
  if (it != used_files.end())
    used_files.erase(it);
}

FileConsumer::FileConsumer(FileConsumerSharedState* shared_state,
                           const AbsolutePath& parse_file)
    : shared_(shared_state), parse_file_(parse_file) {}

IndexFile* FileConsumer::TryConsumeFile(CXFile file,
                                        bool* is_first_ownership) {
  assert(is_first_ownership);

  CXFileUniqueID file_id;
  if (clang_getFileUniqueID(file, &file_id) != 0) {
    EmitError(file);
    return nullptr;
  }

  // Try to find cached local result.
  auto it = local_.find(file_id);
  if (it != local_.end()) {
    *is_first_ownership = false;
    return it->second.get();
  }

  optional<AbsolutePath> file_name = FileName(file);
  if (!file_name) {
    LOG_S(ERROR) << "Could not normalize path "
                 << ToString(clang_getFileName(file));
    return nullptr;
  }

  // No result in local; we need to query global.
  bool did_insert = shared_->Mark(file_name->path);

  // We did not take the file from global. Cache that we failed so we don't try
  // again and return nullptr.
  if (!did_insert) {
    local_[file_id] = nullptr;
    return nullptr;
  }

  // Build IndexFile instance.
  *is_first_ownership = true;
  local_[file_id] = std::make_unique<IndexFile>(file_name->path);
  return local_[file_id].get();
}

std::vector<std::unique_ptr<IndexFile>> FileConsumer::TakeLocalState() {
  std::vector<std::unique_ptr<IndexFile>> result;
  for (auto& entry : local_) {
    if (entry.second)
      result.push_back(std::move(entry.second));
  }
  return result;
}

void FileConsumer::EmitError(CXFile file) const {
  std::string file_name = ToString(clang_getFileName(file));
  // TODO: Investigate this more, why can we get an empty file name?
  if (!file_name.empty()) {
    LOG_S(ERROR) << "Could not get unique file id for " << file_name
                 << " when parsing " << parse_file_;
  }
}