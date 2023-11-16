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

#ifdef ENABLE_GCS
#include <google/cloud/status_or.h>
#include <google/cloud/storage/client.h>
#endif

#include <core23/logger.hpp>
#include <fstream>
#include <io/gcs_filesystem.hpp>
#include <string_view>

namespace HugeCTR {

#ifdef ENABLE_GCS
auto constexpr kUploadBufferSize = 256 * 1024;

namespace gcs = google::cloud::storage;

GCSPath GCSPath::FromString(std::string_view s) {
  // Split URL and check schema.
  auto pos{s.find("://")};
  HCTR_CHECK_HINT(pos != std::string_view::npos,
                  "Not a valid GCS path. Please provide a proper GCS object URI/URL.");
  const std::string_view schema{s.substr(0, pos)};
  HCTR_CHECK_HINT(schema == "gs" || schema == "https",
                  "Path format is incorrect. Please provide a proper GCS URI or URL.");
  s = s.substr(pos + 3);

  // Parse URL.
  pos = s.find("/");

  if (schema == "gs") {
    const std::string bucket{s.substr(0, pos)};
    if (pos == std::string_view::npos) {
      return {bucket};
    }
    s = s.substr(pos + 1);  // s = object

    return {bucket, static_cast<std::string>(s)};
  } else {
    HCTR_CHECK_HINT(pos != std::string::npos, "The URL has no bucket information.");
    const std::string host_name{s.substr(0, pos)};
    s = s.substr(pos + 1);

    pos = s.find("/");

    const std::string bucket{s.substr(0, pos)};
    if (pos == std::string_view::npos) {
      return {bucket, {}, host_name};
    }
    s = s.substr(pos + 1);  // s = object

    return {bucket, static_cast<std::string>(s), host_name};
  }
}

GCSConfigs::GCSConfigs() {
  this->credentials_holder =
      std::make_shared<GCSCredentialsHolder>(google::cloud::MakeGoogleDefaultCredentials());
  this->scheme = "https";
}

void GCSConfigs::set_endpoint_override(const std::string& endpoint) {
  this->endpoint_override = endpoint;
}

GCSConfigs GCSConfigs::Default() {
  GCSConfigs configs;
  return configs;
}

GCSConfigs GCSConfigs::FromUrl(const std::string& url) {
  GCSConfigs configs;
  GCSPath gcs_path = GCSPath::FromString(url);
  if (gcs_path.endpoint_override != "") {
    configs.set_endpoint_override(gcs_path.endpoint_override);
  }
  return configs;
}

GCSConfigs GCSConfigs::FromDataSourceParams(const DataSourceParams& data_source_params) {
  GCSConfigs configs;
  configs.set_endpoint_override(data_source_params.server);
  return configs;
}

GCSConfigs GCSConfigs::FromJSON(const std::string& path) {
  GCSConfigs configs;
  return configs;
}

/**
 * @brief Turn GCSConfigs into Google Cloud Options.
 *
 * @param GCSConfigs
 * @return google::cloud::Options
 */
google::cloud::Options ToGoogleCloudOptions(const GCSConfigs& configs) {
  auto options = google::cloud::Options{};
  std::string scheme = configs.scheme;
  HCTR_CHECK_HINT(!scheme.empty(), "No scheme specified.");
  if (scheme == "https") {
    options.set<google::cloud::UnifiedCredentialsOption>(
        google::cloud::MakeGoogleDefaultCredentials());
  }
  options.set<gcs::UploadBufferSizeOption>(kUploadBufferSize);
  if (!configs.endpoint_override.empty()) {
    options.set<gcs::RestEndpointOption>(scheme + "://" + configs.endpoint_override);
  }
  if (configs.credentials_holder && configs.credentials_holder->credentials) {
    options.set<google::cloud::UnifiedCredentialsOption>(configs.credentials_holder->credentials);
  }
  if (configs.retry_limit_time.has_value()) {
    options.set<gcs::RetryPolicyOption>(
        gcs::LimitedTimeRetryPolicy(
            std::chrono::milliseconds(static_cast<int>(*configs.retry_limit_time * 1000)))
            .clone());
  }
  return options;
}

GCSFileSystem::GCSFileSystem(const GCSConfigs& configs) {
  client_ = std::make_unique<gcs::Client>(ToGoogleCloudOptions(configs));
}

GCSFileSystem::~GCSFileSystem() {}

size_t GCSFileSystem::get_file_size(const std::string& path) const {
  GCSPath gcs_path = GCSPath::FromString(path);
  HCTR_CHECK_HINT(gcs_path.has_bucket_and_object(),
                  "This GCS path does not contain bucket or key information.");
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->GetObjectMetadata(gcs_path.bucket, gcs_path.object);
  HCTR_CHECK_HINT(object_metadata.ok(), "Failed to open the file in GCS.");
  size_t content_length = object_metadata.value().size();
  return content_length;
}

void GCSFileSystem::create_dir(const std::string& path) {
  HCTR_LOG_S(WARNING, WORLD) << "Creating directory in GCS has no effect." << std::endl;
}

void GCSFileSystem::delete_file(const std::string& path) {
  GCSPath gcs_path = GCSPath::FromString(path);
  HCTR_CHECK_HINT(gcs_path.has_bucket_and_object(),
                  "This GCS path does not contain bucket or key information.");
  google::cloud::Status status = client_->DeleteObject(gcs_path.bucket, gcs_path.object);
  HCTR_CHECK_HINT(status.ok(), "Cannot delete file in GCS.");
}

void GCSFileSystem::fetch(const std::string& source_path, const std::string& target_path) {
  GCSPath source_gcs_path = GCSPath::FromString(source_path);
  HCTR_CHECK_HINT(source_gcs_path.has_bucket_and_object(),
                  "The source GCS path does not contain bucket or key information.");
  google::cloud::Status status =
      client_->DownloadToFile(source_gcs_path.bucket, source_gcs_path.object, target_path);
  HCTR_CHECK_HINT(status.ok(), "Failed to download the file from GCS.");
}

void GCSFileSystem::upload(const std::string& source_path, const std::string& target_path) {
  GCSPath target_gcs_path = GCSPath::FromString(target_path);
  HCTR_CHECK_HINT(target_gcs_path.has_bucket_and_object(),
                  "This destination GCS path does not contain bucket or key information.");
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->UploadFile(source_path, target_gcs_path.bucket, target_gcs_path.object,
                          gcs::IfGenerationMatch(0), gcs::NewResumableUploadSession());
  HCTR_CHECK_HINT(object_metadata.ok(), "Failed to upload the file to GCS.");
}

int GCSFileSystem::write(const std::string& path, const void* const data, const size_t data_size,
                         const bool overwrite) {
  GCSPath gcs_path = GCSPath::FromString(path);
  HCTR_CHECK_HINT(gcs_path.has_bucket_and_object(),
                  "This GCS path does not contain bucket or key information.");
  gcs::ObjectWriteStream stream =
      client_->WriteObject(gcs_path.bucket, gcs_path.object, gcs::NewResumableUploadSession(),
                           gcs::AutoFinalizeEnabled());
  stream.write(reinterpret_cast<const char*>(data), data_size);
  stream.flush();
  stream.Close();
  HCTR_CHECK_HINT(
      !stream.IsOpen() && stream.metadata().ok() && stream.metadata().value().size() == data_size,
      "Failed to write to GCS.");
  HCTR_LOG_S(DEBUG, WORLD) << "Successfully write to GCS location:  " << path << std::endl;
  return data_size;
}

int GCSFileSystem::read(const std::string& path, void* const buffer, const size_t buffer_size,
                        const size_t offset) {
  GCSPath gcs_path = GCSPath::FromString(path);
  HCTR_CHECK_HINT(gcs_path.has_bucket_and_object(),
                  "This GCS path does not contain bucket or key information.");
  gcs::ObjectReadStream stream = client_->ReadObject(gcs_path.bucket, gcs_path.object,
                                                     gcs::ReadRange(offset, offset + buffer_size));
  stream.read(reinterpret_cast<char*>(buffer), buffer_size);
  stream.Close();
  HCTR_CHECK_HINT(!stream.IsOpen(), "Failed to read from GCS.");
  return stream.gcount();
}

void GCSFileSystem::copy(const std::string& source_path, const std::string& target_path) {
  GCSPath source_gcs_path = GCSPath::FromString(source_path);
  HCTR_CHECK_HINT(source_gcs_path.has_bucket_and_object(),
                  "The source GCS path does not contain bucket or key information.");
  GCSPath target_gcs_path = GCSPath::FromString(target_path);
  HCTR_CHECK_HINT(target_gcs_path.has_bucket_and_object(),
                  "This destination GCS path does not contain bucket or key information.");
  google::cloud::StatusOr<gcs::ObjectMetadata> copy_meta =
      client_->CopyObject(source_gcs_path.bucket, source_gcs_path.object, target_gcs_path.bucket,
                          target_gcs_path.object);
  HCTR_CHECK_HINT(copy_meta.ok(), "Failed to copy the file in GCS.");
}

void GCSFileSystem::batch_fetch(const std::string& source_path, const std::string& target_path) {
  // TODO
  HCTR_DIE("Not implemented yet!");
}

void GCSFileSystem::batch_upload(const std::string& source_path, const std::string& target_path) {
  // TODO
  HCTR_DIE("Not implemented yet!");
}

#endif

}  // namespace HugeCTR