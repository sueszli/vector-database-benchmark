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
#include <hps/plugin/lookup_manager.hpp>

namespace HierarchicalParameterServer {

std::shared_ptr<LookupManager> LookupManager::Create() {
  return std::shared_ptr<LookupManager>(new LookupManager());
}

LookupManager::LookupManager() : initialized_{false} {}

void LookupManager::init(parameter_server_config& ps_config, pluginType_t plugin_type,
                         int32_t global_batch_size, int32_t num_replicas_in_sync) {
  initialized_ = true;
  auto b2s = [](const char val) { return val ? "True" : "False"; };
  for (auto& inference_params : ps_config.inference_params_array) {
    HCTR_LOG_S(INFO, WORLD) << "HPS plugin uses context stream for model "
                            << inference_params.model_name << ": "
                            << b2s(inference_params.use_context_stream) << std::endl;
  }

  if (!init_check(ps_config, global_batch_size, num_replicas_in_sync, plugin_type)) {
    HCTR_LOG_S(ERROR, WORLD) << "HPS initialization checking failed" << std::endl;
    return;
  }

  // Create the HPS for all models on all the deployed devices
  parameter_server_ = HierParameterServerBase::create(ps_config);

  // Initialie the resources for each model
  for (auto& inference_params : ps_config.inference_params_array) {
    // Create the lookup sessions on all the deployed devices
    std::map<size_t, std::shared_ptr<LookupSessionBase>> lookup_sessions;
    for (const auto& device_id : inference_params.deployed_devices) {
      inference_params.device_id = device_id;
      auto embedding_cache = parameter_server_->get_embedding_cache(inference_params.model_name,
                                                                    inference_params.device_id);
      auto lookup_session = LookupSessionBase::create(inference_params, embedding_cache);
      lookup_sessions.emplace(device_id, lookup_session);
    }
    lookup_session_map_.emplace(inference_params.model_name, lookup_sessions);
  }
}

void LookupManager::forward(const std::string& model_name, int32_t table_id,
                            int32_t global_replica_id, size_t num_keys, size_t emb_vec_size,
                            const void* values_ptr, void* emb_vector_ptr, bool i64_input_tensor,
                            cudaStream_t context_stream) {
  if (!forward_check(model_name, table_id, global_replica_id, num_keys, emb_vec_size,
                     i64_input_tensor)) {
    return;
  }
  auto lookup_session =
      lookup_session_map_.find(model_name)->second.find(global_replica_id)->second;
  auto inference_params = lookup_session->get_inference_params();
  if (inference_params.use_context_stream) {
    lookup_session->lookup_from_device(values_ptr, reinterpret_cast<float*>(emb_vector_ptr),
                                       num_keys, table_id, context_stream);
  } else {
    HCTR_LIB_THROW(cudaStreamSynchronize(context_stream));
    lookup_session->lookup_from_device(values_ptr, reinterpret_cast<float*>(emb_vector_ptr),
                                       num_keys, table_id);
  }
}

bool LookupManager::init_check(parameter_server_config& ps_config, int32_t global_batch_size,
                               const int32_t num_replicas_in_sync, pluginType_t plugin_type) const {
  switch (plugin_type) {
    case TENSORFLOW: {
      if (global_batch_size <= 0) {
        HCTR_LOG_S(ERROR, WORLD) << "global_batch_size must be > 0." << std::endl;
        return false;
      }
      if (num_replicas_in_sync <= 0) {
        HCTR_LOG_S(ERROR, WORLD) << "num_replicas_in_sync must be > 0." << std::endl;
        return false;
      }
      if (global_batch_size % num_replicas_in_sync != 0) {
        HCTR_LOG_S(ERROR, WORLD) << "global_batch_size must be divisible by num_replicas_in_sync."
                                 << std::endl;
        return false;
      }
      size_t local_batch_size = global_batch_size / num_replicas_in_sync;

      for (auto& inference_params : ps_config.inference_params_array) {
        sort(inference_params.deployed_devices.begin(), inference_params.deployed_devices.end());
        auto check = [](const std::vector<int>& vec) {
          for (size_t i{0}; i < vec.size(); ++i) {
            if (vec[i] != i) return false;
          }
          return true;
        };
        if (inference_params.deployed_devices.size() != num_replicas_in_sync) {
          HCTR_LOG_S(ERROR, WORLD)
              << "inference_params.deployed_devices.size() must be equal to num_replicas_in_sync."
              << std::endl;
          return false;
        }
        if (!check(inference_params.deployed_devices)) {
          HCTR_LOG_S(ERROR, WORLD)
              << "inference_params.deployed_devices should contain exactly from 0 "
                 "to num_replicas_in_sync-1."
              << std::endl;
          return false;
        }
        if (local_batch_size > inference_params.max_batchsize) {
          HCTR_LOG_S(ERROR, WORLD) << "global_batch_size / num_replicas_in_sync must be <= "
                                      "max_batchsize configured in ps_config.json."
                                   << std::endl;
          return false;
        }
      }
      break;
    }
    case TENSORRT: {
      for (auto& inference_params : ps_config.inference_params_array) {
        if (inference_params.i64_input_key) {
          HCTR_LOG_S(ERROR, WORLD)
              << "i64_input_key must be false for HPS TensorRT plugin." << std::endl;
          return false;
        }
        if (inference_params.fuse_embedding_table) {
          HCTR_LOG_S(ERROR, WORLD)
              << "fuse_embedding_table must be false for HPS TensorRT plugin." << std::endl;
          return false;
        }
      }
      break;
    }
    case TORCH: {
      // Currently no check is needed for HPS Torch plugin
      break;
    }
    default: {
      assert(!"Error: no such layer && should never get here!");
    }
  }
  return true;
}

bool LookupManager::forward_check(const std::string& model_name, int32_t table_id,
                                  int32_t global_replica_id, size_t num_keys, size_t emb_vec_size,
                                  bool i64_input_tensor) const {
  if (!initialized_) {
    HCTR_LOG_S(ERROR, WORLD) << "HPS must be initialized before execution" << std::endl;
    return false;
  }
  if (lookup_session_map_.find(model_name) == lookup_session_map_.end()) {
    HCTR_LOG_S(ERROR, WORLD) << "Cannot find the model " << model_name << " in HPS" << std::endl;
    return false;
  }
  if (lookup_session_map_.find(model_name)->second.find(global_replica_id) ==
      lookup_session_map_.find(model_name)->second.end()) {
    HCTR_LOG_S(ERROR, WORLD) << "Model " << model_name << " is NOT deployed on the device "
                             << global_replica_id << std::endl;
    return false;
  }

  auto lookup_session =
      lookup_session_map_.find(model_name)->second.find(global_replica_id)->second;
  auto inference_params = lookup_session->get_inference_params();

  size_t original_num_tables{0};
  if (inference_params.fuse_embedding_table) {
    for (auto file_list : inference_params.fused_sparse_model_files) {
      original_num_tables += file_list.size();
    }
  } else {
    original_num_tables = inference_params.sparse_model_files.size();
  }

  if (table_id < 0 || table_id >= original_num_tables) {
    HCTR_LOG_S(ERROR, WORLD) << "table_id for " << model_name << " should be from 0 to "
                             << original_num_tables << "(exclusive)" << std::endl;
    return false;
  }

  size_t temp_table_id = inference_params.fuse_embedding_table
                             ? inference_params.original_table_id_to_fused_table_id_map[table_id]
                             : table_id;
  if (num_keys > inference_params.max_batchsize *
                     inference_params.maxnum_catfeature_query_per_table_per_sample[temp_table_id]) {
    HCTR_LOG_S(ERROR, WORLD)
        << "num_keys must be no larger than "
        << inference_params.max_batchsize *
               inference_params.maxnum_catfeature_query_per_table_per_sample[temp_table_id]
        << std::endl;
    return false;
  }
  if (emb_vec_size != inference_params.embedding_vecsize_per_table[temp_table_id]) {
    HCTR_LOG_S(ERROR, WORLD) << "emb_vec_size must be equal to "
                             << inference_params.embedding_vecsize_per_table[temp_table_id]
                             << std::endl;
    return false;
  }
  if (i64_input_tensor != inference_params.i64_input_key) {
    HCTR_LOG_S(ERROR, WORLD) << "Input tensor dtype should be consistent with HPS configuration"
                             << std::endl;
    return false;
  }
  return true;
}

}  // namespace HierarchicalParameterServer