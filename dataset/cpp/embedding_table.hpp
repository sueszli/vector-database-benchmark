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
#pragma once

#include <embedding/embedding_table.hpp>
#include <embedding_storage/common.hpp>
#include <resource_manager.hpp>

namespace embedding {

// per gpu object
class IGroupedEmbeddingTable : public ILookup {
 public:
  virtual ~IGroupedEmbeddingTable() = default;

  virtual void update(const core23::Tensor &unique_keys, const core23::Tensor &num_unique_keys,
                      const core23::Tensor &table_ids, const core23::Tensor &ev_start_indices,
                      const core23::Tensor &wgrad) = 0;

  virtual void assign(const core23::Tensor &unique_key, size_t num_unique_key,
                      const core23::Tensor &num_unique_key_per_table_offset,
                      size_t num_table_offset, const core23::Tensor &table_id_list,
                      core23::Tensor &embeding_vector,
                      const core23::Tensor &embedding_vector_offset) = 0;

  virtual void load(core23::Tensor &keys, core23::Tensor &id_space_offset,
                    core23::Tensor &embedding_table, core23::Tensor &ev_size_list,
                    core23::Tensor &id_space) = 0;

  virtual void dump(core23::Tensor *keys, core23::Tensor *id_space_offset,
                    core23::Tensor *embedding_table, core23::Tensor *ev_size_list,
                    core23::Tensor *id_space) = 0;

  virtual void dump_by_id(core23::Tensor *h_keys_tensor, core23::Tensor *h_embedding_table,
                          int table_id) = 0;

  virtual void load_by_id(core23::Tensor *h_keys_tensor, core23::Tensor *h_embedding_table,
                          int table_id) = 0;

  virtual size_t size() const = 0;

  virtual size_t capacity() const = 0;

  virtual size_t key_num() const = 0;

  virtual std::vector<size_t> size_per_table() const = 0;

  virtual std::vector<size_t> capacity_per_table() const = 0;

  virtual std::vector<size_t> key_num_per_table() const = 0;

  virtual std::vector<int> table_ids() const = 0;

  virtual std::vector<int> table_evsize() const = 0;

  virtual void clear() = 0;

  virtual void set_learning_rate(float lr) = 0;
};

class IDynamicEmbeddingTable : public IGroupedEmbeddingTable {
  virtual void evict(const core23::Tensor &keys, size_t num_keys,
                     const core23::Tensor &id_space_offset, size_t num_id_space_offset,
                     const core23::Tensor &id_space_list) = 0;
};

std::vector<std::unique_ptr<IGroupedEmbeddingTable>> create_grouped_embedding_tables(
    std::shared_ptr<HugeCTR::ResourceManager> resource_manager,
    std::shared_ptr<CoreResourceManager> core,
    const EmbeddingCollectionParam &embedding_collection_param,
    const std::vector<EmbeddingTableParam> &emb_table_param_list);

}  // namespace embedding
