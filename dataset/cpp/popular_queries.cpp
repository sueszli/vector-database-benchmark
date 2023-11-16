#include "popular_queries.h"
#include "logger.h"
#include <algorithm>
#include <mutex>
#include "string_utils.h"

PopularQueries::PopularQueries(size_t k) : k(k), max_size(k * 2) {

}

void PopularQueries::add(const std::string& key, const bool live_query, const std::string& user_id, uint64_t now_ts_us) {
    if(live_query) {
        // live query must be aggregated first to their final form as they could be prefix queries
        if(now_ts_us == 0) {
            now_ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();
        }

        if(!umutex.try_lock()) {
            // instead of locking we just skip incrementing keys during consolidation time
            return ;
        }

        auto& queries = user_prefix_queries[user_id];
        if(queries.size() < 100) {
            queries.emplace_back(key, now_ts_us);
        }

        umutex.unlock();

    } else {
        if(!lmutex.try_lock()) {
            // instead of locking we just skip incrementing keys during consolidation time
            return ;
        }

        auto it = local_counts.find(key);

        if(it != local_counts.end()) {
            it.value()++;
        } else if(local_counts.size() < max_size) {
            // skip count when map has become too large (to prevent abuse)
            local_counts.emplace(key, 1);
        }

        lmutex.unlock();
    }
}

void PopularQueries::serialize_as_docs(std::string& docs) {
    std::shared_lock lk(lmutex);

    std::string key_buffer;
    for(auto it = local_counts.begin(); it != local_counts.end(); ++it) {
        it.key(key_buffer);
        nlohmann::json doc;
        doc["id"] = std::to_string(StringUtils::hash_wy(key_buffer.c_str(), key_buffer.size()));
        doc["q"] = key_buffer;
        doc["$operations"]["increment"]["count"] = it.value();
        docs += doc.dump() + "\n";
    }

    if(!docs.empty()) {
        docs.pop_back();
    }
}

void PopularQueries::reset_local_counts() {
    std::unique_lock lk(lmutex);
    local_counts.clear();
}

size_t PopularQueries::get_k() {
    return k;
}

void PopularQueries::compact_user_queries(uint64_t now_ts_us) {
    std::unique_lock lk(umutex);
    std::vector<std::string> keys_to_delete;

    for(auto& kv: user_prefix_queries) {
        auto& queries = kv.second;
        int64_t last_consolidated_index = -1;
        for(uint32_t i = 0; i < queries.size(); i++) {
            if(now_ts_us - queries[i].timestamp < QUERY_FINALIZATION_INTERVAL_MICROS) {
                break;
            }

            uint64_t diff_micros = (i == queries.size()-1) ? (now_ts_us - queries[i].timestamp) :
                                   (queries[i + 1].timestamp - queries[i].timestamp);

            if(diff_micros > QUERY_FINALIZATION_INTERVAL_MICROS) {
                add(queries[i].query, false, "");
                last_consolidated_index = i;
            }
        }

        queries.erase(queries.begin(), queries.begin() + last_consolidated_index+1);

        if(queries.empty()) {
            keys_to_delete.push_back(kv.first);
        }
    }

    for(auto& key: keys_to_delete) {
        user_prefix_queries.erase(key);
    }
}

std::unordered_map<std::string, std::vector<PopularQueries::QWithTimestamp>> PopularQueries::get_user_prefix_queries() {
    std::unique_lock lk(umutex);
    return user_prefix_queries;
}

tsl::htrie_map<char, uint32_t> PopularQueries::get_local_counts() {
    std::unique_lock lk(lmutex);
    return local_counts;
}
