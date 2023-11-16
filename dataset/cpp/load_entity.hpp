#pragma once

// entt
#include <entt/entity/fwd.hpp>

// nlohmann
#include <nlohmann/json.hpp>

namespace kengine::meta::json {
	KENGINE_META_JSON_EXPORT void load_entity(const nlohmann::json & entity_json, entt::handle e) noexcept;
}