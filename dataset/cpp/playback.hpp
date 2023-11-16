#pragma once

#include "lib/spotify/track.hpp"
#include "lib/spotify/device.hpp"
#include "lib/spotify/context.hpp"
#include "lib/enum/repeatstate.hpp"
#include "lib/enum/playeraction.hpp"

#include "thirdparty/json.hpp"

#include <unordered_set>

namespace lib
{
	namespace spt
	{
		class playback
		{
		public:
			playback() = default;

			/**
			 * Metadata for MPRIS
			 */
			auto metadata() const -> nlohmann::json;

			/**
			 * Current volume, fetched from active device
			 */
			auto volume() const -> int;

			/**
			 * Is specified action allowed
			 */
			auto is_allowed(const player_action &action) const -> bool;

			/**
			 * Is anything valid currently playing
			 */
			auto is_valid() const -> bool;

			/**
			 * Context currently playing from
			 */
			lib::spt::context context;

			/**
			 * Repeat mode
			 */
			lib::repeat_state repeat = lib::repeat_state::off;

			/**
			 * Current playing track
			 */
			lib::spt::track item;

			/**
			 * Is currently playing something
			 */
			bool is_playing = false;

			/**
			 * Shuffle tracks
			 */
			bool shuffle = false;

			/**
			 * Progress in current track, in milliseconds
			 */
			int progress_ms = 0;

			/**
			 * Current active device
			 */
			lib::spt::device device;

			/**
			 * Disallowed actions
			 */
			std::unordered_set<lib::player_action> disallowed_actions;
		};

		/**
		 * json -> playback
		 */
		void from_json(const nlohmann::json &j, playback &p);

		/**
		 * playback -> json
		 */
		void to_json(nlohmann::json &j, const playback &p);
	}
}
