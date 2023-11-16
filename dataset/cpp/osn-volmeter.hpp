/******************************************************************************
    Copyright (C) 2016-2019 by Streamlabs (General Workings Inc)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once
#include <ipc-server.hpp>
#include <memory>
#include <queue>
#include <array>
#include "obs.h"
#include "utility.hpp"

extern std::mutex mtx;

namespace osn {
class Volmeter {
public:
	class Manager : public utility::generic_object_manager<std::shared_ptr<Volmeter>> {
		friend class std::shared_ptr<Manager>;

	protected:
		Manager() {}
		~Manager() {}

	public:
		Manager(Manager const &) = delete;
		Manager operator=(Manager const &) = delete;

	public:
		static Manager &GetInstance();
	};

private:
	obs_volmeter_t *self;
	uint64_t id;
	size_t callback_count = 0;
	uint64_t *id2 = nullptr;
	uint64_t uid_source = 0;

	struct AudioData {
		std::array<float, MAX_AUDIO_CHANNELS> magnitude{0};
		std::array<float, MAX_AUDIO_CHANNELS> peak{0};
		std::array<float, MAX_AUDIO_CHANNELS> input_peak{0};
		std::chrono::milliseconds lastUpdateTime = std::chrono::milliseconds(0);
		int32_t ch = 0;

		void resetData()
		{
			std::fill(magnitude.begin(), magnitude.end(), -65535.0f);
			std::fill(peak.begin(), peak.end(), -65535.0f);
			std::fill(input_peak.begin(), input_peak.end(), -65535.0f);
			lastUpdateTime = std::chrono::milliseconds(0);
		}
	};

	AudioData current_data;
	std::mutex current_data_mtx;

public:
	Volmeter(obs_fader_type type);
	~Volmeter();

public:
	static void Register(ipc::server &);

	static void ClearVolmeters();
	static void getAudioData(uint64_t id, std::vector<ipc::value> &rval);

	static void Create(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void Destroy(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);

	static void Attach(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void Detach(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void AddCallback(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void RemoveCallback(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);

	static void Query(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void OBSCallback(void *param, const float magnitude[MAX_AUDIO_CHANNELS], const float peak[MAX_AUDIO_CHANNELS],
				const float input_peak[MAX_AUDIO_CHANNELS]);

private:
	static std::chrono::milliseconds GetTime();
	static bool CheckIdle(std::chrono::milliseconds currentTime, std::chrono::milliseconds lastUpdateTime);
};
} // namespace osn