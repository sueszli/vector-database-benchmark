/******************************************************************************
    Copyright (C) 2016-2022 by Streamlabs (General Workings Inc)

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
#include <obs.h>
#include "utility.hpp"
#include "osn-delay.hpp"
#include "osn-reconnect.hpp"
#include "osn-network.hpp"
#include "osn-output-signals.hpp"
#include "osn-video-encoder.hpp"

#include "nodeobs_configManager.hpp"

namespace osn {
class Streaming : public OutputSignals {
public:
	Streaming()
	{
		videoEncoder = nullptr;
		streamArchive = nullptr;
		service = nullptr;
		enforceServiceBitrate = true;
		enableTwitchVOD = false;
		twitchVODSupported = false;
		oldMixer_desktopSource1 = 0;
		oldMixer_desktopSource2 = 0;
		signals = {"start", "stop", "starting", "stopping", "activate", "deactivate", "reconnect", "reconnect_success"};
		delay = new Delay();
		reconnect = new Reconnect();
		network = new Network();
	}
	virtual ~Streaming();

public:
	obs_encoder_t *videoEncoder;
	obs_encoder_t *streamArchive;
	obs_service_t *service;
	bool enforceServiceBitrate;
	bool enableTwitchVOD;
	bool twitchVODSupported;
	uint32_t oldMixer_desktopSource1;
	uint32_t oldMixer_desktopSource2;
	Delay *delay;
	Reconnect *reconnect;
	Network *network;

	bool isTwitchVODSupported();
	void getDelayLegacySettings();
	void getReconnectLegacySettings();
	void getNetworkLegacySettings();
	void setDelayLegacySettings();
	void setReconnectLegacySettings();
	void setNetworkLegacySettings();
};

class IStreaming {
public:
	class Manager : public utility::unique_object_manager<Streaming> {
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

public:
	static void GetService(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetService(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetVideoCanvas(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetVideoCanvas(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetVideoEncoder(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetVideoEncoder(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetEnforceServiceBirate(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetEnforceServiceBirate(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetEnableTwitchVOD(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetEnableTwitchVOD(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetDelay(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetDelay(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetReconnect(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetReconnect(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void GetNetwork(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void SetNetwork(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
	static void Query(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval);
};
}
