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

#include "osn-streaming.hpp"
#include "osn-service.hpp"
#include "osn-error.hpp"
#include "shared.hpp"
#include <osn-video.hpp>

osn::Streaming::~Streaming()
{
	deleteOutput();
	if (streamArchive && !obs_encoder_active(streamArchive)) {
		obs_encoder_release(streamArchive);
		streamArchive = nullptr;
	}
}

void osn::IStreaming::GetService(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Service reference is not valid.");
	}

	uint64_t uid = osn::Service::Manager::GetInstance().find(streaming->service);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetService(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	obs_service_t *service = osn::Service::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!service) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Service reference is not valid.");
	}

	streaming->service = service;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetVideoCanvas(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	uint64_t uid = osn::Video::Manager::GetInstance().find(streaming->canvas);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetVideoCanvas(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	obs_video_info *canvas = osn::Video::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!canvas) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Canvas reference is not valid.");
	}

	streaming->canvas = canvas;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetVideoEncoder(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	uint64_t uid = osn::VideoEncoder::Manager::GetInstance().find(streaming->videoEncoder);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetVideoEncoder(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	obs_encoder_t *encoder = osn::VideoEncoder::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!encoder) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Encoder reference is not valid.");
	}

	streaming->videoEncoder = encoder;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetEnforceServiceBirate(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->enforceServiceBitrate));
	AUTO_DEBUG;
}

void osn::IStreaming::SetEnforceServiceBirate(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	streaming->enforceServiceBitrate = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetEnableTwitchVOD(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->enableTwitchVOD));
	AUTO_DEBUG;
}

void osn::IStreaming::SetEnableTwitchVOD(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	streaming->enableTwitchVOD = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetDelay(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	uint64_t uid = osn::IDelay::Manager::GetInstance().find(streaming->delay);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetDelay(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	Delay *delay = osn::IDelay::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!delay) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Encoder reference is not valid.");
	}

	streaming->delay = delay;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetReconnect(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	uint64_t uid = osn::IReconnect::Manager::GetInstance().find(streaming->reconnect);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetReconnect(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	Reconnect *reconnect = osn::IReconnect::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!reconnect) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Reconnect reference is not valid.");
	}

	streaming->reconnect = reconnect;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IStreaming::GetNetwork(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	uint64_t uid = osn::INetwork::Manager::GetInstance().find(streaming->network);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IStreaming::SetNetwork(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	Network *network = osn::INetwork::Manager::GetInstance().find(args[1].value_union.ui64);
	if (!network) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Network reference is not valid.");
	}

	streaming->network = network;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

bool osn::Streaming::isTwitchVODSupported()
{
	if (!service)
		return false;

	obs_data_t *settings = obs_service_get_settings(service);
	const char *serviceName = obs_data_get_string(settings, "service");
	obs_data_release(settings);

	if (serviceName && strcmp(serviceName, "Twitch") != 0)
		return false;

	bool sourceExists = false;
	obs_enum_sources(
		[](void *param, obs_source_t *source) {
			auto id = obs_source_get_id(source);
			if (strcmp(id, "soundtrack_source") == 0) {
				*reinterpret_cast<bool *>(param) = true;
				return false;
			}
			return true;
		},
		&sourceExists);

	if (!sourceExists)
		return false;
}

void osn::IStreaming::Query(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	std::unique_lock<std::mutex> ulock(streaming->signalsMtx);
	if (streaming->signalsReceived.empty()) {
		rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
		AUTO_DEBUG;
		return;
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));

	auto signal = streaming->signalsReceived.front();
	rval.push_back(ipc::value("streaming"));
	rval.push_back(ipc::value(signal.signal));
	rval.push_back(ipc::value(signal.code));
	rval.push_back(ipc::value(signal.errorMessage));

	streaming->signalsReceived.pop();

	AUTO_DEBUG;
}

osn::IStreaming::Manager &osn::IStreaming::Manager::GetInstance()
{
	static osn::IStreaming::Manager _inst;
	return _inst;
}

void osn::Streaming::getDelayLegacySettings()
{
	delay = new Delay();
	delay->enabled = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "DelayEnable");
	delay->delaySec = config_get_int(ConfigManager::getInstance().getBasic(), "Output", "DelaySec");
	delay->preserveDelay = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "DelayPreserve");
	osn::IDelay::Manager::GetInstance().allocate(delay);
}

void osn::Streaming::getReconnectLegacySettings()
{
	reconnect = new Reconnect();
	reconnect->enabled = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "Reconnect");
	reconnect->retryDelay = config_get_uint(ConfigManager::getInstance().getBasic(), "Output", "RetryDelay");
	reconnect->maxRetries = config_get_uint(ConfigManager::getInstance().getBasic(), "Output", "MaxRetries");
	osn::IReconnect::Manager::GetInstance().allocate(reconnect);
}

void osn::Streaming::getNetworkLegacySettings()
{
	network = new Network();
	network->bindIP = config_get_string(ConfigManager::getInstance().getBasic(), "Output", "BindIP");
	network->enableDynamicBitrate = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "DynamicBitrate");
	network->enableOptimizations = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "NewSocketLoopEnable");
	network->enableLowLatency = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "LowLatencyEnable");
	osn::INetwork::Manager::GetInstance().allocate(network);
}

void osn::Streaming::setDelayLegacySettings()
{
	if (!delay)
		return;

	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "DelayEnable", delay->enabled);
	config_set_int(ConfigManager::getInstance().getBasic(), "Output", "DelaySec", delay->delaySec);
	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "DelayPreserve", delay->preserveDelay);
}

void osn::Streaming::setReconnectLegacySettings()
{
	if (!reconnect)
		return;

	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "Reconnect", reconnect->enabled);
	config_set_uint(ConfigManager::getInstance().getBasic(), "Output", "RetryDelay", reconnect->retryDelay);
	config_set_uint(ConfigManager::getInstance().getBasic(), "Output", "MaxRetries", reconnect->maxRetries);
}

void osn::Streaming::setNetworkLegacySettings()
{
	if (!network)
		return;

	config_set_string(ConfigManager::getInstance().getBasic(), "Output", "BindIP", network->bindIP.c_str());
	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "DynamicBitrate", network->enableDynamicBitrate);
	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "NewSocketLoopEnable", network->enableDynamicBitrate);
	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "LowLatencyEnable", network->enableDynamicBitrate);
}