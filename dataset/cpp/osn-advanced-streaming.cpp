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

#include "osn-advanced-streaming.hpp"
#include "osn-video-encoder.hpp"
#include "osn-service.hpp"
#include "osn-error.hpp"
#include "shared.hpp"
#include "nodeobs_audio_encoders.h"
#include "osn-audio-track.hpp"

void osn::IAdvancedStreaming::Register(ipc::server &srv)
{
	std::shared_ptr<ipc::collection> cls = std::make_shared<ipc::collection>("AdvancedStreaming");
	cls->register_function(std::make_shared<ipc::function>("Create", std::vector<ipc::type>{}, Create));
	cls->register_function(std::make_shared<ipc::function>("Destroy", std::vector<ipc::type>{ipc::type::UInt64}, Destroy));
	cls->register_function(std::make_shared<ipc::function>("GetService", std::vector<ipc::type>{ipc::type::UInt64}, GetService));
	cls->register_function(std::make_shared<ipc::function>("SetService", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetService));
	cls->register_function(std::make_shared<ipc::function>("GetVideoEncoder", std::vector<ipc::type>{ipc::type::UInt64}, GetVideoEncoder));
	cls->register_function(
		std::make_shared<ipc::function>("SetVideoEncoder", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetVideoEncoder));
	cls->register_function(std::make_shared<ipc::function>("GetVideoCanvas", std::vector<ipc::type>{ipc::type::UInt64}, GetVideoCanvas));
	cls->register_function(std::make_shared<ipc::function>("SetVideoCanvas", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetVideoCanvas));
	cls->register_function(std::make_shared<ipc::function>("GetEnforceServiceBirate", std::vector<ipc::type>{ipc::type::UInt64}, GetEnforceServiceBirate));
	cls->register_function(std::make_shared<ipc::function>("SetEnforceServiceBirate", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32},
							       SetEnforceServiceBirate));
	cls->register_function(std::make_shared<ipc::function>("GetEnableTwitchVOD", std::vector<ipc::type>{ipc::type::UInt64}, GetEnableTwitchVOD));
	cls->register_function(
		std::make_shared<ipc::function>("SetEnableTwitchVOD", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetEnableTwitchVOD));
	cls->register_function(std::make_shared<ipc::function>("GetAudioTrack", std::vector<ipc::type>{ipc::type::UInt64}, GetAudioTrack));
	cls->register_function(std::make_shared<ipc::function>("SetAudioTrack", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetAudioTrack));
	cls->register_function(std::make_shared<ipc::function>("GetTwitchTrack", std::vector<ipc::type>{ipc::type::UInt64}, GetTwitchTrack));
	cls->register_function(std::make_shared<ipc::function>("SetTwitchTrack", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetTwitchTrack));
	cls->register_function(std::make_shared<ipc::function>("GetRescaling", std::vector<ipc::type>{ipc::type::UInt64}, GetRescaling));
	cls->register_function(std::make_shared<ipc::function>("SetRescaling", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetRescaling));
	cls->register_function(std::make_shared<ipc::function>("GetOutputWidth", std::vector<ipc::type>{ipc::type::UInt64}, GetOutputWidth));
	cls->register_function(std::make_shared<ipc::function>("SetOutputWidth", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetOutputWidth));
	cls->register_function(std::make_shared<ipc::function>("GetOutputHeight", std::vector<ipc::type>{ipc::type::UInt64}, GetOutputHeight));
	cls->register_function(
		std::make_shared<ipc::function>("SetOutputHeight", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetOutputHeight));
	cls->register_function(std::make_shared<ipc::function>("GetDelay", std::vector<ipc::type>{ipc::type::UInt64}, GetDelay));
	cls->register_function(std::make_shared<ipc::function>("SetDelay", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetDelay));
	cls->register_function(std::make_shared<ipc::function>("GetReconnect", std::vector<ipc::type>{ipc::type::UInt64}, GetReconnect));
	cls->register_function(std::make_shared<ipc::function>("SetReconnect", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetReconnect));
	cls->register_function(std::make_shared<ipc::function>("GetNetwork", std::vector<ipc::type>{ipc::type::UInt64}, GetNetwork));
	cls->register_function(std::make_shared<ipc::function>("SetNetwork", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetNetwork));
	cls->register_function(std::make_shared<ipc::function>("Start", std::vector<ipc::type>{ipc::type::UInt64}, Start));
	cls->register_function(std::make_shared<ipc::function>("Stop", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, Stop));
	cls->register_function(std::make_shared<ipc::function>("Query", std::vector<ipc::type>{ipc::type::UInt64}, Query));
	cls->register_function(std::make_shared<ipc::function>("GetLegacySettings", std::vector<ipc::type>{}, GetLegacySettings));
	cls->register_function(std::make_shared<ipc::function>("SetLegacySettings", std::vector<ipc::type>{ipc::type::UInt64}, SetLegacySettings));

	srv.register_collection(cls);
}

void osn::IAdvancedStreaming::Create(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	uint64_t uid = osn::IAdvancedStreaming::Manager::GetInstance().allocate(new AdvancedStreaming());
	if (uid == UINT64_MAX) {
		PRETTY_ERROR_RETURN(ErrorCode::CriticalError, "Index list is full.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::Destroy(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	osn::IAdvancedStreaming::Manager::GetInstance().free(streaming);
	delete streaming;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetAudioTrack(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->audioTrack));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetAudioTrack(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	streaming->audioTrack = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetTwitchTrack(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->twitchTrack));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetTwitchTrack(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	streaming->twitchTrack = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetRescaling(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->rescaling));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetRescaling(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	streaming->rescaling = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetOutputWidth(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->outputWidth));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetOutputWidth(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	streaming->outputWidth = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetOutputHeight(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(streaming->outputHeight));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetOutputHeight(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	streaming->outputHeight = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

static obs_encoder_t *createAudioEncoder(uint32_t bitrate)
{
	obs_encoder_t *audioEncoder = nullptr;

	audioEncoder = obs_audio_encoder_create(GetAACEncoderForBitrate(bitrate), "audio", nullptr, 0, nullptr);

	return audioEncoder;
}

static bool setAudioEncoder(osn::AdvancedStreaming *streaming)
{
	osn::AudioTrack *audioTrack = osn::IAudioTrack::audioTracks[streaming->audioTrack - 1];
	if (!audioTrack)
		return false;
	if (!audioTrack->audioEnc)
		return false;

	obs_encoder_set_audio(audioTrack->audioEnc, obs_get_audio());
	obs_output_set_audio_encoder(streaming->output, audioTrack->audioEnc, 0);

	return true;
}

static constexpr int kSoundtrackArchiveEncoderIdx = 1;
static constexpr int kSoundtrackArchiveTrackIdx = 5;

static uint32_t setMixer(obs_source_t *source, const int mixerIdx, const bool checked)
{
	uint32_t mixers = obs_source_get_audio_mixers(source);
	uint32_t new_mixers = mixers;
	if (checked) {
		new_mixers |= (1 << mixerIdx);
	} else {
		new_mixers &= ~(1 << mixerIdx);
	}
	obs_source_set_audio_mixers(source, new_mixers);
	return mixers;
}

static void SetupTwitchSoundtrackAudio(osn::AdvancedStreaming *streaming)
{
	// These are magic ints provided by OBS for default sources:
	// 0 is the main scene/transition which you'd see on the main preview,
	// 1-2 are desktop audio 1 and 2 as you'd see in audio settings,
	// 2-4 are mic/aux 1-3 as you'd see in audio settings
	auto desktopSource1 = obs_get_output_source(1);
	auto desktopSource2 = obs_get_output_source(2);

	// Since our plugin duplicates all of the desktop sources, we want to ensure that both of the
	// default desktop sources, provided by OBS, are not set to mix on our custom encoder track.
	streaming->oldMixer_desktopSource1 = setMixer(desktopSource1, kSoundtrackArchiveTrackIdx, false);
	streaming->oldMixer_desktopSource2 = setMixer(desktopSource2, kSoundtrackArchiveTrackIdx, false);

	obs_source_release(desktopSource1);
	obs_source_release(desktopSource2);

	if (streaming->streamArchive && obs_encoder_active(streaming->streamArchive))
		return;

	if (!streaming->streamArchive) {
		streaming->streamArchive =
			obs_audio_encoder_create("ffmpeg_aac", "Soundtrack by Twitch Archive Encoder", nullptr, kSoundtrackArchiveTrackIdx, nullptr);
		obs_encoder_set_audio(streaming->streamArchive, obs_get_audio());
	}

	obs_output_set_audio_encoder(streaming->output, streaming->streamArchive, kSoundtrackArchiveEncoderIdx);

	osn::AudioTrack *audioTrack = osn::IAudioTrack::audioTracks[streaming->twitchTrack];
	if (!audioTrack)
		return;

	obs_data_t *settings = obs_data_create();
	obs_data_set_int(settings, "bitrate", audioTrack->bitrate);
	obs_encoder_update(streaming->streamArchive, settings);
	obs_data_release(settings);
}

static void StopTwitchSoundtrackAudio(osn::Streaming *streaming)
{
	if (streaming->streamArchive) {
		obs_encoder_release(streaming->streamArchive);
		streaming->streamArchive = nullptr;
	}

	auto desktopSource1 = obs_get_output_source(1);
	auto desktopSource2 = obs_get_output_source(2);

	obs_source_set_audio_mixers(desktopSource1, streaming->oldMixer_desktopSource1);
	obs_source_set_audio_mixers(desktopSource2, streaming->oldMixer_desktopSource2);

	obs_source_release(desktopSource1);
	obs_source_release(desktopSource2);
}

void osn::AdvancedStreaming::UpdateEncoders()
{
	if (!videoEncoder)
		return;

	if (obs_encoder_active(videoEncoder))
		return;

	obs_data_t *settings = obs_encoder_get_settings(videoEncoder);
	if (enforceServiceBitrate) {

		int bitrate = (int)obs_data_get_int(settings, "bitrate");
		int keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
		obs_service_apply_encoder_settings(service, settings, nullptr);

		int enforced_keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
		if (keyint_sec != 0 && keyint_sec < enforced_keyint_sec)
			obs_data_set_int(settings, "keyint_sec", keyint_sec);
	}

	video_t *video = obs_get_video();
	enum video_format format = video_output_get_format(video);

	switch (format) {
	case VIDEO_FORMAT_I420:
	case VIDEO_FORMAT_NV12:
		// case VIDEO_FORMAT_I010:
		// case VIDEO_FORMAT_P010:
		break;
	default:
		obs_encoder_set_preferred_video_format(videoEncoder, VIDEO_FORMAT_NV12);
	}

	obs_encoder_update(videoEncoder, settings);
	obs_data_release(settings);

	if (obs_get_multiple_rendering()) {
		obs_encoder_set_video_mix(videoEncoder, obs_video_mix_get(canvas, OBS_STREAMING_VIDEO_RENDERING));
	} else {
		obs_encoder_set_video_mix(videoEncoder, obs_video_mix_get(canvas, OBS_MAIN_VIDEO_RENDERING));
	}
}

void osn::IAdvancedStreaming::Start(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	if (!streaming->videoEncoder) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid video encoder.");
	}

	streaming->UpdateEncoders();

	if (!streaming->service) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid service.");
	}

	const char *type = obs_service_get_output_type(streaming->service);
	if (!type)
		type = "rtmp_output";

	if (!streaming->output || strcmp(obs_output_get_id(streaming->output), type) != 0)
		streaming->createOutput(type, "stream");

	if (!streaming->output) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Error while creating the streaming output.");
	}

	if (!streaming->videoEncoder) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Error while creating the video encoder.");
	}

	if (!setAudioEncoder(streaming)) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Error while creating the audio encoder.");
	}

	if (streaming->rescaling)
		obs_encoder_set_scaled_size(streaming->videoEncoder, streaming->outputWidth, streaming->outputHeight);

	obs_output_set_video_encoder(streaming->output, streaming->videoEncoder);

	if (streaming->enableTwitchVOD) {
		streaming->twitchVODSupported = streaming->isTwitchVODSupported();
		if (streaming->twitchVODSupported)
			SetupTwitchSoundtrackAudio(streaming);
	}

	obs_output_set_service(streaming->output, streaming->service);

	if (!streaming->delay) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid delay.");
	}
	obs_output_set_delay(streaming->output, streaming->delay->enabled ? uint32_t(streaming->delay->delaySec) : 0,
			     streaming->delay->preserveDelay ? OBS_OUTPUT_DELAY_PRESERVE : 0);

	if (!streaming->reconnect) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid reconnect.");
	}
	uint32_t maxReties = streaming->reconnect->enabled ? streaming->reconnect->maxRetries : 0;
	obs_output_set_reconnect_settings(streaming->output, maxReties, streaming->reconnect->retryDelay);

	if (!streaming->network) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid network.");
	}

	obs_data_t *settings = obs_data_create();
	obs_data_set_string(settings, "bind_ip", streaming->network->bindIP.c_str());
	obs_data_set_bool(settings, "dyn_bitrate", streaming->network->enableDynamicBitrate);
	obs_data_set_bool(settings, "new_socket_loop_enabled", streaming->network->enableOptimizations);
	obs_data_set_bool(settings, "low_latency_mode_enabled", streaming->network->enableLowLatency);
	obs_output_update(streaming->output, settings);
	obs_data_release(settings);

	streaming->startOutput();

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::Stop(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	Streaming *streaming = osn::IStreaming::Manager::GetInstance().find(args[0].value_union.ui64);
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple streaming reference is not valid.");
	}

	if (!streaming->output) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid streaming output.");
	}

	bool force = args[1].value_union.ui32;

	if (force)
		obs_output_force_stop(streaming->output);
	else
		obs_output_stop(streaming->output);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::GetLegacySettings(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	osn::AdvancedStreaming *streaming = new osn::AdvancedStreaming();
	const char *encId = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "Encoder"));
	obs_data_t *videoEncSettings = obs_data_create_from_json_file_safe(ConfigManager::getInstance().getStream().c_str(), "bak");
	streaming->videoEncoder = obs_video_encoder_create(encId, "video-encoder", videoEncSettings, nullptr);
	osn::VideoEncoder::Manager::GetInstance().allocate(streaming->videoEncoder);

	streaming->audioTrack = config_get_int(ConfigManager::getInstance().getBasic(), "AdvOut", "TrackIndex") - 1;
	streaming->enableTwitchVOD = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "VodTrackEnabled");
	streaming->twitchTrack = config_get_int(ConfigManager::getInstance().getBasic(), "AdvOut", "VodTrackIndex") - 1;
	streaming->enforceServiceBitrate = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "ApplyServiceSettings");

	streaming->rescaling = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "Rescale");
	const char *rescaleRes = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RescaleRes"));
	unsigned int cx = 0;
	unsigned int cy = 0;
	if (streaming->rescaling && rescaleRes) {
		if (sscanf(rescaleRes, "%ux%u", &cx, &cy) != 2) {
			cx = 0;
			cy = 0;
		}
		streaming->outputWidth = cx;
		streaming->outputHeight = cy;
	}

	streaming->getDelayLegacySettings();
	streaming->getReconnectLegacySettings();
	streaming->getNetworkLegacySettings();

	streaming->service = osn::Service::GetLegacyServiceSettings();
	osn::Service::Manager::GetInstance().allocate(streaming->service);

	uint64_t uid = osn::IAdvancedStreaming::Manager::GetInstance().allocate(streaming);
	if (uid == UINT64_MAX) {
		PRETTY_ERROR_RETURN(ErrorCode::CriticalError, "Index list is full.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IAdvancedStreaming::SetLegacySettings(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	streaming->setDelayLegacySettings();
	streaming->setReconnectLegacySettings();
	streaming->setNetworkLegacySettings();

	osn::Service::SetLegacyServiceSettings(streaming->service);

	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "VodTrackEnabled", streaming->enableTwitchVOD);
	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "ApplyServiceSettings", streaming->enforceServiceBitrate);
	config_set_int(ConfigManager::getInstance().getBasic(), "AdvOut", "TrackIndex", streaming->audioTrack + 1);
	config_set_int(ConfigManager::getInstance().getBasic(), "AdvOut", "VodTrackIndex", streaming->twitchTrack + 1);

	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "Rescale", streaming->rescaling);
	std::string rescaledRes = std::to_string(streaming->outputWidth);
	rescaledRes += 'x';
	rescaledRes += std::to_string(streaming->outputHeight);
	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RescaleRes", rescaledRes.c_str());

	if (streaming->videoEncoder) {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "Encoder", obs_encoder_get_id(streaming->videoEncoder));

		obs_data_t *settings = obs_encoder_get_settings(streaming->videoEncoder);

		if (!obs_data_save_json_safe(settings, ConfigManager::getInstance().getStream().c_str(), "tmp", "bak")) {
			blog(LOG_ERROR, "Failed to save encoder %s", ConfigManager::getInstance().getStream().c_str());
		}
		obs_data_release(settings);
	}

	config_save_safe(ConfigManager::getInstance().getBasic(), "tmp", nullptr);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}