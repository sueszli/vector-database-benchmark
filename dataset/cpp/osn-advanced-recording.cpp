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

#include "osn-advanced-recording.hpp"
#include "osn-error.hpp"
#include "shared.hpp"
#include "osn-audio-track.hpp"
#include "osn-file-output.hpp"

void osn::IAdvancedRecording::Register(ipc::server &srv)
{
	std::shared_ptr<ipc::collection> cls = std::make_shared<ipc::collection>("AdvancedRecording");
	cls->register_function(std::make_shared<ipc::function>("Create", std::vector<ipc::type>{}, Create));
	cls->register_function(std::make_shared<ipc::function>("Destroy", std::vector<ipc::type>{ipc::type::UInt64}, Destroy));
	cls->register_function(std::make_shared<ipc::function>("GetVideoEncoder", std::vector<ipc::type>{ipc::type::UInt64}, GetVideoEncoder));
	cls->register_function(
		std::make_shared<ipc::function>("SetVideoEncoder", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetVideoEncoder));
	cls->register_function(std::make_shared<ipc::function>("GetVideoCanvas", std::vector<ipc::type>{ipc::type::UInt64}, GetVideoCanvas));
	cls->register_function(std::make_shared<ipc::function>("SetVideoCanvas", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetVideoCanvas));
	cls->register_function(std::make_shared<ipc::function>("GetMixer", std::vector<ipc::type>{ipc::type::UInt64}, GetMixer));
	cls->register_function(std::make_shared<ipc::function>("SetMixer", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetMixer));
	cls->register_function(std::make_shared<ipc::function>("GetRescaling", std::vector<ipc::type>{ipc::type::UInt64}, GetRescaling));
	cls->register_function(std::make_shared<ipc::function>("SetRescaling", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetRescaling));
	cls->register_function(std::make_shared<ipc::function>("GetOutputWidth", std::vector<ipc::type>{ipc::type::UInt64}, GetOutputWidth));
	cls->register_function(std::make_shared<ipc::function>("SetOutputWidth", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetOutputWidth));
	cls->register_function(std::make_shared<ipc::function>("GetOutputHeight", std::vector<ipc::type>{ipc::type::UInt64}, GetOutputHeight));
	cls->register_function(
		std::make_shared<ipc::function>("SetOutputHeight", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetOutputHeight));
	cls->register_function(std::make_shared<ipc::function>("GetUseStreamEncoders", std::vector<ipc::type>{ipc::type::UInt64}, GetUseStreamEncoders));
	cls->register_function(
		std::make_shared<ipc::function>("SetUseStreamEncoders", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetUseStreamEncoders));
	cls->register_function(std::make_shared<ipc::function>("Start", std::vector<ipc::type>{ipc::type::UInt64}, Start));
	cls->register_function(std::make_shared<ipc::function>("Stop", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, Stop));
	cls->register_function(std::make_shared<ipc::function>("Query", std::vector<ipc::type>{ipc::type::UInt64}, Query));
	cls->register_function(std::make_shared<ipc::function>("GetLegacySettings", std::vector<ipc::type>{}, GetLegacySettings));
	cls->register_function(std::make_shared<ipc::function>("SetLegacySettings", std::vector<ipc::type>{ipc::type::UInt64}, SetLegacySettings));
	cls->register_function(std::make_shared<ipc::function>("GetStreaming", std::vector<ipc::type>{ipc::type::UInt64}, GetStreaming));
	cls->register_function(std::make_shared<ipc::function>("SetStreaming", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt64}, SetStreaming));
	cls->register_function(std::make_shared<ipc::function>("SplitFile", std::vector<ipc::type>{ipc::type::UInt64}, SplitFile));
	cls->register_function(std::make_shared<ipc::function>("GetEnableFileSplit", std::vector<ipc::type>{ipc::type::UInt64}, GetEnableFileSplit));
	cls->register_function(
		std::make_shared<ipc::function>("SetEnableFileSplit", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetEnableFileSplit));
	cls->register_function(std::make_shared<ipc::function>("GetSplitType", std::vector<ipc::type>{ipc::type::UInt64}, GetSplitType));
	cls->register_function(std::make_shared<ipc::function>("SetSplitType", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetSplitType));
	cls->register_function(std::make_shared<ipc::function>("GetSplitTime", std::vector<ipc::type>{ipc::type::UInt64}, GetSplitTime));
	cls->register_function(std::make_shared<ipc::function>("SetSplitTime", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetSplitTime));
	cls->register_function(std::make_shared<ipc::function>("GetSplitSize", std::vector<ipc::type>{ipc::type::UInt64}, GetSplitSize));
	cls->register_function(std::make_shared<ipc::function>("SetSplitSize", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32}, SetSplitSize));
	cls->register_function(std::make_shared<ipc::function>("GetFileResetTimestamps", std::vector<ipc::type>{ipc::type::UInt64}, GetFileResetTimestamps));
	cls->register_function(std::make_shared<ipc::function>("SetFileResetTimestamps", std::vector<ipc::type>{ipc::type::UInt64, ipc::type::UInt32},
							       SetFileResetTimestamps));

	srv.register_collection(cls);
}

void osn::IAdvancedRecording::Create(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	uint64_t uid = osn::IAdvancedRecording::Manager::GetInstance().allocate(new AdvancedRecording());
	if (uid == UINT64_MAX) {
		PRETTY_ERROR_RETURN(ErrorCode::CriticalError, "Index list is full.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::Destroy(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IAdvancedRecording::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Recording reference is not valid.");
	}

	osn::IAdvancedRecording::Manager::GetInstance().free(recording);
	delete recording;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetRescaling(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(recording->rescaling));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetRescaling(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	recording->rescaling = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetOutputWidth(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(recording->outputWidth));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetOutputWidth(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	recording->outputWidth = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetOutputHeight(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(recording->outputHeight));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetOutputHeight(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	recording->outputHeight = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

bool osn::AdvancedRecording::UpdateEncoders()
{
	if (useStreamEncoders) {
		if (!streaming)
			return false;
		streaming->UpdateEncoders();
		videoEncoder = streaming->videoEncoder;

		if (obs_get_multiple_rendering()) {
			videoEncoder = osn::IRecording::duplicate_encoder(videoEncoder);
		}
	}

	if (!videoEncoder)
		return false;

	if (obs_get_multiple_rendering()) {
		obs_encoder_set_video_mix(videoEncoder, obs_video_mix_get(canvas, OBS_RECORDING_VIDEO_RENDERING));
	} else {
		obs_encoder_set_video_mix(videoEncoder, obs_video_mix_get(canvas, OBS_MAIN_VIDEO_RENDERING));
	}

	return true;
}

void osn::IAdvancedRecording::Start(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	if (!recording->output)
		recording->createOutput("ffmpeg_muxer", "recording");

	if (!recording->output) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Error while creating the recording output.");
	}

	int idx = 0;
	for (int i = 0; i < MAX_AUDIO_MIXES; i++) {
		osn::AudioTrack *audioTrack = osn::IAudioTrack::audioTracks[i];
		if ((recording->mixer & (1 << i)) != 0 && audioTrack && audioTrack->audioEnc) {
			obs_encoder_set_audio(audioTrack->audioEnc, obs_get_audio());
			obs_output_set_audio_encoder(recording->output, audioTrack->audioEnc, idx);
			idx++;
		}
	}

	if (!recording->UpdateEncoders() || !recording->videoEncoder) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Invalid video encoder.");
	}

	obs_output_set_video_encoder(recording->output, recording->videoEncoder);

	std::string path = recording->path;

	char lastChar = path.back();
	if (lastChar != '/' && lastChar != '\\')
		path += "/";

	path += GenerateSpecifiedFilename(recording->format, recording->noSpace, recording->fileFormat, recording->canvas->base_width,
					  recording->canvas->base_height);

	if (!recording->overwrite)
		FindBestFilename(path, recording->noSpace);

	obs_data_t *settings = obs_data_create();
	obs_data_set_string(settings, "path", path.c_str());
	obs_data_set_string(settings, "muxer_settings", recording->muxerSettings.c_str());
	obs_output_update(recording->output, settings);
	obs_data_release(settings);

	if (recording->enableFileSplit)
		recording->ConfigureRecFileSplitting();

	recording->startOutput();

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::Stop(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	obs_output_stop(recording->output);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetMixer(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(recording->mixer));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetMixer(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	recording->mixer = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetUseStreamEncoders(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(recording->useStreamEncoders));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetUseStreamEncoders(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Simple recording reference is not valid.");
	}

	recording->useStreamEncoders = args[1].value_union.ui32;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetLegacySettings(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	osn::AdvancedRecording *recording = new osn::AdvancedRecording();

	recording->path = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFilePath"));
	recording->format = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFormat"));
	recording->muxerSettings = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecMuxerCustom"));
	recording->noSpace = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFileNameWithoutSpace");
	recording->fileFormat = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "Output", "FilenameFormatting"));
	recording->overwrite = config_get_bool(ConfigManager::getInstance().getBasic(), "Output", "OverwriteIfExists");
	recording->muxerSettings = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecMuxerCustom"));

	recording->rescaling = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecRescale");
	const char *rescaleRes = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecRescaleRes"));
	unsigned int cx = 0;
	unsigned int cy = 0;
	if (recording->rescaling && rescaleRes) {
		if (sscanf(rescaleRes, "%ux%u", &cx, &cy) != 2) {
			cx = 0;
			cy = 0;
		}
		recording->outputWidth = cx;
		recording->outputHeight = cy;
	}

	recording->mixer = config_get_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecTracks");

	std::string encId = utility::GetSafeString(config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecEncoder"));
	recording->useStreamEncoders = encId.compare("") == 0 || encId.compare("none") == 0;

	if (!recording->useStreamEncoders) {
		obs_data_t *videoEncSettings = obs_data_create_from_json_file_safe(ConfigManager::getInstance().getRecord().c_str(), "bak");
		recording->videoEncoder = obs_video_encoder_create(encId.c_str(), "video-encoder", videoEncSettings, nullptr);
		osn::VideoEncoder::Manager::GetInstance().allocate(recording->videoEncoder);
	}

	recording->enableFileSplit = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFile");
	const char *splitFileType = config_get_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileType");
	if (strcmp(splitFileType, "Time") == 0)
		recording->splitType = SplitFileType::TIME;
	else if (strcmp(splitFileType, "Size") == 0)
		recording->splitType = SplitFileType::SIZE;
	else
		recording->splitType = SplitFileType::MANUAL;

	recording->splitTime = config_get_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileTime");
	recording->splitSize = config_get_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileSize");
	recording->fileResetTimestamps = config_get_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileResetTimestamps");

	uint64_t uid = osn::IAdvancedRecording::Manager::GetInstance().allocate(recording);
	if (uid == UINT64_MAX) {
		PRETTY_ERROR_RETURN(ErrorCode::CriticalError, "Index list is full.");
	}

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetLegacySettings(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IAdvancedRecording::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Recording reference is not valid.");
	}

	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFilePath", recording->path.c_str());
	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFormat", recording->format.c_str());
	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecMuxerCustom", recording->muxerSettings.c_str());
	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecFileNameWithoutSpace", recording->noSpace);
	config_set_string(ConfigManager::getInstance().getBasic(), "Output", "FilenameFormatting", recording->fileFormat.c_str());
	config_set_bool(ConfigManager::getInstance().getBasic(), "Output", "OverwriteIfExists", recording->overwrite);
	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecMuxerCustom", recording->muxerSettings.c_str());

	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecRescale", recording->rescaling);
	std::string rescaledRes = std::to_string(recording->outputWidth);
	rescaledRes += 'x';
	rescaledRes += std::to_string(recording->outputHeight);
	config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecRescaleRes", rescaledRes.c_str());

	config_set_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecTracks", recording->mixer);

	if (recording->useStreamEncoders) {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecEncoder", "none");
	} else if (!recording->useStreamEncoders && recording->videoEncoder) {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecEncoder", obs_encoder_get_id(recording->videoEncoder));

		obs_data_t *settings = obs_encoder_get_settings(recording->videoEncoder);

		if (!obs_data_save_json_safe(settings, ConfigManager::getInstance().getRecord().c_str(), "tmp", "bak")) {
			blog(LOG_ERROR, "Failed to save encoder %s", ConfigManager::getInstance().getStream().c_str());
		}
		obs_data_release(settings);
	}

	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFile", recording->enableFileSplit);

	switch (recording->splitType) {
	case SplitFileType::TIME: {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileType", "Time");
		break;
	}
	case SplitFileType::SIZE: {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileType", "Size");
		break;
	}
	default: {
		config_set_string(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileType", "Manual");
		break;
	}
	}

	config_set_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileTime", recording->splitTime);
	config_set_int(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileSize", recording->splitSize);
	config_set_bool(ConfigManager::getInstance().getBasic(), "AdvOut", "RecSplitFileResetTimestamps", recording->fileResetTimestamps);

	config_save_safe(ConfigManager::getInstance().getBasic(), "tmp", nullptr);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::GetStreaming(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Recording reference is not valid.");
	}

	uint64_t uid = osn::IAdvancedStreaming::Manager::GetInstance().find(recording->streaming);

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	rval.push_back(ipc::value(uid));
	AUTO_DEBUG;
}

void osn::IAdvancedRecording::SetStreaming(void *data, const int64_t id, const std::vector<ipc::value> &args, std::vector<ipc::value> &rval)
{
	AdvancedRecording *recording = static_cast<AdvancedRecording *>(osn::IFileOutput::Manager::GetInstance().find(args[0].value_union.ui64));
	if (!recording) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Recording reference is not valid.");
	}

	AdvancedStreaming *streaming = static_cast<AdvancedStreaming *>(osn::IAdvancedStreaming::Manager::GetInstance().find(args[1].value_union.ui64));
	if (!streaming) {
		PRETTY_ERROR_RETURN(ErrorCode::InvalidReference, "Streaming reference is not valid.");
	}

	recording->streaming = streaming;

	rval.push_back(ipc::value((uint64_t)ErrorCode::Ok));
	AUTO_DEBUG;
}