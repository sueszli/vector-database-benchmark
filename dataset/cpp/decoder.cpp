// Copyright 2018 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "audio_core/hle/decoder.h"

namespace AudioCore::HLE {

DecoderSampleRate GetSampleRateEnum(u32 sample_rate) {
    switch (sample_rate) {
    case 48000:
        return DecoderSampleRate::Rate48000;
    case 44100:
        return DecoderSampleRate::Rate44100;
    case 32000:
        return DecoderSampleRate::Rate32000;
    case 24000:
        return DecoderSampleRate::Rate24000;
    case 22050:
        return DecoderSampleRate::Rate22050;
    case 16000:
        return DecoderSampleRate::Rate16000;
    case 12000:
        return DecoderSampleRate::Rate12000;
    case 11025:
        return DecoderSampleRate::Rate11025;
    case 8000:
        return DecoderSampleRate::Rate8000;
    default:
        LOG_WARNING(Audio_DSP, "Unknown decoder sample rate: {}", sample_rate);
        return DecoderSampleRate::Rate48000;
    }
}

DecoderBase::~DecoderBase(){};

NullDecoder::NullDecoder() = default;

NullDecoder::~NullDecoder() = default;

std::optional<BinaryMessage> NullDecoder::ProcessRequest(const BinaryMessage& request) {
    BinaryMessage response{};
    switch (request.header.cmd) {
    case DecoderCommand::Init:
    case DecoderCommand::Shutdown:
    case DecoderCommand::SaveState:
    case DecoderCommand::LoadState:
        response = request;
        response.header.result = ResultStatus::Success;
        return response;
    case DecoderCommand::EncodeDecode:
        response.header.codec = request.header.codec;
        response.header.cmd = request.header.cmd;
        response.header.result = ResultStatus::Success;
        response.decode_aac_response.num_channels = 2; // Just assume stereo here
        response.decode_aac_response.size = request.decode_aac_request.size;
        response.decode_aac_response.num_samples = 1024; // Just assume 1024 here
        return response;
    default:
        LOG_ERROR(Audio_DSP, "Got unknown binary request: {}",
                  static_cast<u16>(request.header.cmd));
        return std::nullopt;
    }
};
} // namespace AudioCore::HLE
