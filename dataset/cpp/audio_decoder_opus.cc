//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/codec/audio_decoder_opus.h"

#include "base/logging.h"

#include <cstdint>

#include <opus.h>

namespace base {

namespace {

// Maximum size of an Opus frame in milliseconds.
const std::chrono::milliseconds kMaxFrameSizeMs { 120 };

// Hosts will never generate more than 100 frames in a single packet.
const int kMaxFramesPerPacket = 100;

const proto::AudioPacket::SamplingRate kSamplingRate =
    proto::AudioPacket::SAMPLING_RATE_48000;

} // namespace

//--------------------------------------------------------------------------------------------------
AudioDecoderOpus::AudioDecoderOpus() = default;

//--------------------------------------------------------------------------------------------------
AudioDecoderOpus::~AudioDecoderOpus()
{
    destroyDecoder();
}

//--------------------------------------------------------------------------------------------------
void AudioDecoderOpus::initDecoder()
{
    DCHECK(!decoder_);

    int error;
    decoder_ = opus_decoder_create(kSamplingRate, channels_, &error);
    if (!decoder_)
    {
        LOG(LS_ERROR) << "Failed to create OPUS decoder; Error code: " << error;
    }
}

//--------------------------------------------------------------------------------------------------
void AudioDecoderOpus::destroyDecoder()
{
    if (decoder_)
    {
        opus_decoder_destroy(decoder_);
        decoder_ = nullptr;
    }
}

//--------------------------------------------------------------------------------------------------
bool AudioDecoderOpus::resetForPacket(const proto::AudioPacket& packet)
{
    if (packet.channels() != channels_ || packet.sampling_rate() != sampling_rate_)
    {
        destroyDecoder();

        channels_ = packet.channels();
        sampling_rate_ = packet.sampling_rate();

        if (channels_ <= 0 || channels_ > 2 || sampling_rate_ != kSamplingRate)
        {
            LOG(LS_ERROR) << "Unsupported OPUS parameters: "
                          << channels_ << " channels with "
                          << sampling_rate_ << " samples per second.";
            return false;
        }
    }

    if (!decoder_)
        initDecoder();

    return decoder_ != nullptr;
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<proto::AudioPacket> AudioDecoderOpus::decode(const proto::AudioPacket& packet)
{
    if (packet.encoding() != proto::AUDIO_ENCODING_OPUS)
    {
        LOG(LS_ERROR) << "Received an audio packet with encoding "
                      << packet.encoding() << " when an OPUS packet was expected.";
        return nullptr;
    }

    if (packet.data_size() > kMaxFramesPerPacket)
    {
        LOG(LS_ERROR) << "Received an packet with too many frames.";
        return nullptr;
    }

    if (!resetForPacket(packet))
        return nullptr;

    // Create a new packet of decoded data.
    std::unique_ptr<proto::AudioPacket> decoded_packet(new proto::AudioPacket());
    decoded_packet->set_encoding(proto::AUDIO_ENCODING_RAW);
    decoded_packet->set_sampling_rate(kSamplingRate);
    decoded_packet->set_bytes_per_sample(proto::AudioPacket::BYTES_PER_SAMPLE_2);
    decoded_packet->set_channels(packet.channels());

    int max_frame_samples = static_cast<int>(
        kMaxFrameSizeMs * kSamplingRate / std::chrono::milliseconds(1000));
    int max_frame_bytes = max_frame_samples * channels_ * decoded_packet->bytes_per_sample();

    std::string* decoded_data = decoded_packet->add_data();
    decoded_data->resize(
        static_cast<size_t>(packet.data_size()) * static_cast<size_t>(max_frame_bytes));
    int buffer_pos = 0;

    for (int i = 0; i < packet.data_size(); ++i)
    {
        int16_t* pcm_buffer = reinterpret_cast<int16_t*>(std::data(*decoded_data) + buffer_pos);
        CHECK_LE(buffer_pos + max_frame_bytes, static_cast<int>(decoded_data->size()));
        const std::string& frame = packet.data(i);
        const unsigned char* frame_data = reinterpret_cast<const unsigned char*>(frame.data());
        int result = opus_decode(decoder_, frame_data, static_cast<opus_int32>(frame.size()),
                                 pcm_buffer, max_frame_samples, 0);
        if (result < 0)
        {
            LOG(LS_ERROR) << "Failed decoding Opus frame. Error code: " << result;
            destroyDecoder();
            return nullptr;
        }

        buffer_pos += result * packet.channels() * decoded_packet->bytes_per_sample();
    }

    if (!buffer_pos)
        return nullptr;

    decoded_data->resize(static_cast<size_t>(buffer_pos));
    return decoded_packet;
}

} // namespace base
