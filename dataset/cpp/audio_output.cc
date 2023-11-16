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

#include "base/audio/audio_output.h"

#include "base/logging.h"
#include "build/build_config.h"

#if defined(OS_WIN)
#include "base/audio/audio_output_win.h"
#elif defined(OS_MAC)
#include "base/audio/audio_output_mac.h"
#elif defined(OS_LINUX)
#include "base/audio/audio_output_pulse.h"
#endif

#include <cstring>

namespace base {

//--------------------------------------------------------------------------------------------------
AudioOutput::AudioOutput(const NeedMoreDataCB& need_more_data_cb)
    : need_more_data_cb_(need_more_data_cb)
{
    // Nothing
}

//--------------------------------------------------------------------------------------------------
// static
std::unique_ptr<AudioOutput> AudioOutput::create(const NeedMoreDataCB& need_more_data_cb)
{
#if defined(OS_WIN)
    return std::make_unique<AudioOutputWin>(need_more_data_cb);
#elif defined(OS_MAC)
    return std::make_unique<AudioOutputMac>(need_more_data_cb);
#elif defined(OS_LINUX)
    return std::make_unique<AudioOutputPulse>(need_more_data_cb);
#else
    NOTIMPLEMENTED();
    return nullptr;
#endif
}

//--------------------------------------------------------------------------------------------------
void AudioOutput::onDataRequest(int16_t* audio_samples, size_t audio_samples_count)
{
    static const size_t kSamplesPerChannel10ms = kSampleRate * 10 / 1000;
    static const size_t kSamplesPer10ms = kChannels * kSamplesPerChannel10ms;
    static const size_t kBytesPer10ms = kSamplesPer10ms * sizeof(int16_t);

    const size_t rounds = audio_samples_count / kSamplesPer10ms;
    for (size_t i = 0; i < rounds; ++i)
    {
        // Get 10ms decoded audio.
        size_t num_bytes = need_more_data_cb_(audio_samples + (i * kSamplesPer10ms), kBytesPer10ms);
        if (!num_bytes)
        {
            memset(audio_samples, 0, audio_samples_count * sizeof(int16_t));
            return;
        }
    }
}

} // namespace base
