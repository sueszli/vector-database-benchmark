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

#include "base/audio/audio_capturer.h"

#if defined(OS_WIN)
#include "base/audio/audio_capturer_win.h"
#endif // defined(OS_WIN)

#if defined(OS_LINUX)
#include "base/audio/audio_capturer_linux.h"
#endif // defined(OS_LINUX)

#include "base/logging.h"
#include "proto/desktop.pb.h"

namespace base {

//--------------------------------------------------------------------------------------------------
// Returns true if the sampling rate is supported by Pepper.
bool AudioCapturer::isValidSampleRate(int sample_rate)
{
    switch (sample_rate)
    {
        case proto::AudioPacket::SAMPLING_RATE_44100:
        case proto::AudioPacket::SAMPLING_RATE_48000:
        case proto::AudioPacket::SAMPLING_RATE_96000:
        case proto::AudioPacket::SAMPLING_RATE_192000:
            return true;

        default:
            return false;
    }
}

//--------------------------------------------------------------------------------------------------
std::unique_ptr<AudioCapturer> AudioCapturer::create()
{
#if defined(OS_WIN)
    return std::unique_ptr<AudioCapturer>(new AudioCapturerWin());
#elif defined(OS_LINUX)
    return std::unique_ptr<AudioCapturer>(new AudioCapturerLinux());
#else
    NOTIMPLEMENTED();
    return nullptr;
#endif
}

} // namespace base
