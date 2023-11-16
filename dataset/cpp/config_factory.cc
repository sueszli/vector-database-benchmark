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

#include "client/config_factory.h"

#include "base/logging.h"
#include "base/desktop/pixel_format.h"

namespace client {

namespace {

const proto::VideoEncoding kDefaultVideoEncoding = proto::VIDEO_ENCODING_VP8;
const proto::AudioEncoding kDefaultAudioEncoding = proto::AUDIO_ENCODING_OPUS;

const int kDefCompressRatio = 8;
const int kMinCompressRatio = 1;
const int kMaxCompressRatio = 22;

//--------------------------------------------------------------------------------------------------
void serializePixelFormat(const base::PixelFormat& from, proto::PixelFormat* to)
{
    to->set_bits_per_pixel(from.bitsPerPixel());

    to->set_red_max(from.redMax());
    to->set_green_max(from.greenMax());
    to->set_blue_max(from.blueMax());

    to->set_red_shift(from.redShift());
    to->set_green_shift(from.greenShift());
    to->set_blue_shift(from.blueShift());
}

} // namespace

//--------------------------------------------------------------------------------------------------
// static
proto::DesktopConfig ConfigFactory::defaultDesktopManageConfig()
{
    proto::DesktopConfig config;
    setDefaultDesktopManageConfig(&config);
    return config;
}

//--------------------------------------------------------------------------------------------------
// static
proto::DesktopConfig ConfigFactory::defaultDesktopViewConfig()
{
    proto::DesktopConfig config;
    setDefaultDesktopViewConfig(&config);
    return config;
}

//--------------------------------------------------------------------------------------------------
// static
void ConfigFactory::setDefaultDesktopManageConfig(proto::DesktopConfig* config)
{
    DCHECK(config);

    static const uint32_t kDefaultFlags =
        proto::ENABLE_CLIPBOARD | proto::ENABLE_CURSOR_SHAPE | proto::DISABLE_DESKTOP_EFFECTS |
        proto::DISABLE_DESKTOP_WALLPAPER | proto::CLEAR_CLIPBOARD;

    config->set_flags(kDefaultFlags);
    config->set_video_encoding(kDefaultVideoEncoding);
    config->set_audio_encoding(kDefaultAudioEncoding);
    config->set_compress_ratio(kDefCompressRatio);

    serializePixelFormat(base::PixelFormat::RGB332(), config->mutable_pixel_format());
    fixupDesktopConfig(config);
}

//--------------------------------------------------------------------------------------------------
// static
void ConfigFactory::setDefaultDesktopViewConfig(proto::DesktopConfig* config)
{
    DCHECK(config);

    static const uint32_t kDefaultFlags =
        proto::DISABLE_DESKTOP_EFFECTS | proto::DISABLE_DESKTOP_WALLPAPER;

    config->set_flags(kDefaultFlags);
    config->set_video_encoding(kDefaultVideoEncoding);
    config->set_audio_encoding(kDefaultAudioEncoding);
    config->set_compress_ratio(kDefCompressRatio);

    serializePixelFormat(base::PixelFormat::RGB332(), config->mutable_pixel_format());
    fixupDesktopConfig(config);
}

//--------------------------------------------------------------------------------------------------
// static
void ConfigFactory::fixupDesktopConfig(proto::DesktopConfig* config)
{
    DCHECK(config);

    config->set_scale_factor(100);
    config->set_update_interval(30);

    if (config->compress_ratio() < kMinCompressRatio || config->compress_ratio() > kMaxCompressRatio)
        config->set_compress_ratio(kDefCompressRatio);

    if (config->audio_encoding() == proto::AUDIO_ENCODING_DEFAULT)
        config->set_audio_encoding(kDefaultAudioEncoding);
}

} // namespace client
