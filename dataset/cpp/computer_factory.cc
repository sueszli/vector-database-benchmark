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

#include "console/computer_factory.h"

#include "base/logging.h"
#include "base/desktop/pixel_format.h"
#include "build/build_config.h"

namespace console {

namespace {

const int kDefCompressRatio = 8;

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
void ComputerFactory::setDefaultDesktopManageConfig(proto::DesktopConfig* config)
{
    DCHECK(config);

    static const uint32_t kDefaultFlags =
        proto::ENABLE_CLIPBOARD | proto::ENABLE_CURSOR_SHAPE | proto::DISABLE_DESKTOP_EFFECTS |
        proto::DISABLE_DESKTOP_WALLPAPER | proto::CLEAR_CLIPBOARD;

    config->set_flags(kDefaultFlags);
    config->set_video_encoding(proto::VIDEO_ENCODING_VP8);
    config->set_audio_encoding(proto::AUDIO_ENCODING_OPUS);
    config->set_update_interval(30);
    config->set_scale_factor(100);
    config->set_compress_ratio(kDefCompressRatio);

    serializePixelFormat(base::PixelFormat::RGB332(), config->mutable_pixel_format());
}

//--------------------------------------------------------------------------------------------------
void ComputerFactory::setDefaultDesktopViewConfig(proto::DesktopConfig* config)
{
    DCHECK(config);

    static const uint32_t kDefaultFlags =
        proto::DISABLE_DESKTOP_EFFECTS | proto::DISABLE_DESKTOP_WALLPAPER;

    config->set_flags(kDefaultFlags);
    config->set_video_encoding(proto::VIDEO_ENCODING_VP8);
    config->set_audio_encoding(proto::AUDIO_ENCODING_OPUS);
    config->set_update_interval(30);
    config->set_scale_factor(100);
    config->set_compress_ratio(kDefCompressRatio);

    serializePixelFormat(base::PixelFormat::RGB332(), config->mutable_pixel_format());
}

//--------------------------------------------------------------------------------------------------
// static
proto::address_book::Computer ComputerFactory::defaultComputer()
{
    proto::address_book::Computer computer;

    computer.set_session_type(proto::SESSION_TYPE_DESKTOP_MANAGE);
    computer.set_port(DEFAULT_HOST_TCP_PORT);

    proto::address_book::InheritConfig* inherit = computer.mutable_inherit();
    inherit->set_credentials(false);
    inherit->set_desktop_manage(true);
    inherit->set_desktop_view(true);

    setDefaultDesktopManageConfig(computer.mutable_session_config()->mutable_desktop_manage());
    setDefaultDesktopViewConfig(computer.mutable_session_config()->mutable_desktop_view());

    return computer;
}

} // namespace console
