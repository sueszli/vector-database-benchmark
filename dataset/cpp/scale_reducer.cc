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

#include "base/codec/scale_reducer.h"

#include "base/logging.h"
#include "base/desktop/frame_simple.h"

#include <libyuv/scale_argb.h>

namespace base {

//--------------------------------------------------------------------------------------------------
ScaleReducer::ScaleReducer()
{
    LOG(LS_INFO) << "Ctor";
}

//--------------------------------------------------------------------------------------------------
ScaleReducer::~ScaleReducer()
{
    LOG(LS_INFO) << "Dtor";
}

//--------------------------------------------------------------------------------------------------
const Frame* ScaleReducer::scaleFrame(const Frame* source_frame, const Size& target_size)
{
    DCHECK(source_frame);
    DCHECK(!source_frame->constUpdatedRegion().isEmpty());
    DCHECK(source_frame->format() == PixelFormat::ARGB());

    const Size& source_size = source_frame->size();
    if (source_size.width() == 0 || source_size.height() == 0)
    {
        LOG(LS_ERROR) << "Invalid source frame size: "
                      << source_size.width() << "x" << source_size.height();
        return nullptr;
    }

    if (source_size_ != source_size || target_size_ != target_size)
    {
        const_cast<Frame*>(source_frame)->updatedRegion()->addRect(Rect::makeSize(source_size));

        scale_x_ = static_cast<double>(target_size.width() * 100.0) /
            static_cast<double>(source_size.width());
        scale_y_ = static_cast<double>(target_size.height() * 100.0) /
            static_cast<double>(source_size.height());
        source_size_ = source_size;
        target_size_ = target_size;
        target_frame_.reset();

        LOG(LS_INFO) << "Scale mode changed (source:" << source_size << " target:" << target_size
                     << " scale_x:" << scale_x_ << " scale_y:" << scale_y_ << ")";
    }

    if (source_size == target_size)
        return source_frame;

    Rect target_frame_rect = Rect::makeSize(target_size);

    if (!target_frame_)
    {
        target_frame_ = FrameSimple::create(target_size, PixelFormat::ARGB());
        if (!target_frame_)
        {
            LOG(LS_ERROR) << "Unable to create target frame";
            return nullptr;
        }

        target_frame_->updatedRegion()->addRect(target_frame_rect);

        libyuv::ARGBScale(source_frame->frameData(),
                          source_frame->stride(),
                          source_size.width(),
                          source_size.height(),
                          target_frame_->frameData(),
                          target_frame_->stride(),
                          target_size.width(),
                          target_size.height(),
                          libyuv::kFilterBox);
    }
    else
    {
        Region* updated_region = target_frame_->updatedRegion();
        updated_region->clear();

        for (Region::Iterator it(source_frame->constUpdatedRegion());
             !it.isAtEnd(); it.advance())
        {
            Rect target_rect = scaledRect(it.rect());
            target_rect.intersectWith(target_frame_rect);

            libyuv::ARGBScaleClip(source_frame->frameData(),
                                  source_frame->stride(),
                                  source_size.width(),
                                  source_size.height(),
                                  target_frame_->frameData(),
                                  target_frame_->stride(),
                                  target_size.width(),
                                  target_size.height(),
                                  target_rect.x(),
                                  target_rect.y(),
                                  target_rect.width(),
                                  target_rect.height(),
                                  libyuv::kFilterBox);

            updated_region->addRect(target_rect);
        }
    }

    return target_frame_.get();
}

//--------------------------------------------------------------------------------------------------
Rect ScaleReducer::scaledRect(const Rect& source_rect)
{
    int left = static_cast<int>(
        static_cast<double>(source_rect.left() * scale_x_) / 100.0);
    int top = static_cast<int>(
        static_cast<double>(source_rect.top() * scale_y_) / 100.0);
    int right = static_cast<int>(
        static_cast<double>(source_rect.right() * scale_x_) / 100.0);
    int bottom = static_cast<int>(
        static_cast<double>(source_rect.bottom() * scale_y_) / 100.0);

    return Rect::makeLTRB(left - 1, top - 1, right + 2, bottom + 2);
}

} // namespace base
