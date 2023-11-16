////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "ClearHDRBufferNode.hpp"

#include "HDRBuffer.hpp"

#include <VistaMath/VistaBoundingBox.h>
#include <limits>
#include <utility>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

ClearHDRBufferNode::ClearHDRBufferNode(std::shared_ptr<HDRBuffer> hdrBuffer)
    : mHDRBuffer(std::move(hdrBuffer)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ClearHDRBufferNode::ClearHDRBufferNode::Do() {
  mHDRBuffer->clear();
  mHDRBuffer->bind();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ClearHDRBufferNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float min(std::numeric_limits<float>::lowest());
  float max(std::numeric_limits<float>::max());

  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
