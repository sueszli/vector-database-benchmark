// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_frame_tag_range.h"

#include "doc/frame_tag.h"

namespace app {
namespace cmd {

SetFrameTagRange::SetFrameTagRange(FrameTag* tag, frame_t from, frame_t to)
  : WithFrameTag(tag)
  , m_oldFrom(tag->fromFrame())
  , m_oldTo(tag->toFrame())
  , m_newFrom(from)
  , m_newTo(to)
{
}

void SetFrameTagRange::onExecute()
{
  frameTag()->setFrameRange(m_newFrom, m_newTo);
  frameTag()->incrementVersion();
}

void SetFrameTagRange::onUndo()
{
  frameTag()->setFrameRange(m_oldFrom, m_oldTo);
  frameTag()->incrementVersion();
}

} // namespace cmd
} // namespace app
