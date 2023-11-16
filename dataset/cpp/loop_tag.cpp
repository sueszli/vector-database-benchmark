// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/loop_tag.h"

#include "doc/sprite.h"
#include "doc/frame_tag.h"

namespace app {

const char* kLoopTagName = "Loop";

doc::FrameTag* get_animation_tag(const doc::Sprite* sprite, doc::frame_t frame)
{
  doc::FrameTag* tag = sprite->frameTags().innerTag(frame);
  if (!tag)
    tag = get_loop_tag(sprite);
  return tag;
}

doc::FrameTag* get_loop_tag(const doc::Sprite* sprite)
{
  // Get tag with special "Loop" name
  for (doc::FrameTag* tag : sprite->frameTags())
    if (tag->name() == kLoopTagName)
      return tag;

  return nullptr;
}

doc::FrameTag* create_loop_tag(doc::frame_t from, doc::frame_t to)
{
  doc::FrameTag* tag = new doc::FrameTag(from, to);
  tag->setName(kLoopTagName);
  return tag;
}

} // app
