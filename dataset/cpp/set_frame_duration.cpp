// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_frame_duration.h"

#include "app/document.h"
#include "doc/document_event.h"
#include "doc/sprite.h"

namespace app {
namespace cmd {

SetFrameDuration::SetFrameDuration(Sprite* sprite, frame_t frame, int duration)
  : WithSprite(sprite)
  , m_frame(frame)
  , m_oldDuration(sprite->frameDuration(frame))
  , m_newDuration(duration)
{
}

void SetFrameDuration::onExecute()
{
  sprite()->setFrameDuration(m_frame, m_newDuration);
  sprite()->incrementVersion();
}

void SetFrameDuration::onUndo()
{
  sprite()->setFrameDuration(m_frame, m_oldDuration);
  sprite()->incrementVersion();
}

void SetFrameDuration::onFireNotifications()
{
  Sprite* sprite = this->sprite();
  doc::Document* doc = sprite->document();
  DocumentEvent ev(doc);
  ev.sprite(sprite);
  ev.frame(m_frame);
  doc->notifyObservers<DocumentEvent&>(&DocumentObserver::onFrameDurationChanged, ev);
}

} // namespace cmd
} // namespace app
