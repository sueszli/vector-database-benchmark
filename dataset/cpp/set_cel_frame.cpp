// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_cel_frame.h"

#include "doc/cel.h"
#include "doc/document.h"
#include "doc/document_event.h"
#include "doc/layer.h"
#include "doc/sprite.h"

namespace app {
namespace cmd {

using namespace doc;

SetCelFrame::SetCelFrame(std::shared_ptr<Cel> cel, frame_t frame)
  : WithCel(cel)
  , m_oldFrame(cel->frame())
  , m_newFrame(frame)
{
}

void SetCelFrame::onExecute()
{
  auto cel = this->cel();
  cel->layer()->moveCel(cel, m_newFrame);
  cel->incrementVersion();
}

void SetCelFrame::onUndo()
{
  auto cel = this->cel();
  cel->layer()->moveCel(cel, m_oldFrame);
  cel->incrementVersion();
}

void SetCelFrame::onFireNotifications()
{
  auto cel = this->cel();
  doc::Document* doc = cel->sprite()->document();
  DocumentEvent ev(doc);
  ev.sprite(cel->layer()->sprite());
  ev.layer(cel->layer());
  ev.cel(cel);
  ev.frame(cel->frame());
  doc->notifyObservers<DocumentEvent&>(&DocumentObserver::onCelFrameChanged, ev);
}

} // namespace cmd
} // namespace app
