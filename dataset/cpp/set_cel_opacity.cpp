// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_cel_opacity.h"

#include "app/document.h"
#include "doc/cel.h"
#include "doc/document_event.h"

namespace app {
namespace cmd {

using namespace doc;

SetCelOpacity::SetCelOpacity(std::shared_ptr<Cel> cel, int opacity)
  : WithCel(cel)
  , m_oldOpacity(cel->opacity())
  , m_newOpacity(opacity)
{
}

void SetCelOpacity::onExecute()
{
  cel()->setOpacity(m_newOpacity);
  cel()->incrementVersion();
}

void SetCelOpacity::onUndo()
{
  cel()->setOpacity(m_oldOpacity);
  cel()->incrementVersion();
}

void SetCelOpacity::onFireNotifications()
{
  auto cel = this->cel();
  DocumentEvent ev(cel->document());
  ev.sprite(cel->sprite());
  ev.cel(cel);
  cel->document()->notifyObservers<DocumentEvent&>(&DocumentObserver::onCelOpacityChange, ev);
}

} // namespace cmd
} // namespace app
