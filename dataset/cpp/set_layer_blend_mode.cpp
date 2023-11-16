// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_layer_blend_mode.h"

#include "doc/document.h"
#include "doc/document_event.h"
#include "doc/layer.h"
#include "doc/sprite.h"

namespace app {
namespace cmd {

SetLayerBlendMode::SetLayerBlendMode(LayerImage* layer, BlendMode blendMode)
  : WithLayer(layer)
  , m_oldBlendMode(layer->blendMode())
  , m_newBlendMode(blendMode)
{
}

void SetLayerBlendMode::onExecute()
{
  static_cast<LayerImage*>(layer())->setBlendMode(m_newBlendMode);
  layer()->incrementVersion();
}

void SetLayerBlendMode::onUndo()
{
  static_cast<LayerImage*>(layer())->setBlendMode(m_oldBlendMode);
  layer()->incrementVersion();
}

void SetLayerBlendMode::onFireNotifications()
{
  Layer* layer = this->layer();
  doc::Document* doc = layer->sprite()->document();
  DocumentEvent ev(doc);
  ev.sprite(layer->sprite());
  ev.layer(layer);
  doc->notifyObservers<DocumentEvent&>(&DocumentObserver::onLayerBlendModeChange, ev);
}

} // namespace cmd
} // namespace app
