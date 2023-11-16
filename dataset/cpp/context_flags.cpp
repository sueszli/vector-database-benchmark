// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/context_flags.h"

#include "app/context.h"
#include "app/document.h"
#include "app/modules/editors.h"
#include "app/ui/editor/editor.h"
#include "doc/cel.h"
#include "doc/layer.h"
#include "doc/site.h"
#include "doc/sprite.h"

namespace app {

ContextFlags::ContextFlags()
{
  m_flags = 0;
}

void ContextFlags::update(Context* context)
{
  Site site = context->activeSite();
  Document* document = static_cast<Document*>(site.document());

  m_flags = 0;

  if (document) {
    m_flags |= HasActiveDocument;

    if (document->lock(Document::ReadLock, 0)) {
      m_flags |= ActiveDocumentIsReadable;

      if (document->isMaskVisible())
        m_flags |= HasVisibleMask;

      updateFlagsFromSite(site);

      if (document->lockToWrite(0))
        m_flags |= ActiveDocumentIsWritable;

      document->unlock();
    }

    // TODO this is a hack, try to find a better design to handle this
    // "moving pixels" state.
    if (current_editor &&
        current_editor->document() == document &&
        current_editor->isMovingPixels()) {
      // Flags enabled when we are in MovingPixelsState
      m_flags |=
        HasVisibleMask |
        ActiveDocumentIsReadable |
        ActiveDocumentIsWritable;

      updateFlagsFromSite(current_editor->getSite());
    }
  }
}

void ContextFlags::updateFlagsFromSite(const Site& site)
{
  const Sprite* sprite = site.sprite();
  if (!sprite)
    return;

  m_flags |= HasActiveSprite;

  if (sprite->backgroundLayer())
    m_flags |= HasBackgroundLayer;

  const Layer* layer = site.layer();
  frame_t frame = site.frame();
  if (!layer)
    return;

  m_flags |= HasActiveLayer;

  if (layer->isBackground())
    m_flags |= ActiveLayerIsBackground;

  if (layer->isVisible())
    m_flags |= ActiveLayerIsVisible;

  if (layer->isEditable())
    m_flags |= ActiveLayerIsEditable;

  if (layer->isImage()) {
    m_flags |= ActiveLayerIsImage;

    if (auto cel = layer->cel(frame)) {
      m_flags |= HasActiveCel;

      if (cel->image())
        m_flags |= HasActiveImage;
    }
  }
}

} // namespace app
