// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/clear_rect.h"

#include "app/document.h"
#include "doc/cel.h"
#include "doc/image.h"
#include "doc/layer.h"
#include "doc/primitives.h"

namespace app {
namespace cmd {

using namespace doc;

ClearRect::ClearRect(std::shared_ptr<Cel> cel, const gfx::Rect& bounds)
{
  app::Document* doc = static_cast<app::Document*>(cel->document());

  Image* image = (cel ? cel->image(): NULL);
  if (!image)
    return;

  m_offsetX = bounds.x - cel->x();
  m_offsetY = bounds.y - cel->y();

  gfx::Rect bounds2 =
    image->bounds().createIntersection(
      gfx::Rect(
        m_offsetX, m_offsetY,
        bounds.w, bounds.h));
  if (bounds.isEmpty())
    return;

  m_dstImage.reset(new WithImage(image));
  m_bgcolor = doc->bgColor(cel->layer());

  m_copy.reset(crop_image(image,
      bounds2.x, bounds2.y, bounds2.w, bounds2.h, m_bgcolor));
}

void ClearRect::onExecute()
{
  m_seq.execute(context());
  if (m_dstImage)
    clear();
}

void ClearRect::onUndo()
{
  if (m_dstImage)
    restore();
  m_seq.undo();
}

void ClearRect::onRedo()
{
  m_seq.redo();
  if (m_dstImage)
    clear();
}

void ClearRect::clear()
{
  fill_rect(m_dstImage->image(),
            m_offsetX, m_offsetY,
            m_offsetX + m_copy->width() - 1,
            m_offsetY + m_copy->height() - 1,
            m_bgcolor);
}

void ClearRect::restore()
{
  copy_image(m_dstImage->image(), m_copy.get(), m_offsetX, m_offsetY);
}

} // namespace cmd
} // namespace app
