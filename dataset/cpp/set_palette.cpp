// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/set_palette.h"

#include "base/serialization.h"
#include "doc/palette.h"
#include "doc/sprite.h"

namespace app {
namespace cmd {

using namespace doc;

SetPalette::SetPalette(Sprite* sprite, frame_t frame, const Palette* newPalette)
  : WithSprite(sprite)
  , m_frame(frame)
{
  const Palette* curPalette = sprite->palette(frame);

  m_oldNColors = curPalette->size();
  m_newNColors = newPalette->size();

  // Check differences between current sprite palette and the new one
  m_from = m_to = -1;
  int diffs = curPalette->countDiff(newPalette, &m_from, &m_to);
  (void)diffs;
  ASSERT(diffs > 0);

  if (m_from >= 0 && m_to >= m_from) {
    int oldColors = MIN(m_to+1, m_oldNColors)-m_from;
    if (oldColors > 0)
      m_oldColors.resize(oldColors);

    int newColors = MIN(m_to+1, m_newNColors)-m_from;
    if (newColors > 0)
      m_newColors.resize(newColors);

    for (size_t i=0; i<size_t(m_to-m_from+1); ++i) {
      if (i < m_oldColors.size())
        m_oldColors[i] = curPalette->getEntry(m_from+i);

      if (i < m_newColors.size())
        m_newColors[i] = newPalette->getEntry(m_from+i);
    }
  }
}

void SetPalette::onExecute()
{
  Sprite* sprite = this->sprite();
  Palette* palette = sprite->palette(m_frame);
  palette->resize(m_newNColors);

  for (size_t i=0; i<m_newColors.size(); ++i)
    palette->setEntry(m_from+i, m_newColors[i]);

  palette->incrementVersion();
}

void SetPalette::onUndo()
{
  Sprite* sprite = this->sprite();
  Palette* palette = sprite->palette(m_frame);
  palette->resize(m_oldNColors);

  for (size_t i=0; i<m_oldColors.size(); ++i)
    palette->setEntry(m_from+i, m_oldColors[i]);

  palette->incrementVersion();
}

} // namespace cmd
} // namespace app
