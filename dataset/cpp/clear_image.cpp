// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/clear_image.h"

#include "app/document.h"
#include "doc/image.h"
#include "doc/primitives.h"

namespace app {
namespace cmd {

using namespace doc;

ClearImage::ClearImage(Image* image, color_t color)
  : WithImage(image)
  , m_color(color)
{
}

void ClearImage::onExecute()
{
  Image* image = this->image();

  ASSERT(!m_copy);
  m_copy.reset(Image::createCopy(image));
  clear_image(image, m_color);

  image->incrementVersion();
}

void ClearImage::onUndo()
{
  Image* image = this->image();

  copy_image(image, m_copy.get());
  m_copy.reset();

  image->incrementVersion();
}

} // namespace cmd
} // namespace app
