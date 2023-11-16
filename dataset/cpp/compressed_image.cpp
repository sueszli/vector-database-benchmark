// Aseprite Document Library
// Copyright (c) 2001-2015 David Capello
//
// This file is released under the terms of the MIT license.
// Read LICENSE.txt for more information.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "doc/compressed_image.h"

#include "doc/primitives.h"

namespace doc {

CompressedImage::CompressedImage(const Image* image, bool diffColors)
  : m_image(image)
{
  color_t c1, c2, mask = image->maskColor();

  for (int y=0; y<image->height(); ++y) {
    Scanline scanline(y);

    for (int x=0; x<image->width(); ) {
      c1 = get_pixel(image, x, y);
      if (c1 != mask) {
        scanline.color = c1;
        scanline.x = x;

        for (++x; x<image->width(); ++x) {
          c2 = get_pixel(image, x, y);

          if ((diffColors && c1 != c2) ||
              (!diffColors && c2 == mask))
            break;
        }

        scanline.w = x - scanline.x;
        m_scanlines.push_back(scanline);
      }
      else
        ++x;
    }
  }
}

} // namespace doc
