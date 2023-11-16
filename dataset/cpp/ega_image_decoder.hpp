/* Copyright (C) 2016, Nikolai Wuttke. All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "assets/byte_buffer.hpp"
#include "assets/palette.hpp"
#include "base/image.hpp"
#include "data/game_traits.hpp"


namespace rigel::assets
{

data::PixelBuffer decodeSimplePlanarEgaBuffer(
  ByteBufferCIter begin,
  ByteBufferCIter end,
  const data::Palette16& palette);


data::Image loadTiledImage(
  ByteBufferCIter begin,
  ByteBufferCIter end,
  std::size_t widthInTiles,
  const data::Palette16& palette,
  data::TileImageType type);


inline data::Image loadTiledImage(
  const ByteBuffer& data,
  std::size_t widthInTiles,
  const data::Palette16& palette,
  const data::TileImageType type = data::TileImageType::Unmasked)
{
  return loadTiledImage(data.begin(), data.end(), widthInTiles, palette, type);
}

data::Image loadTiledFontBitmap(
  ByteBufferCIter begin,
  ByteBufferCIter end,
  std::size_t widthInTiles);

} // namespace rigel::assets
