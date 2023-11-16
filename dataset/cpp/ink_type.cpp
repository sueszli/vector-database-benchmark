// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/tools/ink_type.h"

namespace app {
namespace tools {

std::string ink_type_to_string(InkType inkType)
{
  switch (inkType) {
    case tools::InkType::SIMPLE: return "Simple Ink";
    case tools::InkType::ALPHA_COMPOSITING: return "Alpha Compositing";
    case tools::InkType::COPY_COLOR: return "Copy Color+Alpha";
    case tools::InkType::LOCK_ALPHA: return "Lock Alpha";
    case tools::InkType::SHADING: return "Shading";
  }
  return "Unknown";
}

std::string ink_type_to_string_id(InkType inkType)
{
  switch (inkType) {
    case tools::InkType::SIMPLE: return "simple";
    case tools::InkType::ALPHA_COMPOSITING: return "alpha_compositing";
    case tools::InkType::COPY_COLOR: return "copy_color";
    case tools::InkType::LOCK_ALPHA: return "lock_alpha";
    case tools::InkType::SHADING: return "shading";
  }
  return "unknown";
}

InkType string_id_to_ink_type(const std::string& s)
{
  if (s == "simple") return tools::InkType::SIMPLE;
  if (s == "alpha_compositing") return tools::InkType::ALPHA_COMPOSITING;
  if (s == "copy_color") return tools::InkType::COPY_COLOR;
  if (s == "lock_alpha") return tools::InkType::LOCK_ALPHA;
  if (s == "shading") return tools::InkType::SHADING;
  return tools::InkType::DEFAULT;
}

} // namespace tools
} // namespace app
