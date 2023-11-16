// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/cmd/clear_cel.h"

#include "app/cmd/clear_image.h"
#include "app/cmd/remove_cel.h"
#include "app/document.h"
#include "doc/cel.h"
#include "doc/layer.h"

namespace app {
namespace cmd {

using namespace doc;

ClearCel::ClearCel(std::shared_ptr<Cel> cel)
  : WithCel(cel)
{
  app::Document* doc = static_cast<app::Document*>(cel->document());

  if (cel->layer()->isBackground()) {
    Image* image = cel->image();
    ASSERT(image);
    if (image)
      m_seq.add(new cmd::ClearImage(image,
          doc->bgColor(cel->layer())));
  }
  else {
    m_seq.add(new cmd::RemoveCel(cel));
  }
}

void ClearCel::onExecute()
{
  m_seq.execute(context());
}

void ClearCel::onUndo()
{
  m_seq.undo();
}

void ClearCel::onRedo()
{
  m_seq.redo();
}

} // namespace cmd
} // namespace app
