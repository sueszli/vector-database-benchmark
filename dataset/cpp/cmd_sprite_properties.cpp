// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/color.h"
#include "app/commands/command.h"
#include "app/context_access.h"
#include "app/document_api.h"
#include "app/modules/gui.h"
#include "app/ui/color_button.h"
#include "app/transaction.h"
#include "base/bind.h"
#include "base/mem_utils.h"
#include "doc/image.h"
#include "doc/palette.h"
#include "doc/sprite.h"
#include "ui/ui.h"

#include "sprite_properties.xml.h"

#include <cstdio>

namespace app {

using namespace ui;

class SpritePropertiesCommand : public Command {
public:
  SpritePropertiesCommand();
  Command* clone() const override { return new SpritePropertiesCommand(*this); }

protected:
  bool onEnabled(Context* context) override;
  void onExecute(Context* context) override;
};

SpritePropertiesCommand::SpritePropertiesCommand()
  : Command("SpriteProperties",
            "Sprite Properties",
            CmdUIOnlyFlag)
{
}

bool SpritePropertiesCommand::onEnabled(Context* context)
{
  return context->checkFlags(ContextFlags::ActiveDocumentIsWritable |
                             ContextFlags::HasActiveSprite);
}

void SpritePropertiesCommand::onExecute(Context* context)
{
  std::string imgtype_text;
  char buf[256];
  ColorButton* color_button = NULL;

  // Load the window widget
  app::gen::SpriteProperties window;

  // Get sprite properties and fill frame fields
  {
    const ContextReader reader(context);
    const Document* document(reader.document());
    const Sprite* sprite(reader.sprite());

    // Update widgets values
    switch (sprite->pixelFormat()) {
      case IMAGE_RGB:
        imgtype_text = "RGB";
        break;
      case IMAGE_GRAYSCALE:
        imgtype_text = "Grayscale";
        break;
      case IMAGE_INDEXED:
        std::sprintf(buf, "Indexed (%d colors)", sprite->palette(0)->size());
        imgtype_text = buf;
        break;
      default:
        ASSERT(false);
        imgtype_text = "Unknown";
        break;
    }

    // Filename
    window.name()->setText(document->filename());

    // Color mode
    window.type()->setText(imgtype_text.c_str());

    // Sprite size (width and height)
    window.size()->setTextf(
      "%dx%d (%s)",
      sprite->width(),
      sprite->height(),
      base::get_pretty_memory_size(sprite->getMemSize()).c_str());

    // How many frames
    window.frames()->setTextf("%d", (int)sprite->totalFrames());

    if (sprite->pixelFormat() == IMAGE_INDEXED) {
      color_button = new ColorButton(app::Color::fromIndex(sprite->transparentColor()),
                                     IMAGE_INDEXED);

      window.transparentColorPlaceholder()->addChild(color_button);
    }
    else {
      window.transparentColorPlaceholder()->addChild(new Label("(only for indexed images)"));
    }
  }

  window.remapWindow();
  window.centerWindow();

  load_window_pos(&window, "SpriteProperties");
  window.setVisible(true);
  window.openWindowInForeground();

  if (window.closer() == window.ok()) {
    if (color_button) {
      ContextWriter writer(context);
      Sprite* sprite(writer.sprite());

      // If the transparent color index has changed, we update the
      // property in the sprite.
      int index = color_button->getColor().getIndex();
      if (color_t(index) != sprite->transparentColor()) {
        Transaction transaction(writer.context(), "Set Transparent Color");
        DocumentApi api = writer.document()->getApi(transaction);
        api.setSpriteTransparentColor(sprite, index);
        transaction.commit();

        update_screen_for_document(writer.document());
      }
    }
  }

  save_window_pos(&window, "SpriteProperties");
}

Command* CommandFactory::createSpritePropertiesCommand()
{
  return new SpritePropertiesCommand;
}

} // namespace app
