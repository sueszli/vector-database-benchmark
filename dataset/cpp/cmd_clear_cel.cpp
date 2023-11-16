// Aseprite
// Copyright (C) 2001-2016  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/app.h"
#include "app/commands/command.h"
#include "app/context_access.h"
#include "app/document_api.h"
#include "app/modules/gui.h"
#include "app/transaction.h"
#include "app/ui/status_bar.h"
#include "app/ui/timeline.h"
#include "doc/cel.h"
#include "doc/layer.h"
#include "doc/sprite.h"

namespace app {

class ClearCelCommand : public Command {
public:
  ClearCelCommand();
  Command* clone() const override { return new ClearCelCommand(*this); }

protected:
  bool onEnabled(Context* context) override;
  void onExecute(Context* context) override;
};

ClearCelCommand::ClearCelCommand()
  : Command("ClearCel",
            "Clear Cel",
            CmdRecordableFlag)
{
}

bool ClearCelCommand::onEnabled(Context* context)
{
  return context->checkFlags(ContextFlags::ActiveDocumentIsWritable);
}

void ClearCelCommand::onExecute(Context* context)
{
  ContextWriter writer(context);
  Document* document(writer.document());
  bool nonEditableLayers = false;
  {
    Transaction transaction(writer.context(), "Clear Cel");

    // TODO the range of selected frames should be in doc::Site.
    auto range = App::instance()->timeline()->range();
    if (range.enabled()) {
      Sprite* sprite = writer.sprite();

      for (LayerIndex layerIdx = range.layerBegin(); layerIdx <= range.layerEnd(); ++layerIdx) {
        Layer* layer = sprite->indexToLayer(layerIdx);
        if (!layer->isImage())
          continue;

        LayerImage* layerImage = static_cast<LayerImage*>(layer);

        for (frame_t frame = range.frameEnd(),
               begin = range.frameBegin()-1;
             frame != begin;
             --frame) {
          if (layerImage->cel(frame)) {
            if (layerImage->isEditable())
              document->getApi(transaction).clearCel(layerImage, frame);
            else
              nonEditableLayers = true;
          }
        }
      }
    }
    else if (writer.cel()) {
      if (writer.layer()->isEditable())
        document->getApi(transaction).clearCel(writer.cel());
      else
        nonEditableLayers = true;
    }

    transaction.commit();
  }

  if (nonEditableLayers)
    StatusBar::instance()->showTip(1000,
      "There are locked layers");

  update_screen_for_document(document);
}

Command* CommandFactory::createClearCelCommand()
{
  return new ClearCelCommand;
}

} // namespace app
