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
#include "app/ui/color_button.h"
#include "base/bind.h"
#include "app/commands/command.h"
#include "app/commands/filters/filter_manager_impl.h"
#include "app/commands/filters/filter_window.h"
#include "app/context.h"
#include "filters/invert_color_filter.h"
#include "app/ini_file.h"
#include "app/modules/gui.h"
#include "doc/image.h"
#include "doc/mask.h"
#include "doc/sprite.h"
#include "ui/button.h"
#include "ui/label.h"
#include "ui/slider.h"
#include "ui/widget.h"
#include "ui/window.h"

namespace app {

static const char* ConfigSection = "InvertColor";

class InvertColorWindow : public FilterWindow {
public:
  InvertColorWindow(FilterManagerImpl& filterMgr)
    : FilterWindow("Invert Color", ConfigSection, &filterMgr,
                   WithChannelsSelector,
                   WithoutTiledCheckBox)
  {
  }
};

class InvertColorCommand : public Command {
public:
  InvertColorCommand();
  Command* clone() const override { return new InvertColorCommand(*this); }

protected:
  bool onEnabled(Context* context) override;
  void onExecute(Context* context) override;
};

InvertColorCommand::InvertColorCommand()
  : Command("InvertColor",
            "Invert Color",
            CmdRecordableFlag)
{
}

bool InvertColorCommand::onEnabled(Context* context)
{
  return context->checkFlags(ContextFlags::ActiveDocumentIsWritable |
                             ContextFlags::HasActiveSprite);
}

void InvertColorCommand::onExecute(Context* context)
{
  InvertColorFilter filter;
  FilterManagerImpl filterMgr(context, &filter);
  filterMgr.setTarget(TARGET_RED_CHANNEL |
                      TARGET_GREEN_CHANNEL |
                      TARGET_BLUE_CHANNEL |
                      TARGET_GRAY_CHANNEL);

  InvertColorWindow window(filterMgr);
  window.doModal();
}

Command* CommandFactory::createInvertColorCommand()
{
  return new InvertColorCommand;
}

} // namespace app
