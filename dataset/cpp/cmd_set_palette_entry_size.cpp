// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/app.h"
#include "app/commands/command.h"
#include "app/commands/params.h"
#include "app/ui/color_bar.h"

namespace app {

class SetPaletteEntrySizeCommand : public Command {
public:
  SetPaletteEntrySizeCommand();
  Command* clone() const override { return new SetPaletteEntrySizeCommand(*this); }

protected:
  void onLoadParams(const Params& params) override;
  bool onChecked(Context* context) override;
  void onExecute(Context* context) override;

private:
  int m_size;
};

SetPaletteEntrySizeCommand::SetPaletteEntrySizeCommand()
  : Command("SetPaletteEntrySize",
            "Set Palette Entry Size",
            CmdUIOnlyFlag)
  , m_size(7)
{
}

void SetPaletteEntrySizeCommand::onLoadParams(const Params& params)
{
  if (params.has_param("size"))
    m_size = params.get_as<int>("size");
}

bool SetPaletteEntrySizeCommand::onChecked(Context* context)
{
  return (ColorBar::instance()->getPaletteView()->getBoxSize() == m_size);
}

void SetPaletteEntrySizeCommand::onExecute(Context* context)
{
  ColorBar::instance()->getPaletteView()->setBoxSize(m_size);
}

Command* CommandFactory::createSetPaletteEntrySizeCommand()
{
  return new SetPaletteEntrySizeCommand;
}

} // namespace app
