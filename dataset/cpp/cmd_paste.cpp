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
#include "app/ui/input_chain.h"

namespace app {

class PasteCommand : public Command {
public:
  PasteCommand();
  Command* clone() const override { return new PasteCommand(*this); }

protected:
  bool onEnabled(Context* ctx) override;
  void onExecute(Context* ctx) override;
};

PasteCommand::PasteCommand()
  : Command("Paste",
            "Paste",
            CmdUIOnlyFlag)
{
}

bool PasteCommand::onEnabled(Context* ctx)
{
  return App::instance()->inputChain().canPaste(ctx);
}

void PasteCommand::onExecute(Context* ctx)
{
  App::instance()->inputChain().paste(ctx);
}

Command* CommandFactory::createPasteCommand()
{
  return new PasteCommand;
}

} // namespace app
