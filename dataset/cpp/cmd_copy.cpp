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

class CopyCommand : public Command {
public:
  CopyCommand();
  Command* clone() const override { return new CopyCommand(*this); }

protected:
  bool onEnabled(Context* ctx) override;
  void onExecute(Context* ctx) override;
};

CopyCommand::CopyCommand()
  : Command("Copy",
            "Copy",
            CmdUIOnlyFlag)
{
}

bool CopyCommand::onEnabled(Context* ctx)
{
  return App::instance()->inputChain().canCopy(ctx);
}

void CopyCommand::onExecute(Context* ctx)
{
  App::instance()->inputChain().copy(ctx);
}

Command* CommandFactory::createCopyCommand()
{
  return new CopyCommand;
}

} // namespace app
