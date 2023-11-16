// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "app/commands/command.h"
#include "app/context.h"
#include "app/context_access.h"
#include "app/document.h"
#include "app/launcher.h"

namespace app {

class OpenInFolderCommand : public Command {
public:
  OpenInFolderCommand();
  Command* clone() const override { return new OpenInFolderCommand(*this); }

protected:
  bool onEnabled(Context* context) override;
  void onExecute(Context* context) override;
};

OpenInFolderCommand::OpenInFolderCommand()
  : Command("OpenInFolder",
            "Open In Folder",
            CmdUIOnlyFlag)
{
}

bool OpenInFolderCommand::onEnabled(Context* context)
{
  const ContextReader reader(context);
  return
    reader.document() &&
    reader.document()->isAssociatedToFile();
}

void OpenInFolderCommand::onExecute(Context* context)
{
  launcher::open_folder(context->activeDocument()->filename());
}

Command* CommandFactory::createOpenInFolderCommand()
{
  return new OpenInFolderCommand;
}

} // namespace app
