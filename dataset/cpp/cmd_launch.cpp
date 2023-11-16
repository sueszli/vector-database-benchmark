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
#include "app/commands/params.h"
#include "app/launcher.h"
#include "base/fs.h"

namespace app {

class LaunchCommand : public Command {
public:
  LaunchCommand();
  Command* clone() const override { return new LaunchCommand(*this); }

protected:
  void onLoadParams(const Params& params) override;
  void onExecute(Context* context) override;

private:
  enum Type { Url };

  Type m_type;
  std::string m_path;
};

LaunchCommand::LaunchCommand()
  : Command("Launch",
            "Launch",
            CmdUIOnlyFlag)
  , m_type(Url)
  , m_path("")
{
}

void LaunchCommand::onLoadParams(const Params& params)
{
  m_path = params.get("path");

  if (m_type == Url && !m_path.empty() && m_path[0] == '/') {
    m_path = WEBSITE + m_path.substr(1);
  }
}

void LaunchCommand::onExecute(Context* context)
{
  switch (m_type) {

    case Url:
      launcher::open_url(m_path);
      break;

  }
}

Command* CommandFactory::createLaunchCommand()
{
  return new LaunchCommand;
}

} // namespace app
