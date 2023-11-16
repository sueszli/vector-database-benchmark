// Aseprite
// Copyright (C) 2001-2015  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ui/ui.h"

#include "app/app.h"
#include "app/commands/command.h"
#include "app/commands/params.h"
#include "app/context.h"
#include "app/ui/main_window.h"
#include "app/ui/timeline.h"

namespace app {

class TimelineCommand : public Command {
public:
  TimelineCommand();
  Command* clone() const override { return new TimelineCommand(*this); }

protected:
  void onLoadParams(const Params& params) override;
  void onExecute(Context* context) override;
  bool onChecked(Context* ctx) override;

  bool m_open;
  bool m_close;
  bool m_switch;
};

TimelineCommand::TimelineCommand()
  : Command("Timeline",
            "Switch Timeline",
            CmdUIOnlyFlag)
{
  m_open = true;
  m_close = false;
  m_switch = false;
}

void TimelineCommand::onLoadParams(const Params& params)
{
  std::string open_str = params.get("open");
  if (open_str == "true") m_open = true;
  else m_open = false;

  std::string close_str = params.get("close");
  if (close_str == "true") m_close = true;
  else m_close = false;

  std::string switch_str = params.get("switch");
  if (switch_str == "true") m_switch = true;
  else m_switch = false;
}

void TimelineCommand::onExecute(Context* context)
{
  bool visible = App::instance()->mainWindow()->getTimelineVisibility();
  bool newVisible = visible;

  if (m_switch)
    newVisible = !visible;
  else if (m_close)
    newVisible = false;
  else if (m_open)
    newVisible = true;

  if (visible != newVisible)
    App::instance()->mainWindow()->setTimelineVisibility(newVisible);
}

bool TimelineCommand::onChecked(Context* ctx) {
  MainWindow* mainWin = App::instance()->mainWindow();
  if (!mainWin)
    return false;

  Timeline* timelineWin = mainWin->getTimeline();
  return (timelineWin && timelineWin->isVisible());
}

Command* CommandFactory::createTimelineCommand()
{
  return new TimelineCommand;
}

} // namespace app
