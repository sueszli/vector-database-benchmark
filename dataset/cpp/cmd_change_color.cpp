// Aseprite
// Copyright (C) 2001-2016  David Capello
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string>

#include "app/app.h"
#include "app/commands/command.h"
#include "app/commands/params.h"
#include "app/modules/palettes.h"
#include "app/ui/color_bar.h"
#include "doc/palette.h"

namespace app {

class ChangeColorCommand : public Command
{
  enum Change {
    None,
    IncrementIndex,
    DecrementIndex,
  };

  /**
   * True means "change background color", false the foreground color.
   */
  bool m_background;

  Change m_change;

public:
  ChangeColorCommand();

protected:
  void onLoadParams(const Params& params) override;
  void onExecute(Context* context) override;
  std::string onGetFriendlyName() const override;
};

ChangeColorCommand::ChangeColorCommand()
  : Command("ChangeColor",
            "Change Color",
            CmdUIOnlyFlag)
{
  m_background = false;
  m_change = None;
}

void ChangeColorCommand::onLoadParams(const Params& params)
{
  std::string target = params.get("target");
  if (target == "foreground") m_background = false;
  else if (target == "background") m_background = true;

  std::string change = params.get("change");
  if (change == "increment-index") m_change = IncrementIndex;
  else if (change == "decrement-index") m_change = DecrementIndex;
}

void ChangeColorCommand::onExecute(Context* context)
{
  ColorBar* colorbar = ColorBar::instance();
  app::Color color = m_background ? colorbar->getBgColor():
                                    colorbar->getFgColor();

  switch (m_change) {
    case None:
      // do nothing
      break;
    case IncrementIndex: {
      int index = color.getIndex();
      if (index < get_current_palette()->size()-1)
        color = app::Color::fromIndex(index+1);
      break;
    }
    case DecrementIndex: {
      int index = color.getIndex();
      if (index > 0)
        color = app::Color::fromIndex(index-1);
      break;
    }
  }

  if (m_background)
    colorbar->setBgColor(color);
  else
    colorbar->setFgColor(color);
}

std::string ChangeColorCommand::onGetFriendlyName() const
{
  std::string text = "Color";

  switch (m_change) {
    case None:
      return text;
    case IncrementIndex:
      text += ": Increment";
      break;
    case DecrementIndex:
      text += ": Decrement";
      break;
  }

  if (m_background)
    text += " Background Index";
  else
    text += " Foreground Index";

  return text;
}

Command* CommandFactory::createChangeColorCommand()
{
  return new ChangeColorCommand;
}

} // namespace app
