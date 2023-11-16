/* Copyright (C) 2016, Nikolai Wuttke. All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "duke_script_loader.hpp"

#include "base/string_utils.hpp"
#include "base/warnings.hpp"
#include "data/game_traits.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>


// TODO:
//
// HELPTEXT <EP> <Level> Text - define hint globe text for Episode/level
//                              combination. Numbers are 1-based
// ETE - seems unused? Maybe Shareware version only (appears only in
//       ORDERTXT.MNI)
//
// SETCURRENTPAGE - freezes animations/current displayed frame
// SETKEYS <raw byte list of scan-codes> -
//            Sets up hot-keys for menu actions in the main menu. In the
//            Quit_Select, it sets up the 'Y' and 'N' keys. Ignored for now,
//            we just hardcode those keys for Quit_Select.


namespace rigel::assets
{

using namespace std;
using namespace data::script;
using data::GameTraits;


namespace
{

void skipWhiteSpace(istream& sourceStream)
{
  while (!sourceStream.eof() && std::isspace(sourceStream.peek()))
  {
    sourceStream.get();
  }
}


bool isCommand(string_view line)
{
  return strings::startsWith(line, "//");
}


void stripCommandPrefix(string& line)
{
  strings::trimLeft(line, "/");
}


template <typename Callable>
void parseScriptLines(
  istream& sourceTextStream,
  const string& endMarker,
  Callable consumeLine)
{
  skipWhiteSpace(sourceTextStream);
  for (string line; getline(sourceTextStream, line, '\n');)
  {
    strings::trim(line);
    if (isCommand(line))
    {
      strings::trimLeft(line, "/");
      if (line == endMarker)
      {
        return;
      }

      istringstream lineTextStream(line);
      string command;
      lineTextStream >> command;
      strings::trim(command);
      consumeLine(command, lineTextStream);
    }
  }

  throw invalid_argument(
    string("Missing end marker '" + endMarker + "' in Duke Script file"));
}


std::optional<Action>
  parseSingleActionCommand(const string& command, istream& lineTextStream)
{
  if (command == "FADEIN")
  {
    return Action{FadeIn{}};
  }
  else if (command == "FADEOUT")
  {
    return Action{FadeOut{}};
  }
  else if (command == "DELAY")
  {
    int amount = 0;
    lineTextStream >> amount;
    if (amount <= 0)
    {
      throw invalid_argument("Invalid DELAY command in Duke Script file");
    }
    return Action{Delay{amount}};
  }
  else if (command == "BABBLEON")
  {
    int duration = 0;
    lineTextStream >> duration;
    if (duration <= 0)
    {
      throw invalid_argument("Invalid BABBLEON command in Duke Script file");
    }
    return Action{AnimateNewsReporter{duration}};
  }
  else if (command == "BABBLEOFF")
  {
    return Action{StopNewsReporterAnimation{}};
  }
  else if (command == "NOSOUNDS")
  {
    return Action{DisableMenuFunctionality{}};
  }
  else if (command == "KEYS")
  {
    return Action{ShowKeyBindings{}};
  }
  else if (command == "GETNAMES")
  {
    int slot = 0;
    lineTextStream >> slot;
    if (slot < 0 || slot >= 8)
    {
      throw invalid_argument("Invalid GETNAMES command in Duke Script file");
    }
    return Action{ShowSaveSlots{slot}};
  }
  else if (command == "PAK")
  {
    // [P]ress [A]ny [K]ey - this is a shorthand for displaying actor nr. 146,
    // which is an image of the text "Press any key to continue".
    return Action{DrawSprite{0, 0, 146, 0}};
  }
  else if (command == "LOADRAW")
  {
    string imageName;
    lineTextStream >> imageName;
    strings::trim(imageName);
    if (imageName.empty())
    {
      throw invalid_argument("Invalid LOADRAW command in Duke Script file");
    }
    return Action{ShowFullScreenImage{imageName}};
  }
  else if (command == "Z")
  {
    int yPos = 0;
    lineTextStream >> yPos;
    return Action{ShowMenuSelectionIndicator{yPos}};
  }
  else if (command == "GETPAL")
  {
    string paletteFile;
    lineTextStream >> paletteFile;
    strings::trim(paletteFile);
    if (paletteFile.empty())
    {
      throw invalid_argument("Invalid LOADRAW command in Duke Script file");
    }
    return Action{SetPalette{paletteFile}};
  }
  else if (command == "WAIT")
  {
    return Action{WaitForUserInput{}};
  }
  else if (command == "SHIFTWIN")
  {
    return Action{EnableTextOffset{}};
  }
  else if (command == "EXITTODEMO")
  {
    return Action{EnableTimeOutToDemo{}};
  }
  else if (command == "TOGGS")
  {
    int xPos = 0;
    int count = 0;
    lineTextStream >> xPos;
    lineTextStream >> count;

    vector<SetupCheckBoxes::CheckBoxDefinition> definitions;
    for (int i = 0; i < count; ++i)
    {
      SetupCheckBoxes::CheckBoxDefinition definition{0, 0};
      lineTextStream >> definition.yPos;
      lineTextStream >> definition.id;

      definitions.emplace_back(definition);
    }

    return Action{SetupCheckBoxes{xPos, definitions}};
  }
  else if (command == "CENTERWINDOW")
  {
    int y = 0;
    int width = 0;
    int height = 0;
    lineTextStream >> y;
    lineTextStream >> height;
    lineTextStream >> width;

    return Action{ShowMessageBox{y, width, height}};
  }
  else if (command == "SKLINE")
  {
    return Action{DrawMessageBoxText{""}};
  }
  else if (command == "CWTEXT")
  {
    string messageLine;
    lineTextStream.get();
    getline(lineTextStream, messageLine, '\r');

    if (messageLine.empty())
    {
      throw invalid_argument("Corrupt Duke Script file");
    }

    strings::trimRight(messageLine);

    return Action{DrawMessageBoxText{std::move(messageLine)}};
  }
  else
  {
    assert(command != "END");
    static const unordered_set<string> notAllowedHere{
      "APAGE",
      "CENTERWINDOW",
      "CWTEXT",
      "MENU",
      "PAGESEND",
      "PAGESSTART",
      "SKLINE"};

    if (notAllowedHere.count(command) == 1)
    {
      throw invalid_argument(
        string("The command ") + command + " is not allowed in this context");
    }
  }

  return std::nullopt;
}


vector<Action> parseTextCommandWithBigText(
  const int x,
  const int y,
  const string& sourceText,
  const string::const_iterator bigTextMarkerIter)
{
  vector<Action> textActions;

  const auto numPrecedingCharacters =
    static_cast<int>(distance(sourceText.begin(), bigTextMarkerIter));
  if (numPrecedingCharacters > 0)
  {
    string regularTextPart(sourceText.begin(), bigTextMarkerIter);
    textActions.emplace_back(DrawText{x, y, regularTextPart});
  }

  const auto positionOffset =
    numPrecedingCharacters * GameTraits::menuFontCharacterBitmapSizeTiles.width;

  const auto colorIndex = static_cast<uint8_t>(*bigTextMarkerIter) - 0xF0;
  string bigTextPart(next(bigTextMarkerIter), sourceText.end());
  textActions.emplace_back(
    DrawBigText{x + positionOffset, y, colorIndex, move(bigTextPart)});

  return textActions;
}


Action parseDrawSpriteCommand(const int x, const int y, string_view source)
{
  if (source.size() < 5)
  {
    throw invalid_argument("Corrupt Duke Script file");
  }

  string actorNumberString(source.begin() + 1, source.begin() + 4);
  string frameNumberString(source.begin() + 4, source.begin() + 6);

  return {
    DrawSprite{x + 2, y + 1, stoi(actorNumberString), stoi(frameNumberString)}};
}


vector<Action> parseTextCommand(istream& lineTextStream)
{
  // They decided to pack a lot of different functionality into the XYTEXT
  // command, which makes parsing it a bit more involved. There are three
  // variants:
  //
  // 1. Draw normal text
  // 2. Draw sprite
  // 3. Draw big, colorized text (potentially with preceding normal text)
  //
  // Variant 1 is the default, where we just need to take the remainder of
  // the line and draw it at the specified position.  The other two are
  // indicated by special 'markup' bytes in the text. If the text starts with
  // the byte 0xEF, the remaining text is actually interpreted as a sequence
  // of 2 numbers. The first always has 3 digits and indicates the actor ID
  // (index into ACTORINFO.MNI). The next 2 digits make up the second number,
  // which indicates the animation frame to draw for the specified actor's
  // sprite.
  //
  // If the text contains a byte >= 0xF0 at one point, the remaining text
  // will instead be drawn using a bigger font, which is also colorized using
  // the lower nibble of the markup byte as color index into the current
  // palette. E.g. if we have the text \xF7Hello, this will draw 'Hello'
  // using the big font colorized with palette index 7.
  // If there is other text preceding the 'big font' marker, it will be
  // drawn in the normal font.

  int x = 0;
  int y = 0;
  lineTextStream >> x;
  lineTextStream >> y;

  lineTextStream.get(); // skip one character of white-space

  string sourceText;
  getline(lineTextStream, sourceText, '\r');

  if (sourceText.empty())
  {
    throw invalid_argument("Corrupt Duke Script file");
  }

  vector<Action> textActions;
  if (static_cast<uint8_t>(sourceText[0]) == 0xEF)
  {
    textActions.emplace_back(parseDrawSpriteCommand(x, y, sourceText));
  }
  else
  {
    const auto bigTextMarkerIter =
      std::find_if(sourceText.begin(), sourceText.end(), [](const auto ch) {
        return static_cast<uint8_t>(ch) >= 0xF0;
      });

    if (bigTextMarkerIter != sourceText.end())
    {
      return parseTextCommandWithBigText(x, y, sourceText, bigTextMarkerIter);
    }
    else
    {
      textActions.emplace_back(DrawText{x, y, sourceText});
    }
  }

  return textActions;
}


vector<Action> parseCommand(
  const std::string& command,
  istream& sourceTextStream,
  istream& currentLineStream)
{
  vector<Action> actions;

  if (command == "MENU")
  {
    int slot = 0;
    currentLineStream >> slot;

    return {
      ConfigurePersistentMenuSelection{slot},
      ScheduleFadeInBeforeNextWaitState{}};
  }
  else if (command == "XYTEXT")
  {
    actions = parseTextCommand(currentLineStream);
  }
  else
  {
    const auto maybeAction =
      parseSingleActionCommand(command, currentLineStream);
    if (maybeAction)
    {
      actions.emplace_back(*maybeAction);
    }
  }

  return actions;
}


PagesDefinition parsePagesDefinition(istream& sourceTextStream)
{
  vector<data::script::Script> pages(1);
  parseScriptLines(
    sourceTextStream,
    "PAGESEND",
    [&pages, &sourceTextStream](const auto& command, auto& lineTextStream) {
      if (command == "APAGE")
      {
        pages.emplace_back(Script{});
      }
      else
      {
        const auto actions =
          parseCommand(command, sourceTextStream, lineTextStream);
        auto& currentPage = pages.back();
        currentPage.insert(currentPage.end(), actions.begin(), actions.end());
      }
    });

  return PagesDefinition{pages};
}


data::script::Script parseScript(istream& sourceTextStream)
{
  data::script::Script script;

  parseScriptLines(
    sourceTextStream,
    "END",
    [&script, &sourceTextStream](const auto& command, auto& lineTextStream) {
      vector<Action> actions;

      if (command == "PAGESSTART")
      {
        skipWhiteSpace(sourceTextStream);
        actions.emplace_back(std::make_shared<PagesDefinition>(
          parsePagesDefinition(sourceTextStream)));
      }
      else
      {
        actions = parseCommand(command, sourceTextStream, lineTextStream);
      }

      script.insert(script.end(), actions.begin(), actions.end());
    });

  const auto iMsgBoxAction =
    std::find_if(script.begin(), script.end(), [](const Action& action) {
      return std::holds_alternative<ShowMessageBox>(action);
    });

  const auto iMsgBoxTextAction =
    std::find_if(script.begin(), script.end(), [](const Action& action) {
      return std::holds_alternative<DrawMessageBoxText>(action);
    });

  if (iMsgBoxTextAction != script.end())
  {
    if (iMsgBoxAction == script.end() || iMsgBoxAction > iMsgBoxTextAction)
    {
      throw invalid_argument(
        "CWTEXT or SKLINE must be preceded by a CENTERWINDOW command");
    }
  }

  return script;
}


bool skipToHintsSection(istream& sourceTextStream)
{
  while (!sourceTextStream.eof())
  {
    skipWhiteSpace(sourceTextStream);

    string sectionName;
    sourceTextStream >> sectionName;
    strings::trim(sectionName);

    if (sectionName == "Hints")
    {
      skipWhiteSpace(sourceTextStream);
      return true;
    }
  }

  return false;
}

} // namespace


ScriptBundle loadScripts(const string& scriptSource)
{
  istringstream sourceStream(scriptSource);

  ScriptBundle bundle;
  while (!sourceStream.eof())
  {
    skipWhiteSpace(sourceStream);

    string scriptName;
    sourceStream >> scriptName;
    strings::trim(scriptName);

    if (!scriptName.empty())
    {
      bundle.emplace(scriptName, parseScript(sourceStream));
    }
  }

  return bundle;
}


data::LevelHints loadHintMessages(const std::string& scriptSource)
{
  istringstream sourceStream(scriptSource);

  const auto hintsFound = skipToHintsSection(sourceStream);
  if (!hintsFound)
  {
    return {};
  }

  std::vector<data::Hint> hints;

  for (string line; getline(sourceStream, line, '\r');)
  {
    skipWhiteSpace(sourceStream);

    if (!isCommand(line))
    {
      continue;
    }

    strings::trim(line);

    istringstream lineTextStream(line);

    string command;
    lineTextStream >> command;
    stripCommandPrefix(command);

    if (command == "END")
    {
      break;
    }

    if (command == "HELPTEXT")
    {
      int episode;
      int level;
      string message;

      lineTextStream >> episode;
      lineTextStream >> level;

      skipWhiteSpace(lineTextStream);

      getline(lineTextStream, message, '\r');
      hints.emplace_back(episode - 1, level - 1, move(message));
    }
  }

  return data::LevelHints{hints};
}

} // namespace rigel::assets
