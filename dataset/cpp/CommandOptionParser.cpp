/*
 * Copyright 2014-2023 Real Logic Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util/CommandOptionParser.h"

namespace aeron { namespace util
{

CommandOptionParser::CommandOptionParser()
{
    addOption(CommandOption(CommandOption::UNNAMED, 0, 0, "Unnamed Options"));
}

void CommandOptionParser::parse(int argc, char **argv)
{
    char currentOption = CommandOption::UNNAMED;
    getOption(currentOption).setPresent();

    for (int n = 1; n < argc; n++)
    {
        std::string argStr(argv[n]);

        if ((argStr.size() >= 2) && (argStr[0] == '-'))
        {
            for (std::size_t argNum = 1; argNum < argStr.size(); argNum++)
            {
                currentOption = argStr[argNum];

                auto opt = m_options.find(currentOption);
                if (m_options.end() == opt)
                {
                    throw CommandOptionException(
                        std::string("-") + currentOption + " is not a valid command option", SOURCEINFO);
                }
                else
                {
                    opt->second.setPresent();
                }
            }
        }
        else
        {
            CommandOption &opt = getOption(currentOption);
            opt.addParam(argStr);
        }
    }

    for (auto &option : m_options)
    {
        option.second.validate();
    }
}

void CommandOptionParser::addOption(const CommandOption &option)
{
    m_options[option.getOptionChar()] = option;
}

CommandOption &CommandOptionParser::getOption(char optionChar)
{
    auto opt = m_options.find(optionChar);

    if (m_options.end() == opt)
    {
        throw CommandOptionException(
            std::string("CommandOptionParser::getOption invalid option lookup: ") + optionChar, SOURCEINFO);
    }

    return opt->second;
}

void CommandOptionParser::displayOptionsHelp(std::ostream &out) const
{
    for (const auto &opt : m_options)
    {
        if (opt.first != CommandOption::UNNAMED)
        {
            out << "    -" << opt.first << " " << opt.second.getHelpText() << std::endl;
        }
    }
}

}}
