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
#include "util/StringUtil.h"

namespace aeron { namespace util
{

CommandOption::CommandOption(char optionChar, std::size_t minParams, std::size_t maxParams, std::string helpText) :
    m_optionChar(optionChar),
    m_minParams(minParams),
    m_maxParams(maxParams),
    m_helpText(std::move(helpText))
{
}

void CommandOption::checkIndex(std::size_t index) const
{
    if (index > m_params.size())
    {
        throw CommandOptionException(
            std::string("Internal Error: index out of range for option: ") + m_optionChar, SOURCEINFO);
    }
}

std::string CommandOption::getParam(std::size_t index) const
{
    checkIndex(index);
    return m_params[index];
}

std::string CommandOption::getParam(std::size_t index, std::string defaultValue) const
{
    if (!isPresent())
    {
        return defaultValue;
    }

    return getParam(index);
}

int CommandOption::getParamAsInt(std::size_t index) const
{
    checkIndex(index);
    std::string param = m_params[index];

    try
    {
        return parse<int>(param);
    }
    catch (const ParseException &)
    {
        throw CommandOptionException(
            std::string("invalid numeric value: \"") + param + "\" on option -" + m_optionChar, SOURCEINFO);
    }
}

long long CommandOption::getParamAsLong(std::size_t index) const
{
    checkIndex(index);
    std::string param = m_params[index];

    try
    {
        return parse<long long>(param);
    }
    catch (const ParseException &)
    {
        throw CommandOptionException(
            std::string("invalid numeric value: \"") + param + "\" on option -" + m_optionChar, SOURCEINFO);
    }
}

int CommandOption::getParamAsInt(std::size_t index, int minValue, int maxValue, int defaultValue) const
{
    if (!isPresent())
    {
        return defaultValue;
    }

    int value = getParamAsInt(index);
    if (value < minValue || value > maxValue)
    {
        throw CommandOptionException(
            std::string("value \"") + toString(value) + "\" out of range: [" +
                toString(minValue) + ".." + toString(maxValue) + "] on option -" + m_optionChar,
            SOURCEINFO);
    }

    return value;
}

long long CommandOption::getParamAsLong(
    std::size_t index, long long minValue, long long maxValue, long long defaultValue) const
{
    if (!isPresent())
    {
        return defaultValue;
    }

    long long value = getParamAsLong(index);
    if (value < minValue || value > maxValue)
    {
        throw CommandOptionException(
            std::string("value \"") + toString(value) + "\" out of range: [" +
                toString(minValue) + ".." + toString(maxValue) + "] on option -" + m_optionChar,
            SOURCEINFO);
    }

    return value;
}

void CommandOption::validate() const
{
    if (!isPresent())
    {
        return;
    }

    if (m_params.size() > m_maxParams)
    {
        throw CommandOptionException(
            std::string("option -") + m_optionChar + " has too many parameters specified.", SOURCEINFO);
    }

    if (m_params.size() < m_minParams)
    {
        throw CommandOptionException(
            std::string("option -") + m_optionChar + " has too few parameters specified.", SOURCEINFO);
    }
}

}}
