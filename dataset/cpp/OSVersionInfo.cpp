/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#include <aws/core/platform/OSVersionInfo.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/StringUtils.h>

#include <regex>

namespace Aws
{
namespace OSVersionInfo
{

Aws::String GetSysCommandOutput(const char* command)
{
    Aws::String outputStr;
    FILE* outputStream;
    const int maxBufferSize = 256;
    char outputBuffer[maxBufferSize];

    outputStream = popen(command, "r");

    if (outputStream)
    {
        while (!feof(outputStream))
        {
            if (fgets(outputBuffer, maxBufferSize, outputStream) != nullptr)
            {
                outputStr.append(outputBuffer);
            }
        }

        pclose(outputStream);

        return Aws::Utils::StringUtils::Trim(outputStr.c_str());
    }

    return {};
}

Aws::String ComputeOSVersionString()
{
    // regex is not allocator-aware, so technically we're breaking our memory contract here (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3254.pdf)
    // Since it's internal, nothing escapes, and what gets allocated/deallocated is very small, I think that's acceptable for now

    std::string androidBuildVersion = GetSysCommandOutput("cat /proc/version 2>&1").c_str();
    std::regex versionRegex("version (\\S+)\\s");

    std::smatch versionMatchResults;
    std::regex_search(androidBuildVersion, versionMatchResults, versionRegex);

    if(versionMatchResults.size() >= 2)
    {
        return Aws::String("Android#") + versionMatchResults[1].str().c_str();
    }

    return Aws::String("Android");
}

Aws::String ComputeOSVersionArch()
{
    // TODO
    return "";
}

} // namespace OSVersionInfo
} // namespace Aws
