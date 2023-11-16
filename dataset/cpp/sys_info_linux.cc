//
// Aspia Project
// Copyright (C) 2016-2023 Dmitry Chapyshev <dmitry@aspia.ru>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "base/sys_info.h"

#include "base/logging.h"
#include "base/files/file_util.h"

#include <sys/param.h>
#include <sys/sysinfo.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace base {

//--------------------------------------------------------------------------------------------------
//static
std::string SysInfo::operatingSystemName()
{
    struct utsname info;
    if (uname(&info) < 0)
        return std::string();

    return std::string(info.sysname);
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::operatingSystemVersion()
{
    struct utsname info;
    if (uname(&info) < 0)
        return std::string();

    return std::string(info.release);
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::operatingSystemArchitecture()
{
    struct utsname info;
    if (uname(&info) < 0)
        return std::string();

    std::string arch(info.machine);
    if (arch == "i386" || arch == "i486" || arch == "i586" || arch == "i686")
    {
        arch = "x86";
    }
    else if (arch == "amd64")
    {
        arch = "x86_64";
    }
    else if (std::string(info.sysname) == "AIX")
    {
        arch = "ppc64";
    }

    return arch;
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::operatingSystemDir()
{
    NOTIMPLEMENTED();
    return std::string();
}

//--------------------------------------------------------------------------------------------------
// static
uint64_t SysInfo::uptime()
{
    struct sysinfo info;
    if (sysinfo(&info) < 0)
        return 0;

    return info.uptime;
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::computerName()
{
    char buffer[256];
    if (gethostname(buffer, std::size(buffer)) < 0)
        return std::string();

    return std::string(buffer);
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::computerDomain()
{
    NOTIMPLEMENTED();
    return std::string();
}

//--------------------------------------------------------------------------------------------------
// static
std::string SysInfo::computerWorkgroup()
{
    NOTIMPLEMENTED();
    return std::string();
}

//--------------------------------------------------------------------------------------------------
// static
int SysInfo::processorPackages()
{
    std::string contents;
    if (!readFile("/proc/cpuinfo", &contents) || contents.empty())
        return 0;

    const char kCpuModelPrefix[] = "model name";
    std::istringstream iss(contents);
    std::string line;
    int modelNameCount = 0;

    while (std::getline(iss, line))
    {
        if (line.compare(0, strlen(kCpuModelPrefix), kCpuModelPrefix) == 0)
            ++modelNameCount;
    }

    return modelNameCount;
}

//--------------------------------------------------------------------------------------------------
// static
int SysInfo::processorCores()
{
    NOTIMPLEMENTED();
    return 0;
}

//--------------------------------------------------------------------------------------------------
// static
int SysInfo::processorThreads()
{
    long res = sysconf(_SC_NPROCESSORS_CONF);
    if (res == -1)
        return 1;

    return static_cast<int>(res);
}

} // namespace base
