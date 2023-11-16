/*****************************************************************************
 * Copyright (c) 2014-2023 OpenRCT2 developers
 *
 * For a complete list of all authors, please refer to contributors.md
 * Interested in contributing? Visit https://github.com/OpenRCT2/OpenRCT2
 *
 * OpenRCT2 is licensed under the GNU General Public License version 3.
 *****************************************************************************/

#include "ParkImporter.h"

#include "Context.h"
#include "core/Path.hpp"
#include "core/String.hpp"
#include "object/ObjectManager.h"
#include "object/ObjectRepository.h"

#include <memory>

namespace ParkImporter
{
    std::unique_ptr<IParkImporter> Create(const std::string& hintPath)
    {
        std::unique_ptr<IParkImporter> parkImporter;
        std::string extension = Path::GetExtension(hintPath);
        auto* context = OpenRCT2::GetContext();
        if (ExtensionIsOpenRCT2ParkFile(extension))
        {
            parkImporter = CreateParkFile(context->GetObjectRepository());
        }
        else if (ExtensionIsRCT1(extension))
        {
            parkImporter = CreateS4();
        }
        else
        {
            parkImporter = CreateS6(context->GetObjectRepository());
        }
        return parkImporter;
    }

    bool ExtensionIsOpenRCT2ParkFile(std::string_view extension)
    {
        return String::IEquals(extension, ".park");
    }

    bool ExtensionIsRCT1(std::string_view extension)
    {
        return String::IEquals(extension, ".sc4") || String::IEquals(extension, ".sv4");
    }

    bool ExtensionIsScenario(std::string_view extension)
    {
        return String::IEquals(extension, ".sc4") || String::IEquals(extension, ".sc6") || String::IEquals(extension, ".sea");
    }
} // namespace ParkImporter
