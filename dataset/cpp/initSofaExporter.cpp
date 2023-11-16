/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <string>
#include <SofaExporter/initSofaExporter.h>

#include <sofa/helper/system/PluginManager.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Node.h>


using sofa::core::ObjectFactory;

namespace sofa
{

namespace component
{

void initSofaExporter()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaExporter") << "SofaExporter is deprecated. It will be removed at v23.06. Use Sofa.Component.IO.Mesh and/or Sofa.Component.Playback instead.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.IO.Mesh");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Playback");

        first = false;
    }
}

extern "C" {
SOFA_SOFAEXPORTER_API void initExternalModule();
SOFA_SOFAEXPORTER_API const char* getModuleName();
SOFA_SOFAEXPORTER_API const char* getModuleVersion();
SOFA_SOFAEXPORTER_API const char* getModuleLicense();
SOFA_SOFAEXPORTER_API const char* getModuleDescription();
SOFA_SOFAEXPORTER_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaExporter();
}

const char* getModuleName()
{
    return "SofaExporter";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains some exporter to save simulation scenes to various formats. "
            "Supported format are: Sofa internal state format, VTK, STL, Mesh, Blender.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace component

} // namespace sofa
