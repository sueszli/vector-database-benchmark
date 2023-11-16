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
#include <SofaMiscTopology/initSofaMiscTopology.h>

#include <sofa/helper/system/PluginManager.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa::component
{

void initSofaMiscTopology()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaMiscTopology") << "SofaMiscTopology is deprecated. It will be removed at v23.06. Use Sofa.Component.Topology.Utility instead.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Topology.Utility");

        first = false;
    }
}

extern "C" {
    SOFA_SOFAMISCTOPOLOGY_API void initExternalModule();
    SOFA_SOFAMISCTOPOLOGY_API const char* getModuleName();
    SOFA_SOFAMISCTOPOLOGY_API const char* getModuleVersion();
    SOFA_SOFAMISCTOPOLOGY_API const char* getModuleLicense();
    SOFA_SOFAMISCTOPOLOGY_API const char* getModuleDescription();
    SOFA_SOFAMISCTOPOLOGY_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaMiscTopology();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(SOFAMISCTOPOLOGY_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains contains features about Misc Topologies.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace sofa::component
