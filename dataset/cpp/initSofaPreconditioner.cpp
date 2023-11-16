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
#include <SofaPreconditioner/initSofaPreconditioner.h>

#include <sofa/helper/system/PluginManager.h>

#include <sofa/core/ObjectFactory.h>
#include <string>

namespace sofa
{

namespace component
{

void initSofaPreconditioner()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaPreconditioner") << "SofaPreconditioner is deprecated. It will be removed at v23.06. Use Sofa.Component.LinearSolver.Preconditioner and Sofa.Component.LinearSolver.Iterative instead.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.LinearSolver.Preconditioner");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.LinearSolver.Iterative");

        first = false;
    }
}

extern "C" {
SOFA_PRECONDITIONER_API void initExternalModule();
SOFA_PRECONDITIONER_API const char* getModuleName();
SOFA_PRECONDITIONER_API const char* getModuleVersion();
SOFA_PRECONDITIONER_API const char* getModuleLicense();
SOFA_PRECONDITIONER_API const char* getModuleDescription();
SOFA_PRECONDITIONER_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaPreconditioner();
}

const char* getModuleName()
{
    return "SofaPreconditioner";
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
    return "This plugin contains preconditionners to accelerate the solving of linear systems.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // component

} // sofa




