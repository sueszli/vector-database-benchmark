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
#include <SofaSimpleGUI/config.h>

namespace sofa
{

namespace component
{

	//Here are just several convenient functions to help user to know what contains the plugin

	extern "C" {
                SOFA_SOFASIMPLEGUI_API void initExternalModule();
                SOFA_SOFASIMPLEGUI_API const char* getModuleName();
                SOFA_SOFASIMPLEGUI_API const char* getModuleVersion();
                SOFA_SOFASIMPLEGUI_API const char* getModuleLicense();
                SOFA_SOFASIMPLEGUI_API const char* getModuleDescription();
                SOFA_SOFASIMPLEGUI_API const char* getModuleComponentList();
	}
	
	void initExternalModule()
	{
		static bool first = true;
		if (first)
		{
			first = false;
		}
	}

	const char* getModuleName()
	{
	  return "SimpleGUI";
	}

	const char* getModuleVersion()
	{
		return "0.2";
	}

	const char* getModuleLicense()
	{
		return "LGPL";
	}


	const char* getModuleDescription()
	{
		return "TODO: replace this with the description of your plugin";
	}

	const char* getModuleComponentList()
	{
	  /// string containing the names of the classes provided by the plugin
	  return "";
	  //return "MyMappingPendulumInPlane, MyBehaviorModel, MyProjectiveConstraintSet";
	}



} 

} 
